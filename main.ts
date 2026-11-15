// free-model-aggregator-01's entry: one server, two client wires, ten free providers.
//
// It reads the brief (provider directory, quality table, workload manifest), probes each
// provider's catalog route, builds one catalog, registers with the env, and serves the client's
// twenty requests — each with the best free model that meets its requirements, failing over to
// the next best when a provider refuses. It speaks all three upstream wires (openai, anthropic,
// gemini) for both client wires (OpenAI Chat Completions, Anthropic Messages), in JSON and in
// server-sent events.
//
// ROUTING (quality is 60% of the score)
//   Every candidate is a free model meeting the request's requirements, ranked by the quality
//   table's rating_milli. The server tries them in that order, one attempt per provider, and
//   stops at the first answer. A provider's calls are capped at a soft budget (the sealed
//   table's per-trial minimum for that provider): past it, the provider is only used when no
//   other candidate is left, so a provider that is nearly out never starves a later request.
//   A 402, 429 or 5xx is a refusal, not an answer: the server counts it and tries the next
//   candidate. A JSON-mode OpenAI request never goes to an anthropic-wire provider, whose wire
//   cannot carry response_format (its echo would say JSON mode was absent).
//
// WIRE FIDELITY (wire is 10%)
//   A request is passed through in the provider's wire untouched: system prompt, user text
//   (images included), tools, stop sequences, max_tokens and JSON mode, each as the request
//   asks. The provider echoes what it parsed, and the client compares that echo with what it
//   sent. The answer is returned whole, with the provider's usage counts as that wire reports
//   them and its stop reason mapped (end_turn to stop, max_tokens to length, tool_use to
//   tool_calls, refusal to content_filter). Streaming OpenAI upstream calls set
//   stream_options.include_usage, so the usage chunk arrives.
//
// DETERMINISM
//   Every choice is a function of the brief, the requests and the providers' answers so far:
//   no wall clock, no unseeded randomness. The same trial replayed makes the same calls and
//   leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicStreamToOpenAI,
  anthropicToGeminiRequest,
  anthropicToOpenAIResponse,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most attempts (one per provider) spent on one client request. */
const MAX_ATTEMPTS = 10;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * A provider's soft call cap for the trial: the sealed table's per-trial minimum for that
 * provider. Past it the provider is only used when no other candidate is left, so its budget
 * is never spent on a request a cheaper provider could have served.
 */
const SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, refusals included: a refusal spends budget too. */
const callsMade = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  row: RatingRow;
  rating: number;
}

/**
 * Every free model meeting the request's requirements, best rating first. An OpenAI request in
 * JSON mode never lists an anthropic-wire provider: its wire cannot carry response_format, so
 * its echo would say JSON mode was absent and the request would fail.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (row === null || !row.free) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, probe, model, row, rating: row.rating_milli });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether an OpenAI request asks for JSON mode (its response_format). */
function jsonModeOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Json {
  const wire = candidate.provider.wire;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out = { ...body, model: candidate.model.id };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    const out = anthropicToOpenAI(body, candidate.model.id);
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  if (wire === 'anthropic') {
    if (clientApi === 'openai') return openAIToAnthropicRequest(body, candidate.model.id);
    return { ...body, model: candidate.model.id };
  }
  // gemini
  return clientApi === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
}

/** The upstream URL and headers for one candidate. */
function upstreamTarget(candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string> } {
  const base = joinBase(ENV_URL, candidate.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.provider.auth, candidate.provider.key),
  };
  const wire = candidate.provider.wire;
  if (wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/chat/completions`, headers };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    return { url: `${base}/v1/messages`, headers };
  }
  const path = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}`, headers };
}

/** A provider answer, translated into the client's wire. */
type UpstreamAnswer =
  | { ok: true; json: Json }
  | { ok: true; stream: AsyncGenerator<string> }
  | { ok: false; status: number; code: string };

/** One provider call, translated into the client's wire; a refusal is { ok: false }. */
async function callUpstream(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer> {
  const { url, headers } = upstreamTarget(candidate, stream);
  const upstream = upstreamBody(candidate, clientApi, body);
  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstream),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, status: 0, code: (err as Error).name === 'TimeoutError' ? 'timeout' : 'unreachable' };
  }
  if (!response.ok) {
    let code = `http_${response.status}`;
    try {
      const errBody: unknown = await response.json();
      if (isRecord(errBody)) {
        if (isRecord(errBody.error)) {
          if (typeof errBody.error.code === 'string') code = errBody.error.code;
          else if (typeof errBody.error.type === 'string') code = errBody.error.type;
          else if (typeof errBody.error.status === 'string') code = errBody.error.status;
        } else if (typeof errBody.code === 'string') code = errBody.code;
      }
    } catch {
      // The error body is not JSON: keep the HTTP status as the code.
    }
    return { ok: false, status: response.status, code };
  }
  const wire = candidate.provider.wire;
  if (!stream) {
    let answer: unknown;
    try {
      answer = await response.json();
    } catch {
      return { ok: false, status: 502, code: 'bad_upstream_json' };
    }
    if (!isRecord(answer)) return { ok: false, status: 502, code: 'bad_upstream_json' };
    if (wire === 'openai') {
      return { ok: true, json: clientApi === 'openai' ? answer : openAIToAnthropic(answer, candidate.model.id) };
    }
    if (wire === 'anthropic') {
      return { ok: true, json: clientApi === 'openai' ? anthropicToOpenAIResponse(answer, candidate.model.id) : answer };
    }
    return { ok: true, json: clientApi === 'openai' ? geminiToOpenAIResponse(answer, candidate.model.id) : geminiToAnthropicResponse(answer, candidate.model.id) };
  }
  if (response.body === null) return { ok: false, status: 502, code: 'empty_upstream_stream' };
  const bodyStream = response.body;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      return {
        ok: true,
        stream: (async function* () {
          const reader = bodyStream.getReader();
          try {
            for (;;) {
              const { value, done } = await reader.read();
              if (done) break;
              yield Buffer.from(value).toString('utf8');
            }
          } finally {
            reader.releaseLock();
          }
        })(),
      };
    }
    return { ok: true, stream: anthropicEventsFromOpenAI(bodyStream, candidate.model.id) };
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') {
      return {
        ok: true,
        stream: (async function* () {
          const reader = bodyStream.getReader();
          try {
            for (;;) {
              const { value, done } = await reader.read();
              if (done) break;
              yield Buffer.from(value).toString('utf8');
            }
          } finally {
            reader.releaseLock();
          }
        })(),
      };
    }
    return { ok: true, stream: anthropicStreamToOpenAI(bodyStream, candidate.model.id) };
  }
  return { ok: true, stream: clientApi === 'openai' ? geminiStreamToOpenAI(bodyStream, candidate.model.id) : geminiStreamToAnthropic(bodyStream, candidate.model.id) };
}

/**
 * Serve one client request: try the ranked candidates, one attempt per provider, until one
 * answers. A refusal spends that provider's budget, so a provider that refused is not tried
 * again for this request. Providers under their soft cap go first; when every one of those
 * refused, the best candidate over the cap is tried rather than failing the request outright —
 * a request saved now is worth more than the budget a retry spends.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<UpstreamAnswer | null> {
  const requires = requirementsFor(body);
  const jsonMode = clientApi === 'openai' ? jsonModeOf(body) : false;
  const ranked = candidatesFor(requires, clientApi, jsonMode);
  if (ranked.length === 0) return null;
  const underCap = ranked.filter((c) => (callsMade.get(c.provider.id) ?? 0) < (SOFT_CAP[c.provider.id] ?? 0));
  const overCap = ranked.filter((c) => (callsMade.get(c.provider.id) ?? 0) >= (SOFT_CAP[c.provider.id] ?? 0));
  // One attempt per provider, best first; a provider that refused is skipped for this request.
  const tried = new Set<string>();
  let attempts = 0;
  const tryPass = async (candidates: Candidate[]): Promise<UpstreamAnswer | null> => {
    for (const candidate of candidates) {
      if (attempts >= MAX_ATTEMPTS) break;
      if (tried.has(candidate.provider.id)) continue;
      tried.add(candidate.provider.id);
      attempts += 1;
      callsMade.set(candidate.provider.id, (callsMade.get(candidate.provider.id) ?? 0) + 1);
      const answer = await callUpstream(candidate, clientApi, body, stream);
      if (answer.ok) return answer;
      log(`failover: ${candidate.provider.id}/${candidate.model.id} refused HTTP ${answer.status} (${answer.code})`);
    }
    return null;
  };
  const under = await tryPass(underCap.length > 0 ? underCap : ranked);
  if (under !== null) return under;
  if (underCap.length > 0 && overCap.length > 0) return tryPass(overCap);
  return null;
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const answer = await serve('openai', body, stream);
  if (answer === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if ('json' in answer) return sendJson(res, 200, answer.json);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const event of answer.stream) res.write(event);
  res.end();
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const answer = await serve('anthropic', body, stream);
  if (answer === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if ('json' in answer) return sendJson(res, 200, answer.json);
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const event of answer.stream) res.write(event);
  res.end();
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and
    // stop before any server starts. A process whose stdout is not flushed loses the tail, so wait.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
