// The upstream wires the starter does not speak: Anthropic and Gemini.
//
// The trial's client calls this server as OpenAI (POST /v1/chat/completions) or Anthropic
// (POST /v1/messages). The ten providers speak three wires: openai, anthropic and gemini. The
// starter only calls openai-wire providers. This file translates the other two, both ways, so
// every free model can serve a request:
//
//   OpenAI client -> Anthropic provider: chat completions rewritten as a Messages request.
//   OpenAI client -> Gemini provider:   chat completions rewritten as generateContent.
//   Anthropic client -> Gemini provider: Messages rewritten as generateContent.
//   Anthropic client -> Anthropic provider: passed through (only the model id changes).
//
// Every translation preserves the fields the provider echoes back (system prompt, user text,
// images, tools, stop sequences, max_tokens, JSON mode), because the client compares that echo
// with what it sent: a field rewritten or dropped loses the request. The provider's answer text
// is returned whole, its usage counts as that wire reports them, and its stop reason mapped to
// the client's wire. Everything here is a pure function of its arguments: no clock, no
// randomness, so the same body and the same provider answer give the same bytes on a replay.

import { isRecord, type Json } from './providers.ts';
import { sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for translated OpenAI completions. */
const FIXED_OPENAI_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a Gemini parts array. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** One OpenAI content part as an Anthropic content block. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (dataUrl !== null) return { type: 'image', source: { type: 'base64', media_type: dataUrl[1], data: dataUrl[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic image block as a Gemini inline-data or file-data part. */
function geminiImagePartOf(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** One OpenAI image_url part as a Gemini inline-data or file-data part. */
function geminiImagePartOfOpenAI(part: Json): Json | null {
  if (!isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } };
  return { fileData: { fileUri: url } };
}

/** A stop value (string or array) as an array of strings. */
function stopArray(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === 'string');
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'user') {
      if (typeof raw.content === 'string') {
        messages.push({ role: 'user', content: raw.content });
        continue;
      }
      if (Array.isArray(raw.content)) {
        const blocks: Json[] = [];
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const block = anthropicBlockOf(part);
          if (block !== null) blocks.push(block);
        }
        if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
      }
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : '' }],
      });
      continue;
    }
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopArray(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The system text of an OpenAI request: its system messages' text, joined. */
function openAISystemText(body: Json): string {
  const parts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** An OpenAI request's messages as Gemini contents (system handled separately). */
function openAIContents(body: Json): Json[] {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url') {
            const image = geminiImagePartOfOpenAI(part);
            if (image !== null) parts.push(image);
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : '';
      contents.push({ role: 'function', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response: { content: text } } }] });
      continue;
    }
  }
  return contents;
}

/** An OpenAI request's tools as Gemini function declarations. */
function openAITools(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** An OpenAI Chat Completions request as a Gemini generateContent request (model goes in the URL). */
export function openAIToGeminiRequest(body: Json): Json {
  const out: Json = {};
  const system = openAISystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  out.contents = openAIContents(body);
  const declarations = openAITools(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopArray(body.stop);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The system text of an Anthropic request: its system prompt's text. */
function anthropicSystemText(body: Json): string {
  return textOf(body.system);
}

/** An Anthropic Messages request as a Gemini generateContent request (model goes in the URL). */
export function anthropicToGeminiRequest(body: Json): Json {
  const out: Json = {};
  const system = anthropicSystemText(body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image') {
            const image = geminiImagePartOf(block);
            if (image !== null) parts.push(image);
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopArray(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** Gemini's finishReason as an OpenAI finish_reason. */
function geminiFinishToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
function geminiFinishToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The answer text of a Gemini response: its first candidate's parts, joined whole. */
export function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return '';
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null) return '';
  return geminiPartsText(content.parts);
}

/** The usage of a Gemini response: its usageMetadata, or zeros. */
export function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0;
  const output = typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0;
  return { input, output };
}

/** A Gemini response's finish reason: its first candidate's finishReason. */
function geminiFinishOf(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  return candidate === null ? null : candidate.finishReason ?? null;
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const text = geminiTextOf(body);
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: geminiFinishToOpenAI(geminiFinishOf(body)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const text = geminiTextOf(body);
  const usage = geminiUsageOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiFinishToAnthropic(geminiFinishOf(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: stopReasonOf(body.stop_reason) === 'tool_use' ? 'tool_calls' : body.stop_reason === 'max_tokens' ? 'length' : body.stop_reason === 'refusal' ? 'content_filter' : 'stop', logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One SSE frame: its event name (null when unnamed) and its data text. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Read a ReadableStream of SSE text as frames, split on blank lines. */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const raw = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        let event: string | null = null;
        const dataLines: string[] = [];
        for (const line of raw.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
        }
        if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of buffer.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
      }
      if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
    }
  } finally {
    reader.releaseLock();
  }
}

/** An Anthropic SSE stream as OpenAI Chat Completions chunks. `model` is named back to the client. */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  let textStarted = false;
  const toolArgs = new Map<number, string>();
  for await (const frame of readSseFrames(upstream)) {
    if (frame.data.trim() === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage) && typeof data.message.usage.input_tokens === 'number') {
        inputTokens = data.message.usage.input_tokens;
      }
      if (!started) {
        started = true;
        yield `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
      }
      continue;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        if (!started) {
          started = true;
          yield `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
        }
        textStarted = true;
        yield `data: ${JSON.stringify(chunk({ content: data.delta.text }, null))}\n\n`;
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        const index = typeof data.index === 'number' ? data.index : 0;
        toolArgs.set(index, (toolArgs.get(index) ?? '') + data.delta.partial_json);
        yield `data: ${JSON.stringify(chunk({ tool_calls: [{ index, function: { arguments: data.delta.partial_json } }] }, null))}\n\n`;
      }
      continue;
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (!started) {
        started = true;
        yield `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
      }
      yield `data: ${JSON.stringify(chunk({ tool_calls: [{ index, id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`, type: 'function', function: { name: typeof data.content_block.name === 'string' ? data.content_block.name : '', arguments: '' } }] }, null))}\n\n`;
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') {
        stopReason = data.delta.stop_reason === 'max_tokens' ? 'length' : data.delta.stop_reason === 'tool_use' ? 'tool_calls' : data.delta.stop_reason === 'refusal' ? 'content_filter' : 'stop';
      }
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
      continue;
    }
    if (frame.event === 'message_stop') {
      void textStarted;
      break;
    }
  }
  if (!started) {
    yield `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
  }
  yield `data: ${JSON.stringify(chunk({}, stopReason))}\n\n`;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** A Gemini SSE stream as OpenAI Chat Completions chunks. The whole stream is read first (it is small). */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  let text = '';
  let usage = { input: 0, output: 0 };
  let finish: unknown = null;
  for await (const frame of readSseFrames(upstream)) {
    if (frame.data.trim() === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    text += geminiTextOf(data);
    const u = geminiUsageOf(data);
    if (u.input !== 0 || u.output !== 0) usage = u;
    const f = geminiFinishOf(data);
    if (f !== null && f !== undefined) finish = f;
  }
  yield `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null))}\n\n`;
  if (text !== '') yield `data: ${JSON.stringify(chunk({ content: text }, null))}\n\n`;
  yield `data: ${JSON.stringify(chunk({}, geminiFinishToOpenAI(finish)))}\n\n`;
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** A Gemini SSE stream as Anthropic Messages events. The whole stream is read first (it is small). */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let text = '';
  let usage = { input: 0, output: 0 };
  let finish: unknown = null;
  for await (const frame of readSseFrames(upstream)) {
    if (frame.data.trim() === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    text += geminiTextOf(data);
    const u = geminiUsageOf(data);
    if (u.input !== 0 || u.output !== 0) usage = u;
    const f = geminiFinishOf(data);
    if (f !== null && f !== undefined) finish = f;
  }
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: geminiFinishToAnthropic(finish), stop_sequence: null }, usage: { output_tokens: usage.output } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
