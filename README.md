# Free Model Aggregator

One server that answers OpenAI Chat Completions and Anthropic Messages over ten simulated
free-tier providers, picking the best free model for each request and failing over when a
provider refuses.

Built for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01) — "The best free
AI model aggregator".

- **Owner:** [@kingcritket](https://make.fun/u/kingcritket) on make.fun
- **Referral:** use code `97vd661v` when you sign up at
  [make.fun](https://make.fun/r/97vd661v) — it helps the author keep entering bounties.
- **Runtime:** Node.js 22, standard library only. Nothing to install.

## What it does

The server is one process, `main.ts`. It:

- Reads the brief (provider directory, quality table, workload manifest) and probes each
  provider's catalog route, so its catalog is what the providers actually answer.
- Serves `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages` (Anthropic wire),
  in JSON and in server-sent events.
- Answers `GET /v1/models` in the shape of the auth header sent — a Bearer token gets the
  OpenAI list, an `x-api-key` gets the Anthropic list.
- Registers its port and catalog with the env, serves the workload's twenty requests one at a
  time, polls `GET /work`, and exits 0 when the workload says done.

## Routing: the best free model, then the next

Every candidate is a free model that meets the request's requirements — tools, vision, JSON
mode, and a minimum context length — ranked by the brief's quality rating. The server tries
them in that order, one attempt per provider, and stops at the first answer.

A provider's calls are capped at a soft budget (its per-trial minimum from the brief's sealed
table). Past it, the provider is only used when no other candidate is left, so a provider
that is nearly out never starves a later request. A 402, 429 or 5xx is a refusal, not an
answer: the server counts it against that provider's budget and tries the next candidate.

A JSON-mode OpenAI request never goes to an anthropic-wire provider, whose wire cannot carry
`response_format` — its echo would say JSON mode was absent and the request would fail.

## Wire fidelity

A request is passed through in the provider's wire untouched: system prompt, user text
(images included), tools, stop sequences, `max_tokens` and JSON mode, each as the request
asks. The provider echoes what it parsed, and the client compares that echo with what it
sent, so nothing is dropped or reworded. The answer is returned whole, with the provider's
usage counts as that wire reports them and its stop reason mapped (`end_turn` to `stop`,
`max_tokens` to `length`, `tool_use` to `tool_calls`, `refusal` to `content_filter`).
Streaming OpenAI upstream calls set `stream_options.include_usage`, so the usage chunk
arrives and the client can verify it.

The server speaks all three upstream wires — OpenAI, Anthropic, and Gemini — for both client
wires, translating requests and responses between them. That matters: no OpenAI-wire
provider has a model with both tools and vision, so the two requests that need both go over
the Gemini wire (gemini-3.8-flash) or the Anthropic wire (zai glm-4.6v-flash).

## Determinism

Every choice is a function of the brief, the requests and the providers' answers so far: no
wall clock, no unseeded randomness. The same trial replayed makes the same calls and leaves
the same state export.

## Files

- `main.ts` — the server: catalog, registration, routing, failover, both client wires.
- `lib/providers.ts` — brief reading, provider probing, catalog building, capability checks.
- `lib/wire.ts` — the OpenAI and Anthropic client wires (lists, errors, conversions).
- `lib/translate.ts` — translations between the OpenAI, Anthropic and Gemini upstream wires,
  for JSON and for streams.

## Run it

```sh
node main.ts            # serves the workload; exits 0 when it is done
node main.ts --list     # prints the catalog, byte-for-byte as registered
```

`MAKEFUN_ENV_URL` must point at the trial env, and `MAKEFUN_PROMPT_FILE` at the brief
(`prompt.txt`).
