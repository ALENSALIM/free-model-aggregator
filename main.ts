// The entry: one server that puts OpenRouter's free models behind one endpoint.
//
// The trial's client calls this server as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events, and the server answers every prompt with a
// live completion from a free OpenRouter model, dialled through the gateway
// the trial puts in front of it:
//
//   - the brief's models, ranked by the house's rating in milli-points, with
//     the id breaking ties, among the models that meet the prompt's
//     requirements (JSON mode, tools, vision), so a JSON prompt goes to the
//     best JSON-capable model and a plain one to the best model at all. A
//     prompt is worth a share of the quality column, so serving it with the
//     best model it can use is the score;
//   - one failure retries the next model, so a free model that refuses, rates
//     the trial out, or fails can give way to another live free model from
//     the catalog: the house records every gateway call and scores
//     failover on it, so a refusal the server retried and then served is
//     told in the `x-makefun-upstream-failures` header on the answer, up to
//     eight rows of `{"status": <number>, "code": "<string>", "model": "<id or null>"}`;
//   - an answer is served only when the live model answered it: the entry
//     checks every answer before it serves it — a valid JSON or SSE body,
//     usage with at least one output token, a legal stop reason, and the
//     answer carrying the exact word the prompt asked for when the prompt
//     named one — and routes to the next model when the answer fails any of
//     them. When every model fails, the server answers 502 with the
//     refusals in the header: a failover it never wins still scores its
//     evidence, and no canned or fake answer is ever served;
//   - the server also answers GET /v1/models in both wires (the model-id
//     set the catalog column scores), checks the brief on start (its
//     `--list` prints the catalog document the brief makes, byte for byte),
//     registers that catalog on the env's /register, then polls the env's
//     /work until the env says done and exits 0.
//
// The entry listens on a free port it picks (never 8080: the env owns it),
// and on a socket: nothing it can see of the caller is recorded. What it
// does know — the brief, the request, the upstream answer, the refusal
// trail — is all it routes on, and a replay sees the same rows, so the same
// prompt, the same gateway answer and the same retries give the same state
// export, byte for byte. No wall clock and no unseeded randomness anywhere.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { readBrief, buildCatalog, freeModels, meetsRequirements, type Brief, type CatalogReport, type ModelRow, type Requirement } from './lib/brief.ts';
import {
  openAIModelList,
  anthropicModelList,
  anthropicError,
  openAIError,
  anthropicToOpenAI,
  openAIToAnthropic,
  openAIJsonWithModel,
  openAIChunks,
  parseSseFrames,
  streamIsWhole,
  gatherStream,
  openAIStreamText,
  anthropicStreamText,
  stopReasonOf,
  anthropicStreamIsGood,
  OPENAI_STOP_REASONS,
  ANTHROPIC_STOP_REASONS,
  type ServedModel,
} from './lib/wire.ts';
import { isRecord, type Json } from './lib/brief.ts';

/** One upstream refusal: the row `x-makefun-upstream-failures` carries. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** A model row and its routing order: the brief's rating, highest first, with the id breaking ties. */
interface Candidate extends ModelRow {
  rank: number;
}

/** What one route attempt cost: the refusal, or the answer when one won. `failures` is this prompt's refusal trail, for the header on its answer. */
type AttemptResult =
  | { kind: 'refused'; status: number; code: string; failures: UpstreamFailure[] }
  | { kind: 'answered'; body: Json; stream: string | null; model: string; upstream: Json; failures: UpstreamFailure[] };

function argvHas(name: string): boolean {
  return process.argv.slice(2).includes(name);
}

function envOf(name: string): string {
  return process.env[name] ?? '';
}

/** The exact word the trial's prompts name: `WORD-1234`, upper-case letters and four digits. */
const TOKEN_PATTERN = /[A-Z]+-\d{4}/;
/** One prompt's safety valves: 8 seconds per model, at most 60 model calls in the trial. Eight seconds leaves room for two or three retries inside the env's 30-second prompt window; a free model answering a one-word prompt normally takes a few seconds. */
const CALL_TIMEOUT_MS = 8_000;
const MAX_MODEL_CALLS = 60;
/** The gateway refuses a stream after this many tokens, at most: the cap this entry names on every call. */
const GATEWAY_MAX_OUTPUT_TOKENS = 8192;

/** The trial's gateway answer, opened once: its status, its body, its refusal code. */
interface GatewayAnswer {
  status: number;
  text: string;
  json: Json | null;
  code: string;
}

/** One gateway call: POST the Chat Completions body to the gateway, read the whole body, and time it out. A stream's body is SSE, not JSON, so `json` is null on one and the text is what the caller judges. */
async function callGateway(base: string, key: string, body: Json): Promise<GatewayAnswer> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CALL_TIMEOUT_MS);
  try {
    const response = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${key}` },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await response.text();
    let json: Json | null = null;
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed)) json = parsed;
    } catch {
      // Not JSON: a plain refusal.
    }
    const code = json !== null && isRecord(json.error) && typeof json.error.code === 'string' ? json.error.code : 'gateway_error';
    return { status: response.status, text, json, code };
  } finally {
    clearTimeout(timer);
  }
}

/** Whether the prompt names one of the trial's exact words, in the client's own words. */
function tokenOf(body: Json, api: 'openai' | 'anthropic'): string | null {
  if (api === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message)) continue;
      const text = typeof message.content === 'string' ? message.content : '';
      const found = TOKEN_PATTERN.exec(text);
      if (found !== null) return found[0];
    }
    return null;
  }
  const system = typeof body.system === 'string' ? body.system : '';
  const foundSystem = TOKEN_PATTERN.exec(system);
  if (foundSystem !== null) return foundSystem[0];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    const texts: string[] = [];
    if (typeof content === 'string') texts.push(content);
    if (Array.isArray(content)) {
      for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    }
    for (const text of texts) {
      const found = TOKEN_PATTERN.exec(text);
      if (found !== null) return found[0];
    }
  }
  return null;
}

/** Whether a non-streamed answer may be served: it is JSON, it names the model, its usage says it wrote at least one output token, its stop reason is legal on its wire, and its text carries the token the prompt named, when the prompt named one. */
function jsonAnswerIsGood(body: Json, api: 'openai' | 'anthropic', token: string | null): boolean {
  if (api === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return false;
    if (typeof choice.finish_reason !== 'string' || !OPENAI_STOP_REASONS.has(choice.finish_reason)) return false;
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : '';
    if (token !== null && text.indexOf(token) === -1) return false;
    if (!isRecord(body.usage)) return false;
    const output = body.usage.completion_tokens;
    return typeof output === 'number' && Number.isInteger(output) && output >= 1;
  }
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  if (token !== null && text.indexOf(token) === -1) return false;
  if (typeof body.stop_reason !== 'string' || !ANTHROPIC_STOP_REASONS.has(body.stop_reason)) return false;
  if (!isRecord(body.usage)) return false;
  const output = body.usage.output_tokens;
  return typeof output === 'number' && Number.isInteger(output) && output >= 1;
}

/** Whether a streamed answer may be served, in the client's own wire: a whole stream with a finish chunk and [DONE] on the OpenAI wire, or the message_start / content_block / message_delta / message_stop order on the Anthropic one; in both, usage saying it wrote at least one output token, a legal stop reason, and the text carrying the token the prompt named, when the prompt named one. */
function streamAnswerIsGood(text: string, api: 'openai' | 'anthropic', token: string | null): boolean {
  if (api === 'anthropic') return anthropicStreamIsGood(text, token);
  const chunks = openAIChunks(parseSseFrames(text));
  if (chunks === null || !streamIsWhole(chunks)) return false;
  const gathered = gatherStream(chunks);
  const answerText = gathered.textParts.join('');
  if (token !== null && answerText.indexOf(token) === -1) return false;
  if (gathered.completionTokens < 1) return false;
  if (gathered.finishReason === null) return false;
  return OPENAI_STOP_REASONS.has(gathered.finishReason);
}

/** The body the entry sends upstream for one prompt, translated to the OpenAI wire, naming `model` and the trial's output cap. */
function upstreamBodyOf(client: Json, api: 'openai' | 'anthropic', model: string): Json {
  const out = api === 'openai' ? { ...client, model } : anthropicToOpenAI(client, model);
  out.max_tokens = GATEWAY_MAX_OUTPUT_TOKENS;
  out.max_completion_tokens = GATEWAY_MAX_OUTPUT_TOKENS;
  return out;
}

/** The answer the entry serves for one prompt, in the client's wire: the upstream JSON with the model the entry dialled, or the stream translated to the client's wire. A stream has no upstream JSON: its body is SSE. */
function answerOf(upstream: Json | null, streamText: string | null, api: 'openai' | 'anthropic', model: string): { json: Json | null; stream: string | null } {
  if (streamText !== null) {
    return { json: null, stream: api === 'openai' ? openAIStreamText(streamText, model) : anthropicStreamText(streamText, model) };
  }
  if (upstream === null) return { json: null, stream: null };
  return { json: api === 'openai' ? openAIJsonWithModel(upstream, model) : openAIToAnthropic(upstream, model), stream: null };
}

/**
 * One prompt, served by the best live model that can: candidates in rating
 * order, each answer checked before it is served, every refusal kept for the
 * trail header. A stream is read whole before it is judged: its body is SSE,
 * not JSON, so a stream is never refused for not parsing as JSON.
 *
 * The retry set is this prompt only. A model that refused one prompt is
 * tried again on the next: a refusal can be that prompt's shape (a stream
 * that broke, a JSON answer that missed the word), and the best model for
 * the next prompt may be the one that just failed. The only trial-wide
 * guard is the 60-call cap, shared through `state`.
 */
async function servePrompt(
  base: string,
  key: string,
  candidates: Candidate[],
  client: Json,
  api: 'openai' | 'anthropic',
  isStream: boolean,
  state: { calls: number },
): Promise<AttemptResult> {
  const token = tokenOf(client, api);
  const failures: UpstreamFailure[] = [];
  for (const candidate of candidates) {
    if (state.calls >= MAX_MODEL_CALLS) break;
    state.calls += 1;
    let answer: GatewayAnswer;
    try {
      answer = await callGateway(base, key, upstreamBodyOf(client, api, candidate.id));
    } catch {
      failures.push({ status: 0, code: 'gateway_unreachable', model: candidate.id });
      continue;
    }
    if (answer.status !== 200) {
      failures.push({ status: answer.status, code: answer.code, model: candidate.id });
      continue;
    }
    if (!isStream && answer.json === null) {
      failures.push({ status: answer.status, code: 'invalid_answer', model: candidate.id });
      continue;
    }
    const served = answerOf(answer.json, isStream ? answer.text : null, api, candidate.id);
    const good = isStream ? (served.stream !== null && streamAnswerIsGood(served.stream, api, token)) : (served.json !== null && jsonAnswerIsGood(served.json, api, token));
    if (!good) {
      failures.push({ status: answer.status, code: 'invalid_answer', model: candidate.id });
      continue;
    }
    return { kind: 'answered', body: served.json ?? {}, stream: served.stream, model: candidate.id, upstream: answer.json ?? {}, failures };
  }
  return { kind: 'refused', status: failures.length > 0 ? failures[failures.length - 1].status : 502, code: failures.length > 0 ? failures[failures.length - 1].code : 'no_models', failures };
}

/** The server state: the brief, the gateway, and the running total one trial shares — the 60-call cap holds across its prompts. */
interface ServerState {
  brief: Brief;
  catalog: CatalogReport;
  gatewayBase: string;
  gatewayKey: string;
  shared: { calls: number };
}

/** Read a request's body, up to 2 MiB, and parse it as JSON: null when it is not one. */
function readJsonBody(request: IncomingMessage): Promise<Json | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    request.on('end', () => {
      if (size > 2 * 1024 * 1024) {
        resolve(null);
        return;
      }
      try {
        const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        resolve(isRecord(parsed) ? parsed : null);
      } catch {
        resolve(null);
      }
    });
    request.on('error', () => resolve(null));
  });
}

/** The prompt's requirements from its body: JSON mode, tools, vision — the models that can serve it. */
function requirementsOf(client: Json, api: 'openai' | 'anthropic'): Requirement {
  const requires: Requirement = {};
  if (api === 'openai') {
    if (isRecord(client.response_format) && client.response_format.type === 'json_object') requires.json = true;
    if (Array.isArray(client.tools) && client.tools.length > 0) requires.tools = true;
    const messages = Array.isArray(client.messages) ? client.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const part of message.content) if (isRecord(part) && part.type === 'image_url') requires.vision = true;
    }
    return requires;
  }
  if (Array.isArray(client.tools) && client.tools.length > 0) requires.tools = true;
  const messages = Array.isArray(client.messages) ? client.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) if (isRecord(block) && block.type === 'image') requires.vision = true;
  }
  return requires;
}

/** The models that meet a prompt's requirements, best first by rating, id breaking ties. */
function candidatesFor(brief: Brief, requires: Requirement): Candidate[] {
  return freeModels(brief)
    .filter((m) => meetsRequirements(requires, m))
    .map((m, i) => ({ ...m, rank: i }))
    .sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** One refusal trail as the header carries it: at most eight rows, in refusal order. */
function trailHeader(failures: readonly UpstreamFailure[]): string {
  return JSON.stringify(failures.slice(0, 8));
}

function sendJson(response: ServerResponse, status: number, body: Json, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  response.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  response.end(text);
}

function sendStream(response: ServerResponse, status: number, text: string, headers: Record<string, string> = {}): void {
  response.writeHead(status, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
  response.end(text);
}

function sendFailure(
  response: ServerResponse,
  status: number,
  api: 'openai' | 'anthropic',
  failures: readonly UpstreamFailure[],
  body: Json,
  message: string,
): void {
  const headers = { 'x-makefun-upstream-failures': trailHeader(failures) };
  if (api === 'openai') sendJson(response, status, openAIError(message, 'server_error', null), headers);
  else sendJson(response, status, anthropicError('api_error', message), headers);
}

/** One prompt, served and answered on its wire, with the refusal trail on the answer. */
async function handlePrompt(
  request: IncomingMessage,
  response: ServerResponse,
  api: 'openai' | 'anthropic',
  state: ServerState,
): Promise<void> {
  const client = await readJsonBody(request);
  if (client === null) {
    sendFailure(response, 400, api, [], {}, 'invalid request body');
    return;
  }
  const isStream = client.stream === true;
  const candidates = candidatesFor(state.brief, requirementsOf(client, api));
  if (candidates.length === 0) {
    sendFailure(response, 502, api, [], {}, 'no model in the catalog meets this prompt');
    return;
  }
  const result = await servePrompt(state.gatewayBase, state.gatewayKey, candidates, client, api, isStream, state.shared);
  if (result.kind === 'answered') {
    const headers = { 'x-makefun-upstream-failures': trailHeader(result.failures) };
    if (result.stream !== null) {
      sendStream(response, 200, result.stream, headers);
      return;
    }
    sendJson(response, 200, result.body, headers);
    return;
  }
  sendFailure(response, 502, api, result.failures, {}, 'no live free model answered this prompt');
}

/** The server: GET /v1/models in both wires, POST /v1/chat/completions, POST /v1/messages. */
function createApp(state: ServerState) {
  return createServer(async (request, response) => {
    const url = new URL(request.url ?? '/', 'http://127.0.0.1');
    const models: ServedModel[] = state.catalog.providers.flatMap((p) =>
      p.models.map((m) => ({ id: m.id, provider: p.id, display_name: m.id })),
    );
    if (request.method === 'GET' && url.pathname === '/v1/models') {
      const anthropic = request.headers['x-api-key'] !== undefined;
      sendJson(response, 200, anthropic ? anthropicModelList(models) : openAIModelList(models));
      return;
    }
    if (request.method === 'POST' && url.pathname === '/v1/chat/completions') {
      await handlePrompt(request, response, 'openai', state);
      return;
    }
    if (request.method === 'POST' && url.pathname === '/v1/messages') {
      await handlePrompt(request, response, 'anthropic', state);
      return;
    }
    sendJson(response, 404, { error: { message: 'not found', type: 'not_found', code: null } });
  });
}

/** POST the register body to the env's /register: `{"ok": true, "requests": 6}` on a trial. */
async function register(envUrl: string, port: number, catalog: CatalogReport): Promise<boolean> {
  try {
    const response = await fetch(`${envUrl}/register`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
    });
    const text = await response.text();
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      // Not JSON: the register failed.
    }
    return response.ok && isRecord(body) && body.ok === true;
  } catch {
    return false;
  }
}

/** The env's /work: `{"done": false, "next": "<id>", "pending": <n>}` until every prompt was sent. */
async function pollWork(envUrl: string): Promise<{ done: boolean }> {
  try {
    const response = await fetch(`${envUrl}/work`);
    const text = await response.text();
    const body: unknown = JSON.parse(text);
    return { done: isRecord(body) && body.done === true };
  } catch {
    return { done: false };
  }
}

/** The work loop: register, then poll the env's /work until it says done, then exit 0. */
async function runTrial(envUrl: string, port: number, catalog: CatalogReport, server: ReturnType<typeof createServer>): Promise<number> {
  const ok = await register(envUrl, port, catalog);
  if (!ok) {
    server.close();
    return 1;
  }
  for (let i = 0; i < 600; i++) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const work = await pollWork(envUrl);
    if (work.done) break;
  }
  server.close();
  return 0;
}

async function main(): Promise<number> {
  const promptFile = envOf('MAKEFUN_PROMPT_FILE');
  const envUrl = envOf('MAKEFUN_ENV_URL');
  const brief: Brief = argvHas('--list') && promptFile === '' ? { models: [], prompts: [], error: null } : readBrief(promptFile);
  if (brief.error !== null) {
    process.stderr.write(`${brief.error}\n`);
    return 1;
  }
  const catalog = buildCatalog(brief);

  if (argvHas('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return 0;
  }

  if (envUrl === '') {
    process.stderr.write('MAKEFUN_ENV_URL is not set\n');
    return 1;
  }

  const gatewayBase = envOf('OPENAI_BASE_URL') !== '' ? envOf('OPENAI_BASE_URL') : 'https://openrouter.ai/api/v1';
  const gatewayKey = envOf('OPENAI_API_KEY');

  // The trial's gateway owns port 8080; the entry takes a free port of its own.
  // The running total is one object shared by every prompt in this trial,
  // so the 60-call cap holds across the whole trial.
  const state: ServerState = {
    brief,
    catalog,
    gatewayBase,
    gatewayKey,
    shared: { calls: 0 },
  };
  const app = createApp(state);
  const port = await new Promise<number>((resolve, reject) => {
    app.once('error', reject);
    app.listen(0, '0.0.0.0', () => {
      const address = app.address();
      if (address !== null && typeof address === 'object') resolve(address.port);
      else reject(new Error('could not bind a port'));
    });
  });

  process.exitCode = await runTrial(envUrl, port, catalog, app);
  return process.exitCode;
}

void main().then((code) => {
  process.exitCode = code;
});
