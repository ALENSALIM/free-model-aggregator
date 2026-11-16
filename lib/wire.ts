// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls this entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. The entry dials the gateway, an
// OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id)
//     to the gateway and relays the gateway's JSON or SSE answer, with every
//     `model` field set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system
//     prompt, image blocks, tool definitions, tool results and tool choice
//     included) and rewrites the gateway's answer into a Messages object, or
//     into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the
//     stop reason and the usage translated. Tool calls become tool_use blocks
//     and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really
//   answered, so a response's `model` must be the model this entry dialled.
//   The gateway's answer names a free model without the `openrouter/` prefix;
//   these functions put the dialled id back on every answer and every
//   streamed chunk. They never make a model up: the id came from the brief
//   and went to the gateway in the request's `model`.
//
// Streams are buffered, then translated: the upstream stream is read whole
// first, so a stream that breaks halfway can fail over to another model
// instead of serving a half answer. The event order the client sees is the
// same either way, and the trial's answers are short.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same gateway answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The Anthropic wire's stop reasons, as the trial's client checks them. */
export const ANTHROPIC_STOP_REASONS = new Set(['end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'refusal', 'pause_turn']);
/** The OpenAI wire's stop reasons, as the trial's client checks them. */
export const OPENAI_STOP_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
export function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame, split on the blank line that ends it. */
export interface SseFrame {
  event: string | null;
  /** The data lines' payloads, in order (usually one). */
  data: string[];
}

/** An SSE body as its frames: split on blank lines, `event:` and `data:` lines read off each frame. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const raw of normalized.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data });
  }
  return frames;
}

/** One parsed OpenAI chunk: its delta, its finish reason and its usage, if any. */
export interface OpenAIChunk {
  text: string;
  toolCalls: { index: number; id: string; name: string; args: string }[];
  finishReason: string | null;
  promptTokens: number | null;
  completionTokens: number | null;
  sawDone: boolean;
}

/** An SSE body's OpenAI chunks, in order. Returns null when a data payload is not JSON. */
export function openAIChunks(frames: readonly SseFrame[]): OpenAIChunk[] | null {
  const chunks: OpenAIChunk[] = [];
  for (const frame of frames) {
    for (const payload of frame.data) {
      if (payload.trim() === '[DONE]') {
        chunks.push({ text: '', toolCalls: [], finishReason: null, promptTokens: null, completionTokens: null, sawDone: true });
        continue;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(payload);
      } catch {
        return null;
      }
      if (!isRecord(parsed)) return null;
      const chunk: OpenAIChunk = { text: '', toolCalls: [], finishReason: null, promptTokens: null, completionTokens: null, sawDone: false };
      if (isRecord(parsed.usage)) {
        chunk.promptTokens = typeof parsed.usage.prompt_tokens === 'number' ? parsed.usage.prompt_tokens : null;
        chunk.completionTokens = typeof parsed.usage.completion_tokens === 'number' ? parsed.usage.completion_tokens : null;
      }
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null) {
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') chunk.text = delta.content;
        if (Array.isArray(delta.tool_calls)) {
          for (const call of delta.tool_calls) {
            if (!isRecord(call)) continue;
            const fn = isRecord(call.function) ? call.function : {};
            chunk.toolCalls.push({
              index: typeof call.index === 'number' ? call.index : 0,
              id: typeof call.id === 'string' ? call.id : '',
              name: typeof fn.name === 'string' ? fn.name : '',
              args: typeof fn.arguments === 'string' ? fn.arguments : '',
            });
          }
        }
        if (typeof choice.finish_reason === 'string') chunk.finishReason = choice.finish_reason;
      }
      chunks.push(chunk);
    }
  }
  return chunks;
}

/** Whether a parsed stream is whole: it has a finish chunk and ends with [DONE]. */
export function streamIsWhole(chunks: readonly OpenAIChunk[]): boolean {
  return chunks.some((c) => c.finishReason !== null) && chunks.some((c) => c.sawDone);
}

/**
 * A Chat Completions SSE body as it came, every data payload's `model` the id
 * the entry dialled, with [DONE] appended when the upstream ended without one.
 * Frames are split on the blank line that ends them, as the SSE grammar does.
 */
export function openAIStreamText(text: string, model: string): string {
  const frames = parseSseFrames(text);
  const out: string[] = [];
  let sawDone = false;
  for (const frame of frames) {
    const lines: string[] = [];
    if (frame.event !== null) lines.push(`event: ${frame.event}`);
    for (const payload of frame.data) {
      if (payload.trim() === '[DONE]') {
        sawDone = true;
        lines.push('data: [DONE]');
        continue;
      }
      try {
        const chunk: unknown = JSON.parse(payload);
        if (isRecord(chunk)) lines.push(`data: ${JSON.stringify({ ...chunk, model })}`);
        else lines.push(`data: ${payload}`);
      } catch {
        lines.push(`data: ${payload}`);
      }
    }
    out.push(`${lines.join('\n')}\n\n`);
  }
  if (!sawDone) out.push('data: [DONE]\n\n');
  return out.join('');
}

/** The gathered content of a parsed OpenAI stream: its text, its tool calls, its usage and its stop. */
export interface GatheredStream {
  textParts: string[];
  toolCalls: { id: string; name: string; args: string }[];
  promptTokens: number;
  completionTokens: number;
  finishReason: string | null;
}

/** Gather a parsed stream's text, tool calls, usage and finish reason, in order. */
export function gatherStream(chunks: readonly OpenAIChunk[]): GatheredStream {
  const textParts: string[] = [];
  const tools = new Map<number, { id: string; name: string; args: string }>();
  let promptTokens = 0;
  let completionTokens = 0;
  let finishReason: string | null = null;
  for (const chunk of chunks) {
    if (chunk.promptTokens !== null) promptTokens = chunk.promptTokens;
    if (chunk.completionTokens !== null) completionTokens = chunk.completionTokens;
    if (chunk.text !== '') textParts.push(chunk.text);
    for (const call of chunk.toolCalls) {
      const held = tools.get(call.index) ?? { id: call.id, name: call.name, args: '' };
      if (call.id !== '') held.id = call.id;
      if (call.name !== '') held.name = call.name;
      held.args += call.args;
      tools.set(call.index, held);
    }
    if (chunk.finishReason !== null) finishReason = chunk.finishReason;
  }
  return {
    textParts,
    toolCalls: [...tools.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call),
    promptTokens,
    completionTokens,
    finishReason,
  };
}

/**
 * Whether an Anthropic SSE body may be served: message_start first, every
 * content block opened and closed in order, message_delta after the blocks
 * close with a legal stop reason and at least one output token, message_stop
 * last and nothing after it, and the gathered text carrying the token the
 * prompt named, when the prompt named one. The same order the trial's
 * client checks.
 */
export function anthropicStreamIsGood(text: string, token: string | null): boolean {
  const frames = parseSseFrames(text);
  if (frames.length === 0) return false;
  const open = new Set<number>();
  let sawStart = false;
  let sawDelta = false;
  let sawStop = false;
  let stopReason: string | null = null;
  let outputTokens = 0;
  let answerText = '';
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (frame.data.length === 0) return false;
    let event: unknown;
    try {
      event = JSON.parse(frame.data[0]);
    } catch {
      return false;
    }
    if (!isRecord(event) || typeof event.type !== 'string') return false;
    if (sawStop) return false;
    if (i === 0 && event.type !== 'message_start') return false;
    switch (event.type) {
      case 'message_start': {
        if (sawStart) return false;
        sawStart = true;
        break;
      }
      case 'content_block_start': {
        if (!sawStart || typeof event.index !== 'number' || open.has(event.index)) return false;
        open.add(event.index);
        break;
      }
      case 'content_block_delta': {
        if (typeof event.index !== 'number' || !open.has(event.index) || !isRecord(event.delta)) return false;
        if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string') answerText += event.delta.text;
        break;
      }
      case 'content_block_stop': {
        if (typeof event.index !== 'number' || !open.has(event.index)) return false;
        open.delete(event.index);
        break;
      }
      case 'message_delta': {
        if (open.size !== 0 || sawDelta) return false;
        sawDelta = true;
        if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') stopReason = event.delta.stop_reason;
        if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') outputTokens = event.usage.output_tokens;
        break;
      }
      case 'message_stop': {
        if (!sawDelta) return false;
        sawStop = true;
        break;
      }
      default:
        return false;
    }
  }
  if (!sawStart || !sawDelta || !sawStop || open.size !== 0) return false;
  if (stopReason === null || !ANTHROPIC_STOP_REASONS.has(stopReason)) return false;
  if (outputTokens < 1) return false;
  if (token !== null && answerText.indexOf(token) === -1) return false;
  return true;
}

/**
 * A Chat Completions SSE body as the Anthropic event sequence: message_start
 * on the first event, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then message_stop.
 * `model` is the id the entry dialled, named back to the client.
 */
export function anthropicStreamText(text: string, model: string): string {
  const chunks = openAIChunks(parseSseFrames(text));
  if (chunks === null) return '';
  const gathered = gatherStream(chunks);
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: gathered.promptTokens, output_tokens: 0 },
      },
    }),
  );
  let nextIndex = 0;
  const answerText = gathered.textParts.join('');
  if (answerText !== '') {
    const index = nextIndex;
    nextIndex += 1;
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const part of gathered.textParts) {
      events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: part } }));
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  for (const call of gathered.toolCalls) {
    const index = nextIndex;
    nextIndex += 1;
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : `toolu_${index}`, name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }));
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(gathered.finishReason), stop_sequence: null },
      usage: { output_tokens: gathered.completionTokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events.join('');
}
