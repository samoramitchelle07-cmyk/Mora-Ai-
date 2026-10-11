import 'dotenv/config';
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { GoogleGenAI } from '@google/genai';
import { Sandbox } from '@e2b/code-interpreter';
import archiver from 'archiver';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const DATA_DIR = path.join(ROOT, 'data');
const PROJECTS_FILE = path.join(DATA_DIR, 'projects.json');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');
const CONVERSATIONS_FILE = path.join(DATA_DIR, 'conversations.json');
const IMAGE_DIR = path.join(ROOT, 'public', 'generated-images');

const MAX_PROMPT = 12000;
const MAX_FILE_CHARS = 180000;
const MAX_FILES = 25;
const MAX_JSON_BODY = '12mb';
const IS_PRODUCTION = process.env.NODE_ENV === 'production';
const TOKEN = (process.env.MORA_API_TOKEN || '').trim();

const MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const GROQ_MODEL = process.env.GROQ_MODEL || 'openai/gpt-oss-120b';
const OPENROUTER_MODEL =
  process.env.OPENROUTER_MODEL || 'openai/gpt-oss-120b';

const AI_OUTPUT_TOKENS = Math.max(
  1024,
  Math.min(32768, Number(process.env.AI_OUTPUT_TOKENS || 14000))
);

const AI_PROVIDER_ORDER = (
  process.env.AI_PROVIDER_ORDER || 'gemini,groq,openrouter'
)
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

if (IS_PRODUCTION && !TOKEN) {
  console.error(
    'SECURITY ERROR: Set MORA_API_TOKEN before exposing Mora AI in production.'
  );
  process.exit(1);
}

const app = express();

app.disable('x-powered-by');
app.set('trust proxy', process.env.TRUST_PROXY === '1' ? 1 : false);

app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false
  })
);

const allowedOrigins = (process.env.CORS_ORIGIN || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: allowedOrigins.length
      ? allowedOrigins
      : IS_PRODUCTION
        ? false
        : true
  })
);

app.use(express.json({ limit: MAX_JSON_BODY }));

app.use(
  '/api/',
  rateLimit({
    windowMs: 60000,
    limit: Math.max(
      1,
      Number(process.env.MAX_REQUESTS_PER_MINUTE || 20)
    ),
    standardHeaders: 'draft-7',
    legacyHeaders: false
  })
);

app.use(
  '/api/images/generate',
  rateLimit({
    windowMs: 60000,
    limit: 5,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: {
      success: false,
      error:
        'Image generation is limited to 5 requests per minute. Please wait and try again.'
    }
  })
);

function safeEqual(a, b) {
  const aa = Buffer.from(String(a));
  const bb = Buffer.from(String(b));

  return (
    aa.length === bb.length &&
    crypto.timingSafeEqual(aa, bb)
  );
}

function auth(req, res, next) {
  if (!TOKEN) return next();

  const provided =
    req.get('authorization')?.replace(/^Bearer\s+/i, '') ||
    req.get('x-mora-token') ||
    '';

  if (!safeEqual(provided, TOKEN)) {
    return res.status(401).json({
      success: false,
      error:
        'Unauthorized. Configure the Mora API token in Mora AI settings.'
    });
  }

  next();
}

app.use('/api/agent', auth);
app.use('/api/projects', auth);
app.use('/api/history', auth);
app.use('/api/test', auth);
app.use('/api/providers', auth);
app.use('/api/images', auth);
app.use('/api/conversations', auth);

app.get('/', (_req, res) => {
  res.sendFile(path.join(ROOT, 'public', 'index.html'));
});

app.use(
  '/assets',
  express.static(path.join(ROOT, 'public', 'assets'), {
    dotfiles: 'deny',
    index: false,
    fallthrough: false
  })
);

app.use(
  '/generated-images',
  express.static(IMAGE_DIR, {
    dotfiles: 'deny',
    index: false,
    maxAge: '7d',
    immutable: true
  })
);

async function readJson(file, fallback) {
  try {
    return JSON.parse(await fs.readFile(file, 'utf8'));
  } catch (e) {
    if (e.code === 'ENOENT') return fallback;
    throw e;
  }
}

async function writeJson(file, value) {
  await fs.mkdir(DATA_DIR, { recursive: true });

  const tmp =
    `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;

  await fs.writeFile(
    tmp,
    JSON.stringify(value, null, 2),
    {
      encoding: 'utf8',
      mode: 0o600
    }
  );

  await fs.rename(tmp, file);
}

function safeFilename(input, i = 0) {
  const raw = String(input || `file-${i + 1}.txt`)
    .replace(/\\/g, '/')
    .split('/')
    .pop();

  const safe = raw
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .slice(0, 100);

  return safe && safe !== '.' && safe !== '..'
    ? safe
    : `file-${i + 1}.txt`;
}

function normalizeFiles(files) {
  if (!Array.isArray(files)) return [];

  return files.slice(0, MAX_FILES).map((f, i) => ({
    name: safeFilename(f?.name || f?.path, i),
    path: safeFilename(f?.path || f?.name, i),
    language: String(f?.language || '').slice(0, 40),
    content: String(f?.content ?? '').slice(0, MAX_FILE_CHARS)
  }));
}

function parseJson(text) {
  if (!text || !String(text).trim()) {
    throw new Error('The AI provider returned an empty response.');
  }

  const cleaned = String(text)
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/, '');

  try {
    return JSON.parse(cleaned);
  } catch {}

  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');

  if (start >= 0 && end > start) {
    try {
      return JSON.parse(cleaned.slice(start, end + 1));
    } catch {}
  }

  throw new Error(
    'Could not parse the AI response. Please retry with a more specific prompt.'
  );
}

function normalizeResult(raw, prompt) {
  const files = normalizeFiles(raw?.files);

  const canvas =
    raw?.canvas &&
    typeof raw.canvas.html === 'string' &&
    raw.canvas.html.trim()
      ? {
          type: 'html',
          title: String(
            raw.canvas.title || 'Mora AI Preview'
          ).slice(0, 100),
          html: raw.canvas.html.slice(0, MAX_FILE_CHARS)
        }
      : null;

  const execution =
    raw?.execution && typeof raw.execution === 'object'
      ? {
          requested: Boolean(raw.execution.requested),
          language: String(
            raw.execution.language || 'python'
          ).toLowerCase(),
          code: String(raw.execution.code || '').slice(0, 12000)
        }
      : {
          requested: false,
          language: 'python',
          code: ''
        };

  let message = '';

  if (
    typeof raw?.message === 'string' &&
    raw.message.trim()
  ) {
    message = raw.message.trim();
  } else if (files.length) {
    message =
      'I prepared the requested file(s). Open the workspace to review them.';
  } else if (canvas) {
    message = 'I created the requested preview.';
  } else {
    message =
      'I received an empty text response from the AI provider. Please test your text API connection in Settings and try again.';
  }

  return {
    success: true,
    message: message.slice(0, 12000),

    steps: Array.isArray(raw?.steps)
      ? raw.steps.slice(0, 15).map((s, i) => ({
          thought: String(
            s?.thought || `Step ${i + 1}`
          ).slice(0, 300),
          command: String(s?.command || 'process').slice(0, 100),
          output: String(s?.output || '').slice(0, 2500),
          isError: Boolean(s?.isError)
        }))
      : [],

    files,
    canvas,
    execution,

    terminal: String(raw?.terminal || '').slice(0, 12000),
    status: 'completed',
    prompt: prompt.slice(0, MAX_PROMPT)
  };
}

const SYSTEM_PROMPT = `You are Mora AI, a practical personal AI agent for coding, debugging, building websites, research, learning, and general assistance.

Return ONLY one valid JSON object with this exact shape:
{
  "message": "answer",
  "steps": [
    {
      "thought": "brief user-facing status, never hidden chain-of-thought",
      "command": "plan|generate|debug|review|test",
      "output": "short result",
      "isError": false
    }
  ],
  "files": [
    {
      "name": "index.html",
      "path": "index.html",
      "language": "html",
      "content": "complete file contents"
    }
  ],
  "canvas": {
    "type": "html",
    "title": "Preview title",
    "html": "complete standalone HTML"
  },
  "execution": {
    "requested": false,
    "language": "python",
    "code": ""
  },
  "terminal": ""
}

Rules:
- Never reveal private chain-of-thought; summarize actions only.
- For ordinary questions, answer in message and use empty files, canvas null, and execution.requested false.
- For website/interface requests, provide complete standalone HTML in canvas.html and normally a matching index.html in files.
- Inline CSS and JavaScript for preview portability.
- For code requests, return complete files, not snippets.
- Do not claim code was tested unless it actually ran.
- Request execution only for Python when the user asks to run/test Python or needs a Python calculation.
- Never execute shell commands, JavaScript, destructive code, credential theft, unauthorized access, or code that evades security.
- Use concise progress steps.
- Keep files under 180,000 characters and total output reasonable.
`;

function configuredProviderMap() {
  return {
    gemini: {
      id: 'gemini',
      name: 'Gemini',
      configured: Boolean(process.env.GEMINI_API_KEY),
      model: MODEL
    },
    groq: {
      id: 'groq',
      name: 'Groq',
      configured: Boolean(process.env.GROQ_API_KEY),
      model: GROQ_MODEL
    },
    openrouter: {
      id: 'openrouter',
      name: 'OpenRouter',
      configured: Boolean(process.env.OPENROUTER_API_KEY),
      model: OPENROUTER_MODEL
    }
  };
}

function configuredProviders() {
  const settings = configuredProviderMap();
  const seen = new Set();

  return AI_PROVIDER_ORDER
    .filter(id => settings[id]?.configured && !seen.has(id) && seen.add(id))
    .map(id => settings[id]);
}

function providerError(provider, status, message) {
  const e = new Error(
    `${provider}: ${String(message || 'Request failed').slice(0, 700)}`
  );

  e.provider = provider;
  e.status = Number(status) || 502;
  e.providerMessage = String(
    message || 'Request failed'
  ).slice(0, 700);

  return e;
}

function getErrorStatus(error) {
  const direct = Number(
    error?.status ||
    error?.statusCode ||
    error?.response?.status ||
    error?.cause?.status ||
    error?.cause?.statusCode
  );

  if (Number.isFinite(direct) && direct > 0) {
    return direct;
  }

  const msg = String(error?.message || error || '');

  const match = msg.match(
    /\b(?:HTTP\s*)?(400|401|402|403|404|408|413|429|500|502|503|504)\b/i
  );

  if (match) return Number(match[1]);

  if (/quota|rate.?limit|resource_exhausted|too many requests/i.test(msg)) {
    return 429;
  }

  if (/unauthorized|invalid api key|authentication/i.test(msg)) {
    return 401;
  }

  return 502;
}

function validateProviderInput(body = {}) {
  const provider = String(body.provider || '')
    .trim()
    .toLowerCase();

  if (
    provider &&
    !['gemini', 'groq', 'openrouter', 'auto'].includes(provider)
  ) {
    throw providerError(
      'Configuration',
      400,
      'Unsupported provider.'
    );
  }

  const apiKey =
    typeof body.apiKey === 'string'
      ? body.apiKey.trim()
      : '';

  if (apiKey.length > 500) {
    throw providerError(
      'Configuration',
      400,
      'API key value is too long.'
    );
  }

  const model =
    typeof body.model === 'string'
      ? body.model.trim().slice(0, 160)
      : '';

  return {
    provider: provider || 'auto',
    apiKey,
    model,
    fallback: body.fallback !== false
  };
}

function providerCandidates(input = {}) {
  const defaults = configuredProviderMap();
  const selection = validateProviderInput(input);
  const out = [];

  const add = (id, apiKey, model, source) => {
    if (
      !['gemini', 'groq', 'openrouter'].includes(id) ||
      !apiKey
    ) {
      return;
    }

    if (out.some(p => p.id === id && p.apiKey === apiKey)) {
      return;
    }

    out.push({
      id,
      name: defaults[id].name,
      apiKey,
      model: model || defaults[id].model,
      source
    });
  };

  if (selection.provider !== 'auto') {
    const id = selection.provider;

    const envKey =
      id === 'gemini'
        ? process.env.GEMINI_API_KEY
        : id === 'groq'
          ? process.env.GROQ_API_KEY
          : process.env.OPENROUTER_API_KEY;

    if (selection.apiKey) {
      add(id, selection.apiKey, selection.model, 'personal');
    } else {
      add(
        id,
        envKey,
        selection.model || defaults[id].model,
        'admin'
      );
    }

    if (selection.fallback) {
      for (const fallbackId of AI_PROVIDER_ORDER) {
        if (fallbackId === id) continue;

        const key =
          fallbackId === 'gemini'
            ? process.env.GEMINI_API_KEY
            : fallbackId === 'groq'
              ? process.env.GROQ_API_KEY
              : process.env.OPENROUTER_API_KEY;

        add(
          fallbackId,
          key,
          defaults[fallbackId]?.model,
          'admin'
        );
      }
    }
  } else {
    if (selection.apiKey) {
      const personalProvider =
        String(input.providerForKey || '').toLowerCase();

      if (
        !['gemini', 'groq', 'openrouter'].includes(personalProvider)
      ) {
        throw providerError(
          'Configuration',
          400,
          'Choose which provider the personal API key belongs to.'
        );
      }

      add(
        personalProvider,
        selection.apiKey,
        selection.model,
        'personal'
      );
    }

    if (selection.fallback || !selection.apiKey) {
      for (const id of AI_PROVIDER_ORDER) {
        const key =
          id === 'gemini'
            ? process.env.GEMINI_API_KEY
            : id === 'groq'
              ? process.env.GROQ_API_KEY
              : process.env.OPENROUTER_API_KEY;

        add(id, key, defaults[id]?.model, 'admin');
      }
    }
  }

  return { selection, providers: out };
}

async function requestGeminiText(
  prompt,
  provider,
  repairContext = ''
) {
  if (!provider.apiKey) {
    throw providerError(
      'Gemini',
      503,
      'API key is not configured.'
    );
  }

  const contents =
    `${SYSTEM_PROMPT}\n\nUSER REQUEST:\n${prompt}` +
    (
      repairContext
        ? `\n\nREPAIR CONTEXT:\n${repairContext}`
        : ''
    );

  try {
    const client = new GoogleGenAI({
      apiKey: provider.apiKey
    });

    const response = await client.models.generateContent({
      model: provider.model,
      contents,
      config: {
        temperature: 0.3,
        maxOutputTokens: AI_OUTPUT_TOKENS,
        responseMimeType: 'application/json'
      }
    });

    const text = response?.text || '';

    if (!text.trim()) {
      throw new Error('Gemini returned an empty response.');
    }

    return {
      text,
      model: provider.model
    };
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'Gemini',
      getErrorStatus(e),
      e.message || 'Gemini request failed.'
    );
  }
}
// ============================================================
// PART 2 — TEXT PROVIDERS, FALLBACK, AND IMAGE GENERATION
// ============================================================

async function requestOpenAICompatibleText(
  prompt,
  provider,
  repairContext = ''
) {
  const isGroq = provider.id === 'groq';

  const endpoint = isGroq
    ? 'https://api.groq.com/openai/v1/chat/completions'
    : 'https://openrouter.ai/api/v1/chat/completions';

  if (!provider.apiKey) {
    throw providerError(
      provider.name,
      503,
      'API key is not configured.'
    );
  }

  const messages = [
    {
      role: 'system',
      content: SYSTEM_PROMPT
    },
    {
      role: 'user',
      content:
        prompt +
        (
          repairContext
            ? '\n\nREPAIR CONTEXT:\n' + repairContext
            : ''
        )
    }
  ];

  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    120000
  );

  try {
    const headers = {
      Authorization: `Bearer ${provider.apiKey}`,
      'Content-Type': 'application/json'
    };

    if (!isGroq) {
      headers['HTTP-Referer'] =
        process.env.OPENROUTER_SITE_URL ||
        'http://localhost:3000';

      headers['X-Title'] =
        process.env.OPENROUTER_APP_NAME || 'Mora AI';
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        model: provider.model,
        messages,
        temperature: 0.3,
        max_tokens: AI_OUTPUT_TOKENS
      })
    });

    const raw = await response.text();

    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      throw providerError(
        provider.name,
        response.status,
        'The provider returned an invalid response.'
      );
    }

    if (!response.ok) {
      throw providerError(
        provider.name,
        response.status,
        data?.error?.message ||
          data?.message ||
          'The provider request failed.'
      );
    }

    const text =
      data?.choices?.[0]?.message?.content;

    if (typeof text !== 'string' || !text.trim()) {
      throw providerError(
        provider.name,
        502,
        'The provider returned an empty message.'
      );
    }

    return {
      text,
      model: provider.model
    };
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      provider.name,
      e.name === 'AbortError'
        ? 408
        : getErrorStatus(e),
      e.name === 'AbortError'
        ? 'The request timed out.'
        : e.message || 'The request failed.'
    );
  } finally {
    clearTimeout(timeout);
  }
}

async function requestProviderText(
  prompt,
  provider,
  repairContext = ''
) {
  if (provider.id === 'gemini') {
    return requestGeminiText(
      prompt,
      provider,
      repairContext
    );
  }

  return requestOpenAICompatibleText(
    prompt,
    provider,
    repairContext
  );
}

async function generateAIResponse(
  prompt,
  options = {}
) {
  const { selection, providers } =
    providerCandidates(options);

  if (!providers.length) {
    throw providerError(
      'Configuration',
      503,
      'No text AI provider is configured. Add GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY to your server .env file.'
    );
  }

  const errors = [];

  for (const provider of providers) {
    try {
      let result = await requestProviderText(
        prompt,
        provider
      );

      let parsed;

      try {
        parsed = parseJson(result.text);
      } catch (firstError) {
        // Retry once with a format correction.
        const repair = await requestProviderText(
          prompt,
          provider,
          'Your previous response was not valid JSON. Return one complete JSON object matching the required schema. Do not use Markdown fences. Do not return an empty response. Previous response:\n' +
            result.text.slice(0, 12000)
        );

        result = repair;
        parsed = parseJson(result.text);
      }

      const normalized = normalizeResult(
        parsed,
        prompt
      );

      normalized.provider = provider.id;
      normalized.model = result.model;

      return normalized;
    } catch (e) {
      errors.push({
        provider: provider.id,
        status: getErrorStatus(e),
        message: String(
          e.providerMessage ||
          e.message ||
          'Request failed'
        ).slice(0, 500)
      });

      console.error(
        `[Mora AI] ${provider.id} request failed:`,
        getErrorStatus(e),
        e.message
      );

      // If a specific provider was requested and fallback
      // is disabled, stop instead of switching providers.
      if (
        selection.provider !== 'auto' &&
        !selection.fallback
      ) {
        break;
      }
    }
  }

  const statuses = errors
    .map(e => `${e.provider}: ${e.status}`)
    .join(', ');

  const detail = errors
    .map(e => `${e.provider}: ${e.message}`)
    .join(' | ')
    .slice(0, 1200);

  const error = providerError(
    'Mora AI',
    errors.some(e => e.status === 401)
      ? 401
      : errors.some(e => e.status === 429)
        ? 429
        : 502,
    `All available text providers failed. ${statuses}. ${detail}`
  );

  error.providerErrors = errors;

  throw error;
}

// ------------------------------------------------------------
// IMAGE GENERATION UTILITIES
// ------------------------------------------------------------

function imageMimeFromFormat(format) {
  const value = String(format || '').toLowerCase();

  if (value === 'jpeg' || value === 'jpg') {
    return 'image/jpeg';
  }

  if (value === 'webp') {
    return 'image/webp';
  }

  return 'image/png';
}

function decodeImageData(value) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error('The provider returned no image data.');
  }

  const match = value.match(
    /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\r\n]+)$/i
  );

  if (match) {
    return {
      mime: match[1].toLowerCase(),
      buffer: Buffer.from(
        match[2].replace(/\s/g, ''),
        'base64'
      )
    };
  }

  if (
    !/^[A-Za-z0-9+/=\r\n]+$/.test(value) ||
    value.length < 100
  ) {
    throw new Error('The image data is not valid base64.');
  }

  return {
    mime: 'image/png',
    buffer: Buffer.from(
      value.replace(/\s/g, ''),
      'base64'
    )
  };
}

async function saveGeneratedImage(buffer, mime) {
  const maxBytes = Math.max(
    1,
    Number(
      process.env.MAX_GENERATED_IMAGE_MB || 12
    )
  ) * 1024 * 1024;

  if (!Buffer.isBuffer(buffer) || !buffer.length) {
    throw new Error('The generated image is empty.');
  }

  if (buffer.length > maxBytes) {
    throw new Error(
      'The generated image exceeds the configured size limit.'
    );
  }

  const signatures = [
    {
      mime: 'image/png',
      valid: b =>
        b.length >= 8 &&
        b.subarray(0, 8).equals(
          Buffer.from([
            137, 80, 78, 71, 13, 10, 26, 10
          ])
        )
    },
    {
      mime: 'image/jpeg',
      valid: b =>
        b.length >= 3 &&
        b[0] === 0xff &&
        b[1] === 0xd8 &&
        b[2] === 0xff
    },
    {
      mime: 'image/webp',
      valid: b =>
        b.length >= 12 &&
        b.toString('ascii', 0, 4) === 'RIFF' &&
        b.toString('ascii', 8, 12) === 'WEBP'
    }
  ];

  const signature = signatures.find(
    s => s.valid(buffer)
  );

  if (!signature) {
    throw new Error(
      'The provider response was not a supported image file.'
    );
  }

  const extension =
    signature.mime === 'image/jpeg'
      ? 'jpg'
      : signature.mime === 'image/webp'
        ? 'webp'
        : 'png';

  await fs.mkdir(IMAGE_DIR, {
    recursive: true
  });

  const filename =
    `${Date.now()}-${crypto.randomUUID()}.${extension}`;

  const destination = path.join(
    IMAGE_DIR,
    filename
  );

  await fs.writeFile(
    destination,
    buffer,
    {
      flag: 'wx',
      mode: 0o644
    }
  );

  return {
    filename,
    mime: signature.mime,
    bytes: buffer.length,
    url: `/generated-images/${filename}`
  };
}

async function fetchImageBuffer(url) {
  const parsed = new URL(url);

  if (parsed.protocol !== 'https:') {
    throw new Error(
      'The image provider returned an unsupported URL.'
    );
  }

  const controller = new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    90000
  );

  try {
    const response = await fetch(parsed, {
      signal: controller.signal,
      redirect: 'error'
    });

    if (!response.ok) {
      throw new Error(
        `Image download failed with HTTP ${response.status}.`
      );
    }

    const contentType = String(
      response.headers.get('content-type') || ''
    ).toLowerCase();

    if (!contentType.startsWith('image/')) {
      throw new Error(
        'The provider URL did not return an image.'
      );
    }

    const declaredSize = Number(
      response.headers.get('content-length') || 0
    );

    const maxBytes = Math.max(
      1,
      Number(
        process.env.MAX_GENERATED_IMAGE_MB || 12
      )
    ) * 1024 * 1024;

    if (declaredSize > maxBytes) {
      throw new Error(
        'The generated image exceeds the size limit.'
      );
    }

    const chunks = [];
    let total = 0;

    for await (const chunk of response.body) {
      total += chunk.length;

      if (total > maxBytes) {
        throw new Error(
          'The generated image exceeds the size limit.'
        );
      }

      chunks.push(chunk);
    }

    return {
      buffer: Buffer.concat(chunks),
      mime: contentType.split(';')[0]
    };
  } finally {
    clearTimeout(timeout);
  }
}

// ------------------------------------------------------------
// GEMINI IMAGE GENERATION
// ------------------------------------------------------------

async function generateGeminiImage(prompt) {
  const key = process.env.GEMINI_API_KEY;

  if (!key) {
    throw providerError(
      'Gemini Images',
      503,
      'GEMINI_API_KEY is not configured.'
    );
  }

  const model =
    process.env.IMAGE_GEMINI_MODEL ||
    'gemini-2.5-flash-image';

  const controller = new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    180000
  );

  try {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': key
        },
        signal: controller.signal,
        body: JSON.stringify({
          contents: [
            {
              role: 'user',
              parts: [
                {
                  text:
                    'Generate an image from this description: ' +
                    prompt
                }
              ]
            }
          ],
          generationConfig: {
            responseModalities: ['TEXT', 'IMAGE']
          }
        })
      }
    );

    const raw = await response.text();

    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      throw providerError(
        'Gemini Images',
        response.status || 502,
        'The image provider returned invalid JSON.'
      );
    }

    if (!response.ok) {
      throw providerError(
        'Gemini Images',
        response.status,
        data?.error?.message ||
          'Gemini image generation failed.'
      );
    }

    const parts =
      data?.candidates?.flatMap(
        c => c?.content?.parts || []
      ) || [];

    const imagePart = parts.find(
      p => p?.inlineData?.data
    );

    if (!imagePart) {
      const explanation = parts
        .map(p => p?.text || '')
        .filter(Boolean)
        .join('\n')
        .slice(0, 500);

      throw providerError(
        'Gemini Images',
        502,
        explanation ||
          'Gemini returned no image. Check model availability and API permissions.'
      );
    }

    return decodeImageData(
      `data:${imagePart.inlineData.mimeType || 'image/png'};base64,${imagePart.inlineData.data}`
    );
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'Gemini Images',
      e.name === 'AbortError'
        ? 408
        : getErrorStatus(e),
      e.name === 'AbortError'
        ? 'Gemini image generation timed out.'
        : e.message || 'Image generation failed.'
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ------------------------------------------------------------
// OPENAI IMAGE GENERATION
// ------------------------------------------------------------

async function generateOpenAIImage(prompt) {
  const key = process.env.OPENAI_API_KEY;

  if (!key) {
    throw providerError(
      'OpenAI Images',
      503,
      'OPENAI_API_KEY is not configured.'
    );
  }

  const model =
    process.env.OPENAI_IMAGE_MODEL ||
    'gpt-image-1';

  const controller = new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    180000
  );

  try {
    const response = await fetch(
      'https://api.openai.com/v1/images/generations',
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json'
        },
        signal: controller.signal,
        body: JSON.stringify({
          model,
          prompt,
          n: 1,
          size: '1024x1024'
        })
      }
    );

    const raw = await response.text();

    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      throw providerError(
        'OpenAI Images',
        response.status || 502,
        'The image provider returned invalid JSON.'
      );
    }

    if (!response.ok) {
      throw providerError(
        'OpenAI Images',
        response.status,
        data?.error?.message ||
          'OpenAI image generation failed.'
      );
    }

    const item = data?.data?.[0];

    if (item?.b64_json) {
      return decodeImageData(item.b64_json);
    }

    if (item?.url) {
      return await fetchImageBuffer(item.url);
    }

    throw providerError(
      'OpenAI Images',
      502,
      'OpenAI returned no image data.'
    );
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'OpenAI Images',
      e.name === 'AbortError'
        ? 408
        : getErrorStatus(e),
      e.name === 'AbortError'
        ? 'OpenAI image generation timed out.'
        : e.message || 'Image generation failed.'
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ------------------------------------------------------------
// OPENROUTER IMAGE GENERATION
// ------------------------------------------------------------

async function generateOpenRouterImage(prompt) {
  const key = process.env.OPENROUTER_API_KEY;

  if (!key) {
    throw providerError(
      'OpenRouter Images',
      503,
      'OPENROUTER_API_KEY is not configured.'
    );
  }

  const model =
    process.env.OPENROUTER_IMAGE_MODEL ||
    'google/gemini-2.5-flash-image';

  const controller = new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    180000
  );

  try {
    const response = await fetch(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json',
          'HTTP-Referer':
            process.env.OPENROUTER_SITE_URL ||
            'http://localhost:3000',
          'X-Title':
            process.env.OPENROUTER_APP_NAME || 'Mora AI'
        },
        signal: controller.signal,
        body: JSON.stringify({
          model,
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text:
                    'Generate an image from this prompt: ' +
                    prompt
                }
              ]
            }
          ],
          modalities: ['image', 'text']
        })
      }
    );

    const raw = await response.text();

    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      throw providerError(
        'OpenRouter Images',
        response.status || 502,
        'The image provider returned invalid JSON.'
      );
    }

    if (!response.ok) {
      throw providerError(
        'OpenRouter Images',
        response.status,
        data?.error?.message ||
          'OpenRouter image generation failed.'
      );
    }

    const message =
      data?.choices?.[0]?.message;

    const image =
      message?.images?.[0]?.image_url?.url ||
      message?.images?.[0]?.url ||
      message?.content?.find?.(
        p => p?.type === 'image_url'
      )?.image_url?.url;

    if (typeof image === 'string') {
      if (image.startsWith('data:image/')) {
        return decodeImageData(image);
      }

      if (image.startsWith('https://')) {
        return await fetchImageBuffer(image);
      }
    }

    throw providerError(
      'OpenRouter Images',
      502,
      'No image was returned. Confirm that this model supports image generation on OpenRouter.'
    );
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'OpenRouter Images',
      e.name === 'AbortError'
        ? 408
        : getErrorStatus(e),
      e.name === 'AbortError'
        ? 'OpenRouter image generation timed out.'
        : e.message || 'Image generation failed.'
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ------------------------------------------------------------
// REPLICATE IMAGE GENERATION
// ------------------------------------------------------------

async function generateReplicateImage(prompt) {
  const key = process.env.REPLICATE_API_TOKEN;

  if (!key) {
    throw providerError(
      'Replicate Images',
      503,
      'REPLICATE_API_TOKEN is not configured.'
    );
  }

  const model =
    process.env.REPLICATE_IMAGE_MODEL ||
    'black-forest-labs/flux-schnell';

  const controller = new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    180000
  );

  try {
    const response = await fetch(
      'https://api.replicate.com/v1/models/' +
        model +
        '/predictions',
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${key}`,
          'Content-Type': 'application/json',
          Prefer: 'wait=60'
        },
        signal: controller.signal,
        body: JSON.stringify({
          input: {
            prompt
          }
        })
      }
    );

    const raw = await response.text();

    let data;

    try {
      data = JSON.parse(raw);
    } catch {
      throw providerError(
        'Replicate Images',
        response.status || 502,
        'The image provider returned invalid JSON.'
      );
    }

    if (!response.ok) {
      throw providerError(
        'Replicate Images',
        response.status,
        data?.detail ||
          data?.error ||
          'Replicate image generation failed.'
      );
    }

    let prediction = data;

    // Poll only the API URL supplied by Replicate.
    let pollUrl = prediction.urls?.get;

    while (
      ['starting', 'processing'].includes(
        prediction.status
      ) &&
      pollUrl
    ) {
      const parsed = new URL(pollUrl);

      if (
        parsed.protocol !== 'https:' ||
        parsed.hostname !== 'api.replicate.com'
      ) {
        throw new Error(
          'Replicate returned an unexpected polling URL.'
        );
      }

      const pollResponse = await fetch(parsed, {
        headers: {
          Authorization: `Bearer ${key}`
        },
        signal: controller.signal
      });

      const pollRaw = await pollResponse.text();

      try {
        prediction = JSON.parse(pollRaw);
      } catch {
        throw new Error(
          'Replicate returned invalid polling data.'
        );
      }

      if (!pollResponse.ok) {
        throw providerError(
          'Replicate Images',
          pollResponse.status,
          prediction?.detail ||
            'Could not check image-generation progress.'
        );
      }

      if (
        ['starting', 'processing'].includes(
          prediction.status
        )
      ) {
        await new Promise(resolve =>
          setTimeout(resolve, 1500)
        );
      }
    }

    if (prediction.status !== 'succeeded') {
      throw providerError(
        'Replicate Images',
        502,
        prediction?.error ||
          `Image generation ended with status: ${prediction.status || 'unknown'}.`
      );
    }

    const output = Array.isArray(prediction.output)
      ? prediction.output[0]
      : prediction.output;

    if (typeof output !== 'string') {
      throw providerError(
        'Replicate Images',
        502,
        'Replicate returned no image URL.'
      );
    }

    return await fetchImageBuffer(output);
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'Replicate Images',
      e.name === 'AbortError'
        ? 408
        : getErrorStatus(e),
      e.name === 'AbortError'
        ? 'Replicate image generation timed out.'
        : e.message || 'Image generation failed.'
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ------------------------------------------------------------
// IMAGE PROVIDER SELECTION AND FALLBACK
// ------------------------------------------------------------

function getImageProviderOrder() {
  return (
    process.env.IMAGE_PROVIDER_ORDER ||
    'gemini,openai,openrouter,replicate'
  )
    .split(',')
    .map(s => s.trim().toLowerCase())
    .filter(id =>
      ['gemini', 'openai', 'openrouter', 'replicate'].includes(id)
    );
}

function configuredImageProviders() {
  const configured = {
    gemini: Boolean(process.env.GEMINI_API_KEY),
    openai: Boolean(process.env.OPENAI_API_KEY),
    openrouter: Boolean(process.env.OPENROUTER_API_KEY),
    replicate: Boolean(process.env.REPLICATE_API_TOKEN)
  };

  return getImageProviderOrder().filter(
    id => configured[id]
  );
}

async function generateImage(prompt, requestedProvider = 'auto') {
  const allowed = [
    'auto',
    'gemini',
    'openai',
    'openrouter',
    'replicate'
  ];

  const requested = String(
    requestedProvider || 'auto'
  ).toLowerCase();

  if (!allowed.includes(requested)) {
    throw providerError(
      'Image configuration',
      400,
      'Unsupported image provider.'
    );
  }

  const all = configuredImageProviders();

  let providers =
    requested === 'auto'
      ? all
      : [
          requested,
          ...all.filter(id => id !== requested)
        ];

  const generators = {
    gemini: generateGeminiImage,
    openai: generateOpenAIImage,
    openrouter: generateOpenRouterImage,
    replicate: generateReplicateImage
  };

  const errors = [];

  for (const id of providers) {
    try {
      const image = await generators[id](prompt);

      const saved = await saveGeneratedImage(
        image.buffer,
        image.mime
      );

      return {
        success: true,
        provider: id,
        ...saved
      };
    } catch (e) {
      errors.push({
        provider: id,
        status: getErrorStatus(e),
        message: String(
          e.providerMessage ||
          e.message ||
          'Image generation failed.'
        ).slice(0, 500)
      });

      console.error(
        `[Mora AI] Image provider ${id} failed:`,
        e.message
      );
    }
  }

  const summary = errors
    .map(e => `${e.provider} (${e.status}): ${e.message}`)
    .join(' | ');

  throw providerError(
    'Mora AI Images',
    502,
    providers.length
      ? `All image providers failed. ${summary}`
      : 'No image provider is configured. Add an image API key to .env.'
  );
}

// ============================================================
// PART 3 — API ROUTES, PROJECTS, HISTORY, AND SERVER STARTUP
// ============================================================

// ------------------------------------------------------------
// INITIALIZE STORAGE
// ------------------------------------------------------------

await fs.mkdir(DATA_DIR, { recursive: true });
await fs.mkdir(IMAGE_DIR, { recursive: true });

for (const [file, fallback] of [
  [PROJECTS_FILE, []],
  [HISTORY_FILE, []],
  [CONVERSATIONS_FILE, []]
]) {
  try {
    await fs.access(file);
  } catch {
    await writeJson(file, fallback);
  }
}

// ------------------------------------------------------------
// HEALTH CHECK
// ------------------------------------------------------------

app.get('/api/health', (_req, res) => {
  res.json({
    success: true,
    app: 'Mora AI',
    status: 'online',
    time: new Date().toISOString()
  });
});

// ------------------------------------------------------------
// PROVIDER STATUS
// ------------------------------------------------------------

app.get('/api/providers', (_req, res) => {
  const textProviders = configuredProviderMap();

  const imageProviders = {
    gemini: Boolean(process.env.GEMINI_API_KEY),
    openai: Boolean(process.env.OPENAI_API_KEY),
    openrouter: Boolean(process.env.OPENROUTER_API_KEY),
    replicate: Boolean(process.env.REPLICATE_API_TOKEN)
  };

  res.json({
    success: true,
    text: Object.values(textProviders).map(p => ({
      id: p.id,
      name: p.name,
      configured: p.configured,
      model: p.model
    })),
    images: Object.entries(imageProviders).map(
      ([id, configured]) => ({
        id,
        configured
      })
    )
  });
});

// ------------------------------------------------------------
// TEXT PROVIDER CONNECTION TEST
// ------------------------------------------------------------

app.post('/api/test', async (req, res) => {
  try {
    const { providers } = providerCandidates({
      provider: req.body?.provider || 'auto',
      apiKey: req.body?.apiKey || '',
      model: req.body?.model || '',
      providerForKey: req.body?.providerForKey || '',
      fallback: false
    });

    if (!providers.length) {
      return res.status(400).json({
        success: false,
        error:
          'No configured text provider was found. Check your .env file.'
      });
    }

    const provider = providers[0];

    const result = await requestProviderText(
      'Return exactly this JSON object and nothing else: {"message":"Mora AI connection successful","steps":[],"files":[],"canvas":null,"execution":{"requested":false,"language":"python","code":""},"terminal":""}'
    );

    const parsed = parseJson(result.text);

    if (
      !parsed ||
      typeof parsed !== 'object' ||
      typeof parsed.message !== 'string'
    ) {
      throw new Error(
        'The provider responded, but the response format was unexpected.'
      );
    }

    res.json({
      success: true,
      provider: provider.id,
      model: result.model,
      message: 'Text AI provider responded successfully.'
    });
  } catch (e) {
    res.status(getErrorStatus(e)).json({
      success: false,
      error: e.message || 'Provider connection test failed.'
    });
  }
});

// ------------------------------------------------------------
// MAIN AI CHAT / AGENT ENDPOINT
// ------------------------------------------------------------

app.post('/api/agent', async (req, res) => {
  try {
    const body = req.body || {};

    const prompt = String(
      body.prompt || body.message || ''
    ).trim();

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: 'Please enter a message.'
      });
    }

    if (prompt.length > MAX_PROMPT) {
      return res.status(413).json({
        success: false,
        error:
          `Your message is too long. Maximum length is ${MAX_PROMPT} characters.`
      });
    }

    const options = {
      provider: body.provider || 'auto',
      providerForKey: body.providerForKey || '',
      apiKey:
        typeof body.apiKey === 'string'
          ? body.apiKey
          : '',
      model: body.model || '',
      fallback: body.fallback !== false
    };

    const result = await generateAIResponse(
      prompt,
      options
    );

    // Keep the output schema predictable for the frontend.
    const response = {
      success: true,
      message: result.message || '',
      steps: result.steps || [],
      files: result.files || [],
      canvas: result.canvas || null,
      execution: result.execution || {
        requested: false,
        language: 'python',
        code: ''
      },
      terminal: result.terminal || '',
      status: result.status || 'completed',
      provider: result.provider || '',
      model: result.model || ''
    };

    // Never return an unexplained blank response.
    if (
      !response.message.trim() &&
      !response.files.length &&
      !response.canvas
    ) {
      return res.status(502).json({
        success: false,
        error:
          'The AI provider returned no visible answer. Check the text API key, selected model, and server logs, then retry.'
      });
    }

    // Save the request and response without saving API keys.
    try {
      const history = await readJson(
        HISTORY_FILE,
        []
      );

      history.push({
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        prompt,
        message: response.message,
        provider: response.provider,
        model: response.model,
        files: response.files.map(f => f.name)
      });

      await writeJson(
        HISTORY_FILE,
        history.slice(-500)
      );
    } catch (historyError) {
      console.error(
        '[Mora AI] Could not save history:',
        historyError.message
      );
    }

    res.json(response);
  } catch (e) {
    console.error(
      '[Mora AI] Chat request failed:',
      e.message
    );

    res.status(getErrorStatus(e)).json({
      success: false,
      error:
        e.message ||
        'Mora AI could not generate a response.',
      providerErrors: IS_PRODUCTION
        ? undefined
        : e.providerErrors
    });
  }
});

// ------------------------------------------------------------
// IMAGE GENERATION ENDPOINT
// ------------------------------------------------------------

app.post(
  '/api/images/generate',
  async (req, res) => {
    try {
      const prompt = String(
        req.body?.prompt || ''
      ).trim();

      if (!prompt) {
        return res.status(400).json({
          success: false,
          error: 'Please provide an image prompt.'
        });
      }

      if (prompt.length > 4000) {
        return res.status(413).json({
          success: false,
          error:
            'The image prompt cannot exceed 4000 characters.'
        });
      }

      const provider = String(
        req.body?.provider || 'auto'
      ).toLowerCase();

      const result = await generateImage(
        prompt,
        provider
      );

      res.json({
        success: true,
        image: result.url,
        url: result.url,
        filename: result.filename,
        provider: result.provider,
        mime: result.mime,
        bytes: result.bytes
      });
    } catch (e) {
      console.error(
        '[Mora AI] Image generation failed:',
        e.message
      );

      res.status(getErrorStatus(e)).json({
        success: false,
        error:
          e.message ||
          'Image generation failed.'
      });
    }
  }
);

// ------------------------------------------------------------
// CONVERSATION HISTORY
// ------------------------------------------------------------

app.get('/api/history', async (_req, res) => {
  try {
    const history = await readJson(
      HISTORY_FILE,
      []
    );

    res.json({
      success: true,
      history: history.slice(-200).reverse()
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: 'Could not read conversation history.'
    });
  }
});

app.delete('/api/history', async (_req, res) => {
  try {
    await writeJson(HISTORY_FILE, []);

    res.json({
      success: true,
      message: 'Conversation history cleared.'
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: 'Could not clear conversation history.'
    });
  }
});

// ------------------------------------------------------------
// SAVED CONVERSATIONS
// ------------------------------------------------------------

app.get(
  '/api/conversations',
  async (_req, res) => {
    try {
      const conversations = await readJson(
        CONVERSATIONS_FILE,
        []
      );

      res.json({
        success: true,
        conversations
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not load conversations.'
      });
    }
  }
);

app.post(
  '/api/conversations',
  async (req, res) => {
    try {
      const body = req.body || {};

      const title = String(
        body.title || 'New conversation'
      )
        .trim()
        .slice(0, 120);

      const messages = Array.isArray(body.messages)
        ? body.messages.slice(-100).map(m => ({
            role:
              m?.role === 'assistant'
                ? 'assistant'
                : 'user',
            content: String(
              m?.content || ''
            ).slice(0, 12000),
            timestamp:
              m?.timestamp ||
              new Date().toISOString()
          }))
        : [];

      const conversations = await readJson(
        CONVERSATIONS_FILE,
        []
      );

      const conversation = {
        id: crypto.randomUUID(),
        title,
        messages,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      conversations.unshift(conversation);

      await writeJson(
        CONVERSATIONS_FILE,
        conversations.slice(0, 200)
      );

      res.status(201).json({
        success: true,
        conversation
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not save the conversation.'
      });
    }
  }
);

app.delete(
  '/api/conversations/:id',
  async (req, res) => {
    try {
      const conversations = await readJson(
        CONVERSATIONS_FILE,
        []
      );

      const remaining = conversations.filter(
        c => c.id !== req.params.id
      );

      if (remaining.length === conversations.length) {
        return res.status(404).json({
          success: false,
          error: 'Conversation not found.'
        });
      }

      await writeJson(
        CONVERSATIONS_FILE,
        remaining
      );

      res.json({
        success: true,
        message: 'Conversation deleted.'
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not delete the conversation.'
      });
    }
  }
);

// ------------------------------------------------------------
// PROJECT MANAGEMENT
// ------------------------------------------------------------

app.get('/api/projects', async (_req, res) => {
  try {
    const projects = await readJson(
      PROJECTS_FILE,
      []
    );

    res.json({
      success: true,
      projects
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: 'Could not load projects.'
    });
  }
});

app.post('/api/projects', async (req, res) => {
  try {
    const body = req.body || {};

    const name = String(
      body.name || 'Untitled project'
    )
      .trim()
      .slice(0, 120);

    const files = normalizeFiles(body.files);

    const project = {
      id: crypto.randomUUID(),
      name,
      description: String(
        body.description || ''
      ).slice(0, 2000),
      files,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    const projects = await readJson(
      PROJECTS_FILE,
      []
    );

    projects.unshift(project);

    await writeJson(
      PROJECTS_FILE,
      projects.slice(0, 200)
    );

    res.status(201).json({
      success: true,
      project
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: 'Could not save project.'
    });
  }
});

app.get(
  '/api/projects/:id',
  async (req, res) => {
    try {
      const projects = await readJson(
        PROJECTS_FILE,
        []
      );

      const project = projects.find(
        p => p.id === req.params.id
      );

      if (!project) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      res.json({
        success: true,
        project
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not load project.'
      });
    }
  }
);

app.put(
  '/api/projects/:id',
  async (req, res) => {
    try {
      const projects = await readJson(
        PROJECTS_FILE,
        []
      );

      const index = projects.findIndex(
        p => p.id === req.params.id
      );

      if (index < 0) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      const body = req.body || {};

      projects[index] = {
        ...projects[index],
        name: String(
          body.name ?? projects[index].name
        ).slice(0, 120),
        description: String(
          body.description ??
          projects[index].description ??
          ''
        ).slice(0, 2000),
        files: body.files
          ? normalizeFiles(body.files)
          : projects[index].files,
        updatedAt: new Date().toISOString()
      };

      await writeJson(
        PROJECTS_FILE,
        projects
      );

      res.json({
        success: true,
        project: projects[index]
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not update project.'
      });
    }
  }
);

app.delete(
  '/api/projects/:id',
  async (req, res) => {
    try {
      const projects = await readJson(
        PROJECTS_FILE,
        []
      );

      const remaining = projects.filter(
        p => p.id !== req.params.id
      );

      if (remaining.length === projects.length) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      await writeJson(
        PROJECTS_FILE,
        remaining
      );

      res.json({
        success: true,
        message: 'Project deleted.'
      });
    } catch (e) {
      res.status(500).json({
        success: false,
        error: 'Could not delete project.'
      });
    }
  }
);

// ------------------------------------------------------------
// DOWNLOAD PROJECT AS ZIP
// ------------------------------------------------------------

app.get(
  '/api/projects/:id/export',
  async (req, res) => {
    try {
      const projects = await readJson(
        PROJECTS_FILE,
        []
      );

      const project = projects.find(
        p => p.id === req.params.id
      );

      if (!project) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      const files = normalizeFiles(project.files);

      if (!files.length) {
        return res.status(400).json({
          success: false,
          error: 'This project contains no files.'
        });
      }

      res.setHeader(
        'Content-Type',
        'application/zip'
      );

      res.setHeader(
        'Content-Disposition',
        'attachment; filename="mora-project.zip"'
      );

      const archive = archiver('zip', {
        zlib: { level: 6 }
      });

      archive.on('error', err => {
        console.error(
          '[Mora AI] ZIP export failed:',
          err.message
        );

        if (!res.headersSent) {
          res.status(500).end();
        } else {
          res.destroy(err);
        }
      });

      archive.pipe(res);

      for (const file of files) {
        archive.append(
          file.content,
          {
            name: safeFilename(file.name)
          }
        );
      }

      await archive.finalize();
    } catch (e) {
      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: 'Could not export project.'
        });
      }
    }
  }
);

// ------------------------------------------------------------
// OPTIONAL PYTHON EXECUTION THROUGH E2B
// ------------------------------------------------------------

app.post(
  '/api/execute/python',
  async (req, res) => {
    let sandbox;

    try {
      const code = String(
        req.body?.code || ''
      );

      if (!code.trim()) {
        return res.status(400).json({
          success: false,
          error: 'Please provide Python code.'
        });
      }

      if (code.length > 12000) {
        return res.status(413).json({
          success: false,
          error: 'Python code is too long.'
        });
      }

      const apiKey = process.env.E2B_API_KEY;

      if (!apiKey) {
        return res.status(503).json({
          success: false,
          error:
            'Python execution is not configured. Add E2B_API_KEY to .env.'
        });
      }

      sandbox = await Sandbox.create({
        apiKey
      });

      const execution = await sandbox.runCode(
        code,
        {
          language: 'python'
        }
      );

      res.json({
        success: true,
        stdout: String(
          execution?.logs?.stdout?.join('\n') || ''
        ).slice(0, 12000),
        stderr: String(
          execution?.logs?.stderr?.join('\n') || ''
        ).slice(0, 12000),
        results: Array.isArray(execution?.results)
          ? execution.results.map(r => ({
              text: String(r?.text || '').slice(0, 4000)
            }))
          : []
      });
    } catch (e) {
      console.error(
        '[Mora AI] Python execution failed:',
        e.message
      );

      res.status(getErrorStatus(e)).json({
        success: false,
        error:
          e.message ||
          'Python execution failed.'
      });
    } finally {
      if (sandbox) {
        try {
          await sandbox.close();
        } catch {}
      }
    }
  }
);

// ------------------------------------------------------------
// API NOT FOUND
// ------------------------------------------------------------

app.use('/api', (_req, res) => {
  res.status(404).json({
    success: false,
    error: 'Mora AI API endpoint not found.'
  });
});

// ------------------------------------------------------------
// FRONTEND FALLBACK
// ------------------------------------------------------------

app.use((req, res, next) => {
  if (req.method !== 'GET') {
    return next();
  }

  if (
    req.path.startsWith('/api/') ||
    req.path.startsWith('/generated-images/')
  ) {
    return next();
  }

  res.sendFile(
    path.join(ROOT, 'public', 'index.html'),
    err => {
      if (err) next(err);
    }
  );
});

// ------------------------------------------------------------
// GLOBAL ERROR HANDLER
// ------------------------------------------------------------

app.use((err, req, res, _next) => {
  console.error(
    '[Mora AI] Server error:',
    err.message
  );

  if (res.headersSent) {
    return res.end();
  }

  const status = Number(err.status || 500);

  res.status(
    status >= 400 && status < 600
      ? status
      : 500
  ).json({
    success: false,
    error:
      status >= 500 && IS_PRODUCTION
        ? 'An internal server error occurred. Check the server logs.'
        : err.message || 'An unexpected error occurred.'
  });
});

// ------------------------------------------------------------
// START SERVER
// ------------------------------------------------------------

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log('');
  console.log('========================================');
  console.log('          MORA AI SERVER');
  console.log('========================================');
  console.log(`Port: ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`Text providers: ${configuredProviders().map(p => p.id).join(', ') || 'NONE CONFIGURED'}`);
  console.log(`Image providers: ${configuredImageProviders().join(', ') || 'NONE CONFIGURED'}`);
  console.log(`Frontend: http://localhost:${PORT}`);
  console.log('========================================');
  console.log('');
});

server.requestTimeout = 190000;
server.headersTimeout = 195000;
server.keepAliveTimeout = 5000;

async function shutdown(signal) {
  console.log(`[Mora AI] Received ${signal}; shutting down.`);

  server.close(() => {
    console.log('[Mora AI] Server stopped.');
    process.exit(0);
  });

  setTimeout(() => {
    console.error('[Mora AI] Forced shutdown.');
    process.exit(1);
  }, 10000).unref();
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

process.on('unhandledRejection', reason => {
  console.error(
    '[Mora AI] Unhandled promise rejection:',
    reason
  );
});

process.on('uncaughtException', error => {
  console.error(
    '[Mora AI] Uncaught exception:',
    error
  );

  process.exit(1);
});

// END OF PART 3 — SERVER.JS COMPLETE
