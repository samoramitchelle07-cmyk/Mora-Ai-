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

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
const GROQ_MODEL = process.env.GROQ_MODEL || 'openai/gpt-oss-120b';
const OPENROUTER_MODEL =
  process.env.OPENROUTER_MODEL || 'microsoft/microsoft-decision-1';

const AI_PROVIDER_ORDER = (
  process.env.AI_PROVIDER_ORDER || 'gemini,groq,openrouter'
)
  .split(',')
  .map(s => s.trim().toLowerCase())
  .filter(Boolean);

const AI_OUTPUT_TOKENS = Math.max(
  1024,
  Math.min(32768, Number(process.env.AI_OUTPUT_TOKENS || 14000))
);

const DATA_DIR = path.join(ROOT, 'data');
const PROJECTS_FILE = path.join(DATA_DIR, 'projects.json');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');

const MAX_PROMPT = 12000;
const MAX_FILE_CHARS = 180000;
const MAX_FILES = 25;
const MAX_JSON_BODY = '1mb';

const IS_PRODUCTION = process.env.NODE_ENV === 'production';
const TOKEN = (process.env.MORA_API_TOKEN || '').trim();

if (IS_PRODUCTION && !TOKEN) {
  console.error(
    'SECURITY ERROR: Set MORA_API_TOKEN before exposing Mora AI in production.'
  );
  process.exit(1);
}

const ai = process.env.GEMINI_API_KEY
  ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })
  : null;

const app = express();

app.disable('x-powered-by');

app.set(
  'trust proxy',
  process.env.TRUST_PROXY === '1' ? 1 : false
);

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

/* =========================================================
   AUTHENTICATION
========================================================= */

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
      error: 'Unauthorized. Configure the Mora API token in the client or server.'
    });
  }

  next();
}

app.use('/api/agent', auth);
app.use('/api/projects', auth);
app.use('/api/history', auth);
app.use('/api/test', auth);

/* =========================================================
   STATIC FRONTEND
========================================================= */

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

/* =========================================================
   JSON STORAGE
========================================================= */

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

  const tmp = `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;

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

/* =========================================================
   FILE UTILITIES
========================================================= */

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

/* =========================================================
   JSON RESPONSE PARSING
========================================================= */

function parseJson(text) {
  if (!text) {
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

  return {
    success: true,
    message: String(raw?.message || 'Finished.').slice(0, 12000),

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

/* =========================================================
   AI SYSTEM INSTRUCTIONS
========================================================= */

const SYSTEM_PROMPT = `
You are Mora AI, a practical personal AI agent for coding,
debugging, building websites, research, learning, and general assistance.

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
- Return valid JSON only.
- Do not reveal private chain of thought.
- For ordinary questions, answer in message and use empty files,
  canvas null and execution.requested false.
- For website/interface requests, provide complete standalone HTML
  in canvas.html and normally a matching index.html in files.
- Use inline CSS and JavaScript for preview portability.
- For code requests, return complete files, not snippets.
- Do not claim code was tested unless it actually ran.
- Request execution only for Python when the user asks to run/test
  Python or needs a Python calculation.
- Never execute shell commands, JavaScript, destructive code,
  credential theft, unauthorized access, or code that evades security.
- Use concise progress steps.
- Keep files under 180000 characters and total output reasonable.
`;

/* =========================================================
   STREAMED MESSAGE EXTRACTION
========================================================= */

function extractPartialMessage(raw) {
  const match = /"message"\s*:\s*"/.exec(raw);

  if (!match) return '';

  let out = '';
  let escaped = false;

  const start = match.index + match[0].length;

  for (let i = start; i < raw.length; i++) {
    const ch = raw[i];

    if (escaped) {
      escaped = false;

      if (ch === 'n') out += '\n';
      else if (ch === 'r') out += '\r';
      else if (ch === 't') out += '\t';
      else if (ch === 'b') out += '\b';
      else if (ch === 'f') out += '\f';
      else if (ch === 'u') {
        const hex = raw.slice(i + 1, i + 5);

        if (/^[0-9a-f]{4}$/i.test(hex)) {
          out += String.fromCharCode(parseInt(hex, 16));
          i += 4;
        } else {
          break;
        }
      } else {
        out += ch;
      }

      continue;
    }

    if (ch === '\\') {
      escaped = true;
      continue;
    }

    if (ch === '"') break;

    out += ch;
  }

  return out;
}

/* =========================================================
   AI PROVIDER CONFIGURATION
========================================================= */

function configuredProviders() {
  const settings = {
    gemini: {
      id: 'gemini',
      name: 'Gemini',
      configured: Boolean(
        process.env.GEMINI_API_KEY?.trim()
      ),
      model: MODEL
    },

    groq: {
      id: 'groq',
      name: 'Groq',
      configured: Boolean(
        process.env.GROQ_API_KEY?.trim()
      ),
      model: GROQ_MODEL
    },

    openrouter: {
      id: 'openrouter',
      name: 'OpenRouter',
      configured: Boolean(
        process.env.OPENROUTER_API_KEY?.trim()
      ),
      model: OPENROUTER_MODEL
    }
  };

  // Preserve the preferred order, but never silently exclude
  // another provider that has a configured API key.
  const preferredOrder = [
    ...AI_PROVIDER_ORDER,
    'gemini',
    'groq',
    'openrouter'
  ];

  const seen = new Set();
  const providers = [];

  for (const id of preferredOrder) {
    const provider = settings[id];

    if (
      !provider ||
      !provider.configured ||
      seen.has(id)
    ) {
      continue;
    }

    seen.add(id);

    providers.push({
      id: provider.id,
      name: provider.name,
      model: provider.model,
      configured: true
    });
  }

  return providers;
}
/* =========================================================
   GEMINI PROVIDER
========================================================= */

async function requestGeminiText(prompt, repairContext = '') {
  if (!ai) {
    throw providerError(
      'Gemini',
      503,
      'GEMINI_API_KEY is not configured.'
    );
  }

  const contents =
    `${SYSTEM_PROMPT}\n\nUSER REQUEST:\n${prompt}` +
    (repairContext
      ? `\n\nREPAIR CONTEXT:\n${repairContext}`
      : '');

  try {
    const response = await ai.models.generateContent({
      model: MODEL,
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
      model: MODEL
    };
  } catch (e) {
    if (e.provider) throw e;

    throw providerError(
      'Gemini',
      function getErrorStatus(error) {
  const directStatus =
    Number(error?.status) ||
    Number(error?.statusCode) ||
    Number(error?.response?.status);

  if (directStatus > 0) {
    return directStatus;
  }

  // Some SDK errors contain the HTTP status only in their message.
  const message = String(
    error?.message || error || ''
  );

  const match = message.match(
    /\b(?:HTTP\s*)?(400|401|402|403|404|408|413|429|500|502|503|504)\b/i
  );

  return match ? Number(match[1]) : 502;
}
/* =========================================================
   GROQ AND OPENROUTER PROVIDERS
========================================================= */

async function requestCompatibleText(
  providerId,
  prompt,
  repairContext = ''
) {
  const isGroq = providerId === 'groq';

  const providerName = isGroq ? 'Groq' : 'OpenRouter';

  const apiKey = isGroq
    ? process.env.GROQ_API_KEY
    : process.env.OPENROUTER_API_KEY;

  const model = isGroq ? GROQ_MODEL : OPENROUTER_MODEL;

  const endpoint = isGroq
    ? 'https://api.groq.com/openai/v1/chat/completions'
    : 'https://openrouter.ai/api/v1/chat/completions';

  if (!apiKey) {
    throw providerError(
      providerName,
      503,
      'API key is not configured.'
    );
  }

  const userContent = repairContext
    ? `Original request:\n${prompt}\n\nRepair context:\n${repairContext}\n\nReturn the complete corrected JSON object.`
    : prompt;

  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json'
  };

  if (!isGroq) {
    headers['X-Title'] = 'Mora AI';

    if (process.env.OPENROUTER_SITE_URL) {
      headers['HTTP-Referer'] =
        process.env.OPENROUTER_SITE_URL;
    }
  }

  let response;

  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers,
      signal: AbortSignal.timeout(120000),

      body: JSON.stringify({
        model,

        messages: [
          {
            role: 'system',
            content: SYSTEM_PROMPT
          },
          {
            role: 'user',
            content: userContent
          }
        ],

        max_tokens: AI_OUTPUT_TOKENS,
        temperature: 0.2
      })
    });
  } catch (e) {
    throw providerError(
      providerName,
      502,
      e.message || 'Network request failed.'
    );
  }

  const rawBody = await response.text();

  let body;

  try {
    body = JSON.parse(rawBody);
  } catch {
    body = null;
  }

  if (!response.ok) {
    const message =
      body?.error?.message ||
      body?.message ||
      rawBody.slice(0, 500) ||
      `HTTP ${response.status}`;

    throw providerError(
      providerName,
      response.status,
      message
    );
  }

  const content = body?.choices?.[0]?.message?.content;

  const text = typeof content === 'string'
    ? content
    : Array.isArray(content)
      ? content
          .map(part =>
            typeof part === 'string'
              ? part
              : part?.text || ''
          )
          .join('')
      : '';

  if (!text.trim()) {
    throw providerError(
      providerName,
      502,
      'The model returned an empty response.'
    );
  }

  return {
    text,
    model
  };
}

/* =========================================================
   FALLBACK POLICY
========================================================= */

function shouldFallback(error) {
  const status = getErrorStatus(error);

  if (
    [
      400, 401, 403, 404, 408, 413, 429,
      500, 502, 503, 504
    ].includes(status) ||
    status >= 500
  ) {
    return true;
  }

  const message = String(
    error?.message || ''
  ).toLowerCase();

  return /quota|rate.?limit|resource_exhausted|timeout|network|empty response|parse the ai response|incomplete json|invalid json/.test(message);
}

/* =========================================================
   MULTI-PROVIDER AI ROUTER
========================================================= */

// Retain the existing askGemini() function name so existing routes
// can use the new provider router without changing their interface.

async function askGemini(
  prompt,
  repairContext = '',
  onChunk = null
) {
  const providers = configuredProviders();

  if (!providers.length) {
    throw new Error(
      'No AI provider is configured. Add GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in Render Environment.'
    );
  }

  const failures = [];

  for (const provider of providers) {
    console.log(
      `[Mora AI] Trying ${provider.name} (${provider.model})`
    );

    try {
      const generated = provider.id === 'gemini'
        ? await requestGeminiText(prompt, repairContext)
        : await requestCompatibleText(
            provider.id,
            prompt,
            repairContext
          );

      // Parse and validate before accepting a provider's response.
      const parsed = parseJson(generated.text);

      if (
        !parsed ||
        typeof parsed !== 'object' ||
        Array.isArray(parsed) ||
        !(
          typeof parsed.message === 'string' ||
          Array.isArray(parsed.files) ||
          (
            parsed.canvas &&
            typeof parsed.canvas === 'object'
          ) ||
          (
            parsed.execution &&
            typeof parsed.execution === 'object'
          )
        )
      ) {
        throw new Error(
          'The AI response did not match Mora AI’s required JSON structure.'
        );
      }

      const result = normalizeResult(parsed, prompt);

      result.provider = provider.id;
      result.model = generated.model;

      console.log(
        `[Mora AI] Success: ${provider.name}`
      );

      // Only send a validated result, preventing partial output from
      // a failed provider from being mixed with the fallback response.
      if (typeof onChunk === 'function') {
        onChunk({
          chunkLength: result.message.length,
          accumulated: result.message.length,
          raw: generated.text,
          partialMessage: result.message,
          provider: provider.id,
          model: generated.model
        });
      }

      return result;
    } catch (e) {
      const failure = {
        provider: provider.name,
        model: provider.model,
        status: getErrorStatus(e),
        message: String(
          e.providerMessage || e.message || 'Request failed'
        ).slice(0, 700)
      };

      failures.push(failure);

      console.error(
        `[Mora AI] ${provider.name} failed (${failure.status}): ${failure.message}`
      );

      if (!shouldFallback(e)) break;
    }
  }

  const error = new Error(
    `All configured AI providers failed. ${
      failures
        .map(f => `${f.provider}: ${f.message}`)
        .join(' | ')
    }`
  );

  error.failures = failures;

  error.status =
    failures.length &&
    failures.every(f => f.status === 429)
      ? 429
      : 502;

  throw error;
}

/* =========================================================
   PYTHON SAFETY FILTERS
========================================================= */

function unsafePythonReason(code) {
  const checks = [
    [
      /\b(subprocess|socket|ftplib|telnetlib|paramiko)\b/i,
      'network/process-control imports are not permitted'
    ],
    [
      /\b(os\.system|os\.popen|pty\.spawn|Popen\s*\(|execv\s*\(|spawn\s*\()/i,
      'shell/process execution is not permitted'
    ],
    [
      /\b(requests\.(get|post|put|delete)|urllib\.request|httpx\.|aiohttp\.)/i,
      'outbound network requests are not permitted'
    ],
    [
      /\b(os\.environ|environ\s*\[|getenv\s*\()/i,
      'environment-variable access is not permitted'
    ],
    [
      /\b(shutil\.rmtree|Path\([^\n]*\)\.unlink|remove\s*\(|rmdir\s*\()/i,
      'filesystem deletion patterns are not permitted'
    ]
  ];

  for (const [re, message] of checks) {
    if (re.test(code)) return message;
  }

  return '';
}

/* =========================================================
   E2B PYTHON EXECUTION
========================================================= */

async function executePython(code) {
  if (!process.env.E2B_API_KEY) {
    return {
      ok: false,
      output: 'E2B_API_KEY is not configured; execution skipped.'
    };
  }

  if (!code.trim()) {
    return {
      ok: false,
      output: 'No Python code was provided.'
    };
  }

  if (code.length > 12000) {
    return {
      ok: false,
      output: 'Execution blocked: Python code exceeds 12,000 characters.'
    };
  }

  const reason = unsafePythonReason(code);

  if (reason) {
    return {
      ok: false,
      blocked: true,
      output: `Execution blocked by Mora AI safety filter: ${reason}.`
    };
  }

  let sandbox;

  try {
    sandbox = await Sandbox.create({
      timeoutMs: 60000
    });

    const result = await sandbox.runCode(code, {
      timeoutMs: 20000
    });

    const stdout = Array.isArray(result.logs?.stdout)
      ? result.logs.stdout.join('\n')
      : '';

    const stderr = Array.isArray(result.logs?.stderr)
      ? result.logs.stderr.join('\n')
      : '';

    const errorText = result.error
      ? (
          result.error.traceback ||
          result.error.name ||
          String(result.error)
        )
      : '';

    const output = [
      result.text,
      stdout,
      stderr,
      errorText
    ]
      .filter(Boolean)
      .join('\n')
      .slice(0, 10000) ||
      'Execution completed with no printed output.';

    return {
      ok: !result.error,
      output
    };
  } catch (e) {
    return {
      ok: false,
      output: `E2B execution failed: ${String(
        e.message || e
      ).slice(0, 1200)}`
    };
  } finally {
    if (sandbox?.kill) {
      try {
        await sandbox.kill();
      } catch {}
    }
  }
}

/* =========================================================
   HISTORY
========================================================= */

async function saveHistory(prompt, result) {
  try {
    const history = await readJson(
      HISTORY_FILE,
      []
    );

    history.unshift({
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      prompt,
      message: result.message.slice(0, 4000),
      files: result.files.map(f => f.name),
      status: result.status,
      provider: result.provider || null,
      model: result.model || null
    });

    await writeJson(
      HISTORY_FILE,
      history.slice(0, 100)
    );
  } catch (e) {
    console.error(
      'History save warning:',
      e.message
    );
  }
}

/* =========================================================
   AGENT WORKFLOW AND AUTOMATIC PYTHON REPAIR
========================================================= */

async function runAgent(prompt, notify = () => {}) {
  let result = await askGemini(prompt, '', () => {});

  notify({
    stage: 'generated',
    message: 'Initial response generated.'
  });

  if (result.execution.requested) {
    if (result.execution.language !== 'python') {
      result.terminal =
        'Execution skipped: Mora AI currently permits Python only.';
    } else {
      notify({
        stage: 'testing',
        message: 'Running Python in isolated E2B sandbox…'
      });

      const run = await executePython(
        result.execution.code
      );

      result.terminal = run.output;

      result.steps.push({
        thought: run.ok
          ? 'Python test completed.'
          : 'Python test did not pass.',
        command: 'test-python',
        output: run.output.slice(0, 2500),
        isError: !run.ok
      });

      if (
        !run.ok &&
        !run.blocked &&
        !/not configured/i.test(run.output)
      ) {
        notify({
          stage: 'repairing',
          message: 'Analyzing the error and generating one repair attempt…'
        });

        try {
          const repair = await askGemini(
            prompt,
            `Python execution failed. Error/output:\n${run.output}\nCode that failed:\n${result.execution.code}\nReturn a corrected solution. If appropriate, include corrected Python in execution.code. Do not repeat unsafe behavior.`
          );

          if (repair.files.length) {
            result.files = repair.files;
          }

          if (repair.canvas) {
            result.canvas = repair.canvas;
          }

          result.message +=
            `\n\nRepair attempt: ${repair.message}`;

          if (
            repair.execution.requested &&
            repair.execution.language === 'python'
          ) {
            const second = await executePython(
              repair.execution.code
            );

            result.terminal +=
              `\n\n--- Repair attempt ---\n${second.output}`;

            result.steps.push({
              thought: second.ok
                ? 'Repair attempt passed.'
                : 'Repair attempt did not pass.',
              command: 'repair-test',
              output: second.output.slice(0, 2500),
              isError: !second.ok
            });
          }
        } catch (e) {
          result.steps.push({
            thought: 'Automatic repair was unavailable.',
            command: 'auto-repair',
            output: String(e.message || e).slice(0, 1000),
            isError: true
          });
        }
      }
    }
  }

  result.status = 'completed';

  await saveHistory(prompt, result);

  return result;
}

/* =========================================================
   HEALTH CHECK
========================================================= */

app.get('/api/health', (_req, res) => {
  const providers = configuredProviders();

  res.json({
    success: true,
    status: 'online',
    service: 'Mora AI',

    providers: providers.map(p => ({
      id: p.id,
      name: p.name,
      model: p.model,
      configured: true
    })),

    providerOrder: providers.map(p => p.id),

    e2bConfigured: Boolean(
      process.env.E2B_API_KEY
    ),

    authRequired: Boolean(TOKEN),

    projectStorage: 'local-json-single-user',
    browserTesting: 'optional-playwright',

    timestamp: new Date().toISOString()
  });
});

/* =========================================================
   PROMPT VALIDATION
========================================================= */

function validatePrompt(req, res) {
  const prompt =
    typeof req.body?.prompt === 'string'
      ? req.body.prompt.trim()
      : '';

  if (!prompt) {
    res.status(400).json({
      success: false,
      error: 'Enter a prompt first.'
    });

    return null;
  }

  if (prompt.length > MAX_PROMPT) {
    res.status(413).json({
      success: false,
      error: `Prompt exceeds ${MAX_PROMPT} characters.`
    });

    return null;
  }

  if (!configuredProviders().length) {
    res.status(503).json({
      success: false,
      error: 'No AI provider is configured. Add at least one provider API key in Render Environment.'
    });

    return null;
  }

  return prompt;
}

function sendSse(res, event, data) {
  res.write(
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  );
}

/* =========================================================
   STREAMING AGENT ENDPOINT
========================================================= */

app.post('/api/agent/stream', async (req, res) => {
  const prompt = validatePrompt(req, res);

  if (!prompt) return;

  res.status(200).set({
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no'
  });

  res.flushHeaders?.();

  let closed = false;

  res.on('close', () => {
    if (!res.writableEnded) closed = true;
  });

  try {
    sendSse(res, 'progress', {
      stage: 'planning',
      message: 'Mora AI is analyzing your request…'
    });

    // Only validated output from the successful provider is emitted.
    const result = await askGemini(
      prompt,
      '',
      info => {
        if (closed) return;

        sendSse(res, 'progress', {
          stage: 'generating',
          message: `Response generated by ${info.provider}.`
        });

        if (info.partialMessage) {
          sendSse(res, 'delta', {
            text: info.partialMessage
          });
        }
      }
    );

    if (!closed) {
      sendSse(res, 'progress', {
        stage: 'generated',
        message: 'Response generated. Preparing tests and files…'
      });
    }

    if (result.execution.requested) {
      if (result.execution.language !== 'python') {
        result.terminal =
          'Execution skipped: only Python is enabled.';
      } else {
        if (!closed) {
          sendSse(res, 'progress', {
            stage: 'testing',
            message: 'Running Python in isolated E2B sandbox…'
          });
        }

        const run = await executePython(
          result.execution.code
        );

        result.terminal = run.output;

        result.steps.push({
          thought: run.ok
            ? 'Python test completed.'
            : 'Python test did not pass.',
          command: 'test-python',
          output: run.output.slice(0, 2500),
          isError: !run.ok
        });

        if (
          !run.ok &&
          !run.blocked &&
          !/not configured/i.test(run.output)
        ) {
          if (!closed) {
            sendSse(res, 'progress', {
              stage: 'repairing',
              message: 'Analyzing error and trying one automatic repair…'
            });
          }

          try {
            const repair = await askGemini(
              prompt,
              `Python execution failed. Error/output:\n${run.output}\nOriginal code:\n${result.execution.code}\nProvide a corrected solution. If appropriate, include corrected Python in execution.code.`
            );

            if (repair.files.length) {
              result.files = repair.files;
            }

            if (repair.canvas) {
              result.canvas = repair.canvas;
            }

            result.message +=
              `\n\nRepair attempt: ${repair.message}`;

            if (
              repair.execution.requested &&
              repair.execution.language === 'python'
            ) {
              const second = await executePython(
                repair.execution.code
              );

              result.terminal +=
                `\n\n--- Repair attempt ---\n${second.output}`;

              result.steps.push({
                thought: second.ok
                  ? 'Repair test passed.'
                  : 'Repair test did not pass.',
                command: 'repair-test',
                output: second.output.slice(0, 2500),
                isError: !second.ok
              });
            }
          } catch (e) {
            result.steps.push({
              thought: 'Automatic repair failed.',
              command: 'auto-repair',
              output: String(e.message || e).slice(0, 1000),
              isError: true
            });
          }
        }
      }
    }

    result.status = 'completed';

    await saveHistory(prompt, result);

    if (!closed) {
      sendSse(res, 'result', result);
      sendSse(res, 'done', { ok: true });
      res.end();
    }
  } catch (e) {
    console.error('Agent stream error:', e);

    if (!closed) {
      sendSse(res, 'error', {
        error: friendlyError(e)
      });

      res.end();
    }
  }
});

/* =========================================================
   STANDARD AGENT ENDPOINT
========================================================= */

app.post('/api/agent', async (req, res) => {
  const prompt = validatePrompt(req, res);

  if (!prompt) return;

  try {
    res.json(await runAgent(prompt));
  } catch (e) {
    console.error('Agent error:', e);

    res.status(errorStatus(e)).json({
      success: false,
      error: friendlyError(e)
    });
  }
});

/* =========================================================
   USER-FRIENDLY ERRORS
========================================================= */

function friendlyError(e) {
  const msg = String(
    e?.message || e || 'Unknown error'
  );

  if (
    e?.status === 429 ||
    /all configured ai providers failed.*(quota|rate.?limit)/i.test(msg)
  ) {
    return `All available AI providers are currently rate-limited or out of quota. Wait and try again. Details: ${msg.slice(0, 700)}`;
  }

  if (
    /api key|unauthorized|authentication failed|invalid api key|permission/i.test(msg)
  ) {
    return `An AI provider rejected its API key or permissions. Check the provider keys and model access in Render Environment. Details: ${msg.slice(0, 700)}`;
  }

  if (
    /model.*(not found|unknown|unavailable)|no such model/i.test(msg)
  ) {
    return `An AI provider rejected its configured model ID. Verify GEMINI_MODEL, GROQ_MODEL, and OPENROUTER_MODEL in Render Environment. Details: ${msg.slice(0, 700)}`;
  }

  if (
    /could not parse the ai response|empty response/i.test(msg)
  ) {
    return `The configured AI providers could not return a valid structured response. Try again or review the model settings. Details: ${msg.slice(0, 700)}`;
  }

  return `Mora AI request failed: ${msg.slice(0, 900)}`;
}

function errorStatus(e) {
  if (
    Number(e?.status) === 429 ||
    /quota|rate.?limit|resource_exhausted/i.test(
      String(e?.message || '')
    )
  ) {
    return 429;
  }

  return 502;
}

/* =========================================================
   HTML BROWSER TESTING
========================================================= */

app.post('/api/test/html', async (req, res) => {
  const html =
    typeof req.body?.html === 'string'
      ? req.body.html
      : '';

  if (!html.trim()) {
    return res.status(400).json({
      success: false,
      error: 'No HTML supplied to test.'
    });
  }

  if (html.length > 180000) {
    return res.status(413).json({
      success: false,
      error: 'HTML exceeds the 180,000 character testing limit.'
    });
  }

  let chromium;

  try {
    ({ chromium } = await import('playwright'));
  } catch {
    return res.status(503).json({
      success: false,
      error: 'Browser testing is not installed. On supported Linux/Windows/macOS, run: npm install and npx playwright install chromium. Some Android Termux environments cannot run Playwright browsers; use a supported host for this feature.'
    });
  }

  let browser;

  try {
    browser = await chromium.launch({
      headless: true,
      args: ['--no-sandbox']
    });

    const context = await browser.newContext({
      viewport: {
        width: 1280,
        height: 800
      },
      javaScriptEnabled: true,
      serviceWorkers: 'block'
    });

    // Block external network requests during browser testing.
    await context.route('**/*', route => {
      const url = route.request().url();

      if (
        url === 'about:blank' ||
        url.startsWith('data:') ||
        url.startsWith('blob:')
      ) {
        return route.continue();
      }

      return route.abort('blockedbyclient');
    });

    const page = await context.newPage();

    const consoleErrors = [];
    const pageErrors = [];

    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(
          msg.text().slice(0, 500)
        );
      }
    });

    page.on('pageerror', err => {
      pageErrors.push(
        String(err.message || err).slice(0, 500)
      );
    });

    await page.setContent(html, {
      waitUntil: 'domcontentloaded',
      timeout: 10000
    });

    await page.waitForTimeout(500);

    const details = await page.evaluate(() => ({
      title: document.title || '(no title)',
      bodyTextLength: (
        document.body?.innerText || ''
      ).trim().length,
      elementCount: document.querySelectorAll('*').length,
      hasViewport: Boolean(
        document.querySelector('meta[name="viewport"]')
      ),
      horizontalOverflow:
        document.documentElement.scrollWidth >
        document.documentElement.clientWidth
    }));

    const passed =
      pageErrors.length === 0 &&
      consoleErrors.length === 0;

    res.json({
      success: true,
      passed,
      browser: 'Chromium (Playwright)',
      details,
      consoleErrors: consoleErrors.slice(0, 20),
      pageErrors: pageErrors.slice(0, 20),
      note: 'External network requests were blocked during this test. This is a smoke test, not a complete security audit.'
    });

    await context.close();
  } catch (e) {
    if (!res.headersSent) {
      res.status(500).json({
        success: false,
        error: `Browser test failed: ${String(
          e.message || e
        ).slice(0, 700)}`
      });
    }
  } finally {
    if (browser) {
      try {
        await browser.close();
      } catch {}
    }
  }
});

/* =========================================================
   HISTORY ROUTE
========================================================= */

app.get('/api/history', async (_req, res) => {
  try {
    res.json({
      success: true,
      history: await readJson(HISTORY_FILE, [])
    });
  } catch (e) {
    res.status(500).json({
      success: false,
      error: e.message
    });
  }
});

/* =========================================================
   PROJECT LIST
========================================================= */

app.get('/api/projects', async (_req, res) => {
  try {
    const projects = await readJson(
      PROJECTS_FILE,
      []
    );

    res.json({
      success: true,

      projects: projects.map(
        ({ id, title, updatedAt, createdAt, prompt, files }) => ({
          id,
          title,
          updatedAt,
          createdAt,
          prompt,

          files: (files || []).map(f => ({
            name: f.name,
            path: f.path,
            language: f.language
          }))
        })
      )
    });
  } catch {
    res.status(500).json({
      success: false,
      error: 'Could not read project storage.'
    });
  }
});

/* =========================================================
   SAVE PROJECT
========================================================= */

app.post('/api/projects', async (req, res) => {
  const title = String(
    req.body?.title || 'Untitled Mora AI Project'
  )
    .trim()
    .slice(0, 120);

  const prompt = String(
    req.body?.prompt || ''
  ).slice(0, MAX_PROMPT);

  const files = normalizeFiles(req.body?.files);

  if (!files.length) {
    return res.status(400).json({
      success: false,
      error: 'Add at least one generated file before saving.'
    });
  }

  if (
    files.reduce((n, f) => n + f.content.length, 0) > 500000
  ) {
    return res.status(413).json({
      success: false,
      error: 'Project is too large to save (maximum 500 KB of file content).'
    });
  }

  try {
    const projects = await readJson(
      PROJECTS_FILE,
      []
    );

    const id =
      req.body?.id &&
      /^[a-f0-9-]{20,40}$/i.test(req.body.id)
        ? req.body.id
        : crypto.randomUUID();

    const now = new Date().toISOString();

    const existing = projects.find(
      p => p.id === id
    );

    const project = {
      id,
      title,
      prompt,
      files,
      createdAt: existing?.createdAt || now,
      updatedAt: now
    };

    await writeJson(
      PROJECTS_FILE,
      [
        project,
        ...projects.filter(p => p.id !== id)
      ].slice(0, 100)
    );

    res.json({
      success: true,
      project
    });
  } catch {
    res.status(500).json({
      success: false,
      error: 'Could not save project.'
    });
  }
});

/* =========================================================
   LOAD PROJECT
========================================================= */

app.get('/api/projects/:id', async (req, res) => {
  try {
    const p = (
      await readJson(PROJECTS_FILE, [])
    ).find(x => x.id === req.params.id);

    if (!p) {
      return res.status(404).json({
        success: false,
        error: 'Project not found.'
      });
    }

    res.json({
      success: true,
      project: p
    });
  } catch {
    res.status(500).json({
      success: false,
      error: 'Could not load project.'
    });
  }
});

/* =========================================================
   EXPORT PROJECT AS ZIP
========================================================= */

app.get(
  '/api/projects/:id/export.zip',
  async (req, res) => {
    try {
      const p = (
        await readJson(PROJECTS_FILE, [])
      ).find(x => x.id === req.params.id);

      if (!p) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      const zipName =
        `${safeFilename(p.title || 'mora-project').replace(/\.[^.]+$/, '')}.zip`;

      res.setHeader(
        'Content-Type',
        'application/zip'
      );

      res.setHeader(
        'Content-Disposition',
        `attachment; filename="${zipName}"`
      );

      const zip = archiver('zip', {
        zlib: {
          level: 6
        }
      });

      zip.on('error', e => {
        console.error('ZIP error:', e);

        if (!res.headersSent) {
          res.status(500).end('ZIP export failed.');
        } else {
          res.end();
        }
      });

      zip.pipe(res);

      for (const [i, f] of normalizeFiles(p.files).entries()) {
        zip.append(f.content, {
          name: safeFilename(f.path || f.name, i)
        });
      }

      zip.append(
        JSON.stringify(
          {
            id: p.id,
            title: p.title,
            prompt: p.prompt,
            createdAt: p.createdAt,
            updatedAt: p.updatedAt
          },
          null,
          2
        ),
        {
          name: 'mora-project.json'
        }
      );

      await zip.finalize();
    } catch {
      if (!res.headersSent) {
        res.status(500).json({
          success: false,
          error: 'Could not export project.'
        });
      }
    }
  }
);

/* =========================================================
   DELETE PROJECT
========================================================= */

app.delete(
  '/api/projects/:id',
  async (req, res) => {
    try {
      const old = await readJson(
        PROJECTS_FILE,
        []
      );

      const next = old.filter(
        p => p.id !== req.params.id
      );

      if (old.length === next.length) {
        return res.status(404).json({
          success: false,
          error: 'Project not found.'
        });
      }

      await writeJson(
        PROJECTS_FILE,
        next
      );

      res.json({
        success: true,
        deleted: true
      });
    } catch {
      res.status(500).json({
        success: false,
        error: 'Could not delete project.'
      });
    }
  }
);

/* =========================================================
   GLOBAL ERROR HANDLER
========================================================= */

app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err);

  if (res.headersSent) return;

  res.status(err.status || 500).json({
    success: false,
    error: err.status === 413
      ? 'Request too large.'
      : 'Unexpected server error. Check server logs.'
  });
});

/* =========================================================
   START SERVER
========================================================= */

await fs.mkdir(DATA_DIR, {
  recursive: true
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`\nMora AI v3 listening on port ${PORT}`);

  const providers = configuredProviders();

  console.log(
    `AI provider order: ${
      providers.length
        ? providers
            .map(p => `${p.name} (${p.model})`)
            .join(' -> ')
        : 'NONE CONFIGURED'
    }`
  );

  console.log(
    `E2B: ${
      process.env.E2B_API_KEY
        ? 'configured'
        : 'not configured (Python execution disabled)'
    }`
  );

  console.log('Browser testing: optional Playwright');

  console.log(
    `Auth: ${
      TOKEN
        ? 'API token required'
        : 'disabled (set MORA_API_TOKEN for production)'
    }`
  );

  console.log(`Storage: ${DATA_DIR}\n`);
});
