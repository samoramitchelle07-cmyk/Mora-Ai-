import 'dotenv/config';

import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { Sandbox } from '@e2b/code-interpreter';
import archiver from 'archiver';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

// ============================================================
// MORA AI — CONFIGURATION
// ============================================================

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);

const IS_PRODUCTION = process.env.NODE_ENV === 'production';
const TOKEN = (process.env.MORA_API_TOKEN || '').trim();

const MODELS = {
    gemini: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    groq: process.env.GROQ_MODEL || 'openai/gpt-oss-120b',
    openrouter: process.env.OPENROUTER_MODEL || 'openai/gpt-oss-120b'
};

const AI_PROVIDER_ORDER = (
    process.env.AI_PROVIDER_ORDER || 'gemini,groq,openrouter'
)
    .split(',')
    .map(value => value.trim().toLowerCase())
    .filter((value, index, array) =>
        ['gemini', 'groq', 'openrouter'].includes(value) &&
        array.indexOf(value) === index
    );

const AI_OUTPUT_TOKENS = Math.max(
    1024,
    Math.min(
        32768,
        Number(process.env.AI_OUTPUT_TOKENS || 14000)
    )
);

const MAX_PROMPT = 12000;
const MAX_FILE_CHARS = 180000;
const MAX_FILES = 25;
const MAX_JSON_BODY = '1mb';

const DATA_DIR = path.join(ROOT, 'data');
const PROJECTS_FILE = path.join(DATA_DIR, 'projects.json');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');

if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) {
    throw new Error('PORT must be a valid port number.');
}

if (IS_PRODUCTION && !TOKEN) {
    console.error(
        'SECURITY ERROR: Set MORA_API_TOKEN before exposing Mora AI in production.'
    );

    process.exit(1);
}

if (!AI_PROVIDER_ORDER.length) {
    console.warn(
        '[Mora AI] No valid providers in AI_PROVIDER_ORDER.'
    );
}

// ============================================================
// EXPRESS APP AND SECURITY
// ============================================================

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
    .map(value => value.trim())
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

app.use(express.json({
    limit: MAX_JSON_BODY
}));

app.use(
    '/api/',
    rateLimit({
        windowMs: 60_000,
        limit: Math.max(
            1,
            Number(process.env.MAX_REQUESTS_PER_MINUTE || 20)
        ),
        standardHeaders: 'draft-7',
        legacyHeaders: false
    })
);

// ============================================================
// AUTHENTICATION
// ============================================================

function safeEqual(a, b) {
    const aa = Buffer.from(String(a));
    const bb = Buffer.from(String(b));

    return (
        aa.length === bb.length &&
        crypto.timingSafeEqual(aa, bb)
    );
}

function auth(req, res, next) {
    if (!TOKEN) {
        return next();
    }

    const provided =
        req.get('authorization')?.replace(/^Bearer\s+/i, '') ||
        req.get('x-mora-token') ||
        '';

    if (!safeEqual(provided, TOKEN)) {
        return res.status(401).json({
            success: false,
            error: 'Unauthorized. Check your Mora AI access token.'
        });
    }

    next();
}

app.use('/api/agent', auth);
app.use('/api/projects', auth);
app.use('/api/history', auth);
app.use('/api/test', auth);
app.use('/api/providers', auth);
app.use('/api/execute', auth);

// ============================================================
// FRONTEND
// ============================================================

app.get('/', (_req, res) => {
    res.sendFile(
        path.join(ROOT, 'public', 'index.html')
    );
});

app.use(
    '/assets',
    express.static(
        path.join(ROOT, 'public', 'assets'),
        {
            dotfiles: 'deny',
            index: false,
            fallthrough: false
        }
    )
);

// ============================================================
// JSON STORAGE
// ============================================================

async function readJson(file, fallback = []) {
    try {
        return JSON.parse(
            await fs.readFile(file, 'utf8')
        );
    } catch (error) {
        if (error.code === 'ENOENT') {
            return fallback;
        }

        throw error;
    }
}

async function writeJson(file, value) {
    await fs.mkdir(DATA_DIR, {
        recursive: true
    });

    const temporary =
        `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;

    await fs.writeFile(
        temporary,
        JSON.stringify(value, null, 2),
        {
            encoding: 'utf8',
            mode: 0o600
        }
    );

    await fs.rename(temporary, file);
}

async function ensureDataDirectory() {
    await fs.mkdir(DATA_DIR, {
        recursive: true
    });
}

async function saveHistory(entry) {
    const history = await readJson(
        HISTORY_FILE,
        []
    );

    history.unshift({
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        ...entry
    });

    await writeJson(
        HISTORY_FILE,
        history.slice(0, 500)
    );
}

// ============================================================
// FILE HELPERS
// ============================================================

function safeFilename(input, index = 0) {
    const raw = String(
        input || `file-${index + 1}.txt`
    )
        .replace(/\\/g, '/')
        .split('/')
        .pop();

    const safe = raw
        .replace(/[^a-zA-Z0-9._-]/g, '_')
        .slice(0, 100);

    return safe && safe !== '.' && safe !== '..'
        ? safe
        : `file-${index + 1}.txt`;
}

function normalizeFiles(files) {
    if (!Array.isArray(files)) {
        return [];
    }

    return files
        .slice(0, MAX_FILES)
        .map((file, index) => ({
            name: safeFilename(
                file?.name || file?.path,
                index
            ),

            path: safeFilename(
                file?.path || file?.name,
                index
            ),

            language: String(
                file?.language || ''
            ).slice(0, 40),

            content: String(
                file?.content ?? ''
            ).slice(0, MAX_FILE_CHARS)
        }));
}

// ============================================================
// ERROR HANDLING
// ============================================================

function providerError(provider, status, message) {
    const cleanMessage = String(
        message || 'Request failed'
    ).slice(0, 700);

    const error = new Error(
        `${provider}: ${cleanMessage}`
    );

    error.provider = provider;
    error.status = Number(status) || 502;
    error.providerMessage = cleanMessage;

    return error;
}

function getErrorStatus(error) {
    const status = Number(
        error?.status ||
        error?.statusCode ||
        error?.response?.status ||
        error?.cause?.status ||
        error?.cause?.statusCode
    );

    if (
        Number.isFinite(status) &&
        status >= 400 &&
        status <= 599
    ) {
        return status;
    }

    const message = String(
        error?.message || error || ''
    );

    const match = message.match(
        /\b(?:HTTP\s*)?(400|401|402|403|404|408|413|429|500|502|503|504)\b/i
    );

    if (match) {
        return Number(match[1]);
    }

    if (
        /quota|rate.?limit|resource_exhausted|too many requests/i
            .test(message)
    ) {
        return 429;
    }

    if (
        /invalid api key|unauthorized|authentication failed/i
            .test(message)
    ) {
        return 401;
    }

    return 502;
}

function friendlyError(error) {
    const message = String(
        error?.message || error || 'Unknown error'
    );

    const lower = message.toLowerCase();
    const status = getErrorStatus(error);

    if (
        status === 401 ||
        /invalid api key|incorrect api key|authentication failed|unauthorized/.test(lower)
    ) {
        return 'The provider rejected the API key. Verify that the key belongs to the selected provider.';
    }

    if (
        status === 402 ||
        /insufficient credits|payment required|billing issue/.test(lower)
    ) {
        return 'The provider requires credits or billing before this request can run. Check the provider account.';
    }

    if (
        status === 403 ||
        /permission denied|access denied|not authorized/.test(lower)
    ) {
        return 'The provider denied access. Check account permissions, model access, and API-key restrictions.';
    }

    if (
        status === 404 ||
        /model.*(?:not found|does not exist|unavailable)/.test(lower)
    ) {
        return `The selected model may be unavailable or incorrectly named. Details: ${message.slice(0, 300)}`;
    }

    if (
        status === 429 ||
        /quota|rate limit|too many requests/.test(lower)
    ) {
        return 'The provider rate limit or quota was reached. Check usage and billing, or use another provider.';
    }

    if (
        /timeout|timed out|abort/.test(lower)
    ) {
        return 'The provider took too long to respond. Please try again.';
    }

    if (
        /e2b|sandbox/.test(lower)
    ) {
        return `The execution sandbox could not complete the operation: ${message.slice(0, 300)}`;
    }

    if (
        /fetch failed|network|enotfound|econnrefused/.test(lower)
    ) {
        return 'Mora AI could not connect to the requested service. Check your network and provider configuration.';
    }

    return message.length > 500
        ? `${message.slice(0, 500)}…`
        : message;
}

function errorStatus(error) {
    const status = Number(
        error?.status ||
        error?.statusCode ||
        0
    );

    if (
        Number.isInteger(status) &&
        status >= 400 &&
        status <= 599
    ) {
        return status;
    }

    return 500;
}

// ============================================================
// AI RESPONSE PARSING
// ============================================================

function parseJson(text) {
    if (
        typeof text !== 'string' ||
        !text.trim()
    ) {
        throw new Error(
            'The AI provider returned an empty response.'
        );
    }

    const cleaned = text
        .trim()
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/, '');

    try {
        return JSON.parse(cleaned);
    } catch {
        // Try extracting a JSON object below.
    }

    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');

    if (start >= 0 && end > start) {
        try {
            return JSON.parse(
                cleaned.slice(start, end + 1)
            );
        } catch {
            // Report a useful error below.
        }
    }

    throw new Error(
        'Could not parse the AI response as JSON. Try again or use another provider.'
    );
}

// ============================================================
// NORMALIZE AI RESULTS
// IMPORTANT: prompt now has a safe default.
// This fixes the undefined .slice() error.
// ============================================================

function normalizeResult(raw, prompt = '') {
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

                html: raw.canvas.html.slice(
                    0,
                    MAX_FILE_CHARS
                )
            }
            : null;

    const execution =
        raw?.execution &&
        typeof raw.execution === 'object'
            ? {
                requested: Boolean(
                    raw.execution.requested
                ),

                language: String(
                    raw.execution.language || 'python'
                ).toLowerCase(),

                code: String(
                    raw.execution.code || ''
                ).slice(0, 12000)
            }
            : {
                requested: false,
                language: 'python',
                code: ''
            };

    return {
        success: true,

        message: String(
            raw?.message || 'Finished.'
        ).slice(0, 12000),

        steps: Array.isArray(raw?.steps)
            ? raw.steps.slice(0, 15).map(
                (step, index) => ({
                    thought: String(
                        step?.thought || `Step ${index + 1}`
                    ).slice(0, 300),

                    command: String(
                        step?.command || 'process'
                    ).slice(0, 100),

                    output: String(
                        step?.output || ''
                    ).slice(0, 2500),

                    isError: Boolean(step?.isError)
                })
            )
            : [],

        files,
        canvas,
        execution,

        terminal: String(
            raw?.terminal || ''
        ).slice(0, 12000),

        status: 'completed',

        prompt: String(
            prompt || raw?.prompt || ''
        ).slice(0, MAX_PROMPT),

        provider: raw?.provider || null,
        model: raw?.model || null
    };
}

// ============================================================
// AI SYSTEM INSTRUCTIONS
// ============================================================

const SYSTEM_PROMPT = `You are Mora AI, a practical AI assistant for coding, debugging, website development, research, learning, and general assistance.

Return ONLY one valid JSON object in this structure:
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
- Never reveal private chain-of-thought.
- Ordinary questions use message, empty files, canvas null, and execution.requested false.
- Website requests should include a complete standalone HTML document in canvas.html and normally a matching index.html in files.
- Code requests should return complete files rather than incomplete snippets.
- Never claim code was tested unless it actually ran.
- Request Python execution only when the user asks to run or test Python or needs a Python calculation.
- Never execute shell commands, JavaScript, destructive code, credential theft, unauthorized access, or code that evades security.
- Keep each file under 180,000 characters.
- Keep responses practical and concise.
- Return valid JSON, without Markdown fences around the JSON object.`;

// ============================================================
// PROVIDER CONFIGURATION
// ============================================================

const PROVIDER_CONFIG = {
    gemini: {
        id: 'gemini',
        name: 'Gemini',

        endpoint:
            'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',

        defaultModel: MODELS.gemini,
        envKey: 'GEMINI_API_KEY'
    },

    groq: {
        id: 'groq',
        name: 'Groq',

        endpoint:
            'https://api.groq.com/openai/v1/chat/completions',

        defaultModel: MODELS.groq,
        envKey: 'GROQ_API_KEY'
    },

    openrouter: {
        id: 'openrouter',
        name: 'OpenRouter',

        endpoint:
            'https://openrouter.ai/api/v1/chat/completions',

        defaultModel: MODELS.openrouter,
        envKey: 'OPENROUTER_API_KEY'
    }
};

function resolveProviderConfig(providerId, request = {}) {
    const id = String(
        providerId || ''
    ).trim().toLowerCase();

    const base = PROVIDER_CONFIG[id];

    if (!base) {
        throw providerError(
            'Mora AI',
            400,
            'Unsupported provider. Choose Gemini, Groq, or OpenRouter.'
        );
    }

    const personalKey =
        typeof request.apiKey === 'string'
            ? request.apiKey.trim()
            : '';

    const environmentKey = process.env[base.envKey] || '';

    const apiKey = personalKey || environmentKey;

    const requestedModel =
        typeof request.model === 'string'
            ? request.model.trim()
            : '';

    const model = requestedModel || base.defaultModel;

    if (!apiKey) {
        throw providerError(
            base.name,
            401,
            `No API key is configured for ${base.name}. Add a personal key or configure the administrator key.`
        );
    }

    if (apiKey.length > 500) {
        throw providerError(
            base.name,
            400,
            'The API key is too long.'
        );
    }

    if (
        !model ||
        model.length > 200 ||
        /[\r\n]/.test(model)
    ) {
        throw providerError(
            base.name,
            400,
            'Invalid model ID.'
        );
    }

    return {
        id,
        name: base.name,
        endpoint: base.endpoint,
        apiKey,
        model,
        personalKeyUsed: Boolean(personalKey)
    };
}

function publicProviderList() {
    return Object.values(PROVIDER_CONFIG).map(config => ({
        id: config.id,
        name: config.name,
        defaultModel: config.defaultModel,

        administratorConfigured: Boolean(
            process.env[config.envKey]
        ),

        personalKeySupported: true
    }));
}

function configuredProviders() {
    const seen = new Set();

    return AI_PROVIDER_ORDER
        .filter(id => {
            const config = PROVIDER_CONFIG[id];

            if (
                !config ||
                !process.env[config.envKey] ||
                seen.has(id)
            ) {
                return false;
            }

            seen.add(id);
            return true;
        })
        .map(id => ({
            id,
            name: PROVIDER_CONFIG[id].name,
            configured: true,
            model: PROVIDER_CONFIG[id].defaultModel
        }));
}

app.get('/api/providers', (_req, res) => {
    res.json({
        success: true,
        providers: publicProviderList(),
        fallbackOrder: AI_PROVIDER_ORDER
    });
});

// ============================================================
// PROVIDER REQUEST
// FIXED: Defensive response parsing and clearer errors.
// ============================================================

async function requestProviderText(
    providerConfig,
    prompt,
    repairContext = ''
) {
    const userContent = repairContext
        ? `Original request:\n${prompt}\n\nRepair context:\n${repairContext}\n\nReturn the complete corrected JSON object.`
        : prompt;

    const headers = {
        Authorization: `Bearer ${providerConfig.apiKey}`,
        'Content-Type': 'application/json'
    };

    if (providerConfig.id === 'openrouter') {
        headers['X-Title'] = 'Mora AI';

        if (process.env.OPENROUTER_SITE_URL) {
            headers['HTTP-Referer'] =
                process.env.OPENROUTER_SITE_URL;
        }
    }

    let response;

    try {
        response = await fetch(
            providerConfig.endpoint,
            {
                method: 'POST',
                headers,

                signal: AbortSignal.timeout(120000),

                body: JSON.stringify({
                    model: providerConfig.model,

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
            }
        );
    } catch (error) {
        throw providerError(
            providerConfig.name,
            502,
            error.name === 'TimeoutError' ||
            error.name === 'AbortError'
                ? 'The provider request timed out.'
                : 'Unable to connect to the provider.'
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
            body?.error?.metadata?.raw ||
            body?.message ||
            `The provider returned HTTP ${response.status}.`;

        throw providerError(
            providerConfig.name,
            response.status,
            String(message).slice(0, 700)
        );
    }

    if (!body || typeof body !== 'object') {
        throw providerError(
            providerConfig.name,
            502,
            'The provider returned an invalid JSON response.'
        );
    }

    const choice = body?.choices?.[0];
    const content = choice?.message?.content;

    let text = '';

    if (typeof content === 'string') {
        text = content;
    } else if (Array.isArray(content)) {
        text = content
            .map(part => {
                if (typeof part === 'string') {
                    return part;
                }

                if (typeof part?.text === 'string') {
                    return part.text;
                }

                return '';
            })
            .join('');
    }

    if (!text.trim()) {
        throw providerError(
            providerConfig.name,
            502,
            `The model returned no usable text. Finish reason: ${String(choice?.finish_reason || 'unknown')}. Check model compatibility and output-token limits.`
        );
    }

    return {
        text,
        model: providerConfig.model,
        provider: providerConfig.id
    };
}

// ============================================================
// PROVIDER REQUEST WRAPPERS
// ============================================================

async function requestGeminiText(
    prompt,
    repairContext = '',
    request = {}
) {
    return requestProviderText(
        resolveProviderConfig('gemini', request),
        prompt,
        repairContext
    );
}

async function requestGroqText(
    prompt,
    repairContext = '',
    request = {}
) {
    return requestProviderText(
        resolveProviderConfig('groq', request),
        prompt,
        repairContext
    );
}

async function requestOpenRouterText(
    prompt,
    repairContext = '',
    request = {}
) {
    return requestProviderText(
        resolveProviderConfig('openrouter', request),
        prompt,
        repairContext
    );
}

// ============================================================
// PROVIDER FALLBACK
// ============================================================

function shouldFallback(error) {
    const status = getErrorStatus(error);

    if (
        [
            400,
            401,
            402,
            403,
            404,
            408,
            413,
            429,
            500,
            502,
            503,
            504
        ].includes(status)
    ) {
        return true;
    }

    const message = String(
        error?.message || ''
    ).toLowerCase();

    return /quota|rate.?limit|resource_exhausted|timeout|network|empty response|no usable text|could not parse|invalid json|insufficient credits|payment required/.test(
        message
    );
}

function getRequestedProviders(request = {}) {
    const selected = String(
        request.provider || ''
    ).trim().toLowerCase();

    if (
        selected &&
        !Object.hasOwn(PROVIDER_CONFIG, selected)
    ) {
        throw providerError(
            'Mora AI',
            400,
            'Invalid provider selection.'
        );
    }

    const hasPersonalKey =
        typeof request.apiKey === 'string' &&
        Boolean(request.apiKey.trim());

    // Personal keys apply only to the explicitly selected provider.
    if (selected && hasPersonalKey) {
        return [selected];
    }

    if (selected) {
        return [
            selected,
            ...AI_PROVIDER_ORDER.filter(
                id => id !== selected
            )
        ];
    }

    return AI_PROVIDER_ORDER;
}

function providerHasKey(providerId, request = {}) {
    if (
        providerId === request.provider &&
        typeof request.apiKey === 'string' &&
        request.apiKey.trim()
    ) {
        return true;
    }

    const config = PROVIDER_CONFIG[providerId];

    return Boolean(
        config && process.env[config.envKey]
    );
}

// ============================================================
// MAIN AI ROUTER
// ============================================================

async function askGemini(
    prompt,
    repairContext = '',
    onChunk = null,
    request = {}
) {
    const providerIds = getRequestedProviders(request);

    const available = providerIds.filter(
        id => providerHasKey(id, request)
    );

    if (!available.length) {
        throw providerError(
            'Mora AI',
            503,
            'No API key is available for the selected provider or configured fallback providers.'
        );
    }

    const failures = [];

    for (const providerId of available) {
        let config;

        try {
            const providerRequest =
                providerId === request.provider
                    ? request
                    : {};

            config = resolveProviderConfig(
                providerId,
                providerRequest
            );

            console.log(
                `[Mora AI] Trying ${config.name} (${config.model})`
            );

            let generated;

            if (providerId === 'gemini') {
                generated = await requestGeminiText(
                    prompt,
                    repairContext,
                    providerRequest
                );
            } else if (providerId === 'groq') {
                generated = await requestGroqText(
                    prompt,
                    repairContext,
                    providerRequest
                );
            } else {
                generated = await requestOpenRouterText(
                    prompt,
                    repairContext,
                    providerRequest
                );
            }

            const parsed = parseJson(
                generated.text
            );

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

            const result = normalizeResult(
                parsed,
                prompt
            );

            result.provider = generated.provider;
            result.model = generated.model;

            console.log(
                `[Mora AI] Success: ${config.name}`
            );

            if (typeof onChunk === 'function') {
                onChunk({
                    partialMessage: result.message,
                    provider: generated.provider,
                    model: generated.model
                });
            }

            return result;
        } catch (error) {
            const status = getErrorStatus(error);

            const failure = {
                provider:
                    PROVIDER_CONFIG[providerId]?.name ||
                    providerId,

                model: config?.model || '',

                status,

                message: String(
                    error.providerMessage ||
                    error.message ||
                    'Request failed'
                ).slice(0, 700)
            };

            failures.push(failure);

            console.error(
                `[Mora AI] ${failure.provider} failed (${status}): ${failure.message}`
            );

            if (!shouldFallback(error)) {
                break;
            }
        }
    }

    const summary = failures
        .map(item =>
            `${item.provider}: ${item.message}`
        )
        .join(' | ');

    const finalError = new Error(
        `All available AI providers failed. ${summary}`
    );

    finalError.failures = failures;

    finalError.status =
        failures.length &&
        failures.every(item => item.status === 429)
            ? 429
            : failures.length &&
              failures.every(item =>
                  item.status === 401 ||
                  item.status === 403
              )
                ? 401
                : 502;

    throw finalError;
}

// ============================================================
// PROVIDER CONNECTION TEST
// ============================================================

app.post('/api/providers/test', async (req, res) => {
    const provider = String(
        req.body?.provider || ''
    ).trim().toLowerCase();

    try {
        if (!Object.hasOwn(PROVIDER_CONFIG, provider)) {
            return res.status(400).json({
                success: false,
                error: 'Choose Gemini, Groq, or OpenRouter.'
            });
        }

        const config = resolveProviderConfig(
            provider,
            {
                apiKey: req.body?.apiKey,
                model: req.body?.model
            }
        );

        const testPrompt =
            'Return only this JSON object, without Markdown: {"message":"Connection successful","steps":[],"files":[],"canvas":null,"execution":{"requested":false,"language":"python","code":""},"terminal":""}';

        const result = await requestProviderText(
            config,
            testPrompt
        );

        const parsed = parseJson(
            result.text
        );

        if (
            !parsed ||
            typeof parsed.message !== 'string'
        ) {
            throw providerError(
                config.name,
                502,
                'The provider responded, but its response did not match Mora AI’s expected format.'
            );
        }

        res.json({
            success: true,
            provider,
            model: result.model,
            message: 'The provider accepted the request.'
        });
    } catch (error) {
        const status = getErrorStatus(error);

        res.status(
            status >= 400 && status <= 599
                ? status
                : 502
        ).json({
            success: false,
            provider,
            error: String(
                error.providerMessage ||
                error.message ||
                'Connection test failed.'
            ).slice(0, 700)
        });
    }
});

// ============================================================
// PROMPT VALIDATION
// ============================================================

function validatePrompt(prompt) {
    if (typeof prompt !== 'string') {
        throw Object.assign(
            new Error(
                'Please enter a message for Mora AI.'
            ),
            { status: 400 }
        );
    }

    const cleaned = prompt.trim();

    if (!cleaned) {
        throw Object.assign(
            new Error('Your message cannot be empty.'),
            { status: 400 }
        );
    }

    if (cleaned.length > MAX_PROMPT) {
        throw Object.assign(
            new Error(
                `Your message is too long. Maximum length: ${MAX_PROMPT} characters.`
            ),
            { status: 413 }
        );
    }

    return cleaned;
}

// ============================================================
// PYTHON SANDBOX
// ============================================================

function unsafePythonReason(code) {
    if (
        typeof code !== 'string' ||
        !code.trim()
    ) {
        return 'Python code is empty.';
    }

    if (code.length > MAX_FILE_CHARS) {
        return `Python code exceeds the ${MAX_FILE_CHARS}-character limit.`;
    }

    const blockedPatterns = [
        {
            pattern: /\bsubprocess\b/,
            reason: 'Subprocess execution is disabled.'
        },
        {
            pattern: /\b(?:os\.system|os\.popen|os\.exec[a-z_]*)\s*\(/,
            reason: 'Operating-system command execution is disabled.'
        },
        {
            pattern: /\b(?:eval|exec)\s*\(/,
            reason: 'Dynamic Python code execution is disabled.'
        },
        {
            pattern: /\b(?:getpass|input)\s*\(/,
            reason: 'Interactive input is not supported in the sandbox.'
        }
    ];

    for (const item of blockedPatterns) {
        if (item.pattern.test(code)) {
            return item.reason;
        }
    }

    return null;
}

async function executePython(code) {
    const safetyIssue = unsafePythonReason(code);

    if (safetyIssue) {
        throw Object.assign(
            new Error(safetyIssue),
            { status: 400 }
        );
    }

    if (!process.env.E2B_API_KEY) {
        throw Object.assign(
            new Error(
                'Python execution is unavailable because E2B_API_KEY is not configured.'
            ),
            { status: 503 }
        );
    }

    let sandbox;

    try {
        sandbox = await Sandbox.create({
            apiKey: process.env.E2B_API_KEY,
            timeoutMs: 120000
        });

        const result = await sandbox.runCode(
            code,
            {
                timeoutMs: 120000
            }
        );

        const logs = result.logs || {};

        return {
            success: !result.error,

            results: result.results || [],

            stdout: Array.isArray(logs.stdout)
                ? logs.stdout.join('\n')
                : String(logs.stdout || ''),

            stderr: Array.isArray(logs.stderr)
                ? logs.stderr.join('\n')
                : String(logs.stderr || ''),

            error: result.error
                ? String(
                    result.error.value ||
                    result.error.message ||
                    result.error
                )
                : null
        };
    } catch (error) {
        throw Object.assign(
            new Error(
                `Python sandbox failed: ${friendlyError(error)}`
            ),
            {
                status: errorStatus(error)
            }
        );
    } finally {
        if (
            sandbox &&
            typeof sandbox.close === 'function'
        ) {
            try {
                await sandbox.close();
            } catch {
                // Preserve the original result.
            }
        }
    }
}

// ============================================================
// AGENT WORKFLOW
// ============================================================

async function runAgent(input, options = {}) {
    const prompt = validatePrompt(input);

    const files = normalizeFiles(
        Array.isArray(options.files)
            ? options.files
            : []
    );

    const selectedProvider = options.provider
        ? String(options.provider).trim().toLowerCase()
        : '';

    const apiKey =
        typeof options.apiKey === 'string'
            ? options.apiKey.trim()
            : '';

    const model =
        typeof options.model === 'string'
            ? options.model.trim()
            : '';

    const requestOptions = {
        provider: selectedProvider || undefined,
        apiKey: apiKey || undefined,
        model: model || undefined
    };

    let fileContext = '';

    if (files.length) {
        fileContext = '\n\nUSER-PROVIDED FILES:\n';

        for (const file of files) {
            fileContext +=
                `\n--- FILE: ${file.name} ---\n` +
                `${file.content}\n`;
        }
    }

    const fullPrompt = `${prompt}${fileContext}`;

    if (fullPrompt.length > MAX_FILE_CHARS * 2) {
        throw Object.assign(
            new Error(
                'The combined message and file contents are too large.'
            ),
            { status: 413 }
        );
    }

    // One provider-routing call handles fallback internally.
    // Avoid repeating the entire request when a provider fails.
    const result = await askGemini(
        fullPrompt,
        '',
        null,
        requestOptions
    );

    // Preserve the provider/model from the successful response.
    const finalResult = normalizeResult(
        result,
        prompt
    );

    finalResult.provider = result.provider;
    finalResult.model = result.model;

    await saveHistory({
        prompt,
        response: finalResult.message || '',
        provider: finalResult.provider || selectedProvider || null,
        model: finalResult.model || model || null,
        files: files.map(file => file.name)
    });

    return finalResult;
}

// ============================================================
// HEALTH ENDPOINT
// ============================================================

app.get('/api/health', (_req, res) => {
    res.json({
        success: true,
        service: 'Mora AI',
        status: 'online',
        timestamp: new Date().toISOString(),

        providers: configuredProviders().map(
            provider => ({
                id: provider.id,
                name: provider.name,
                configured: provider.configured
            })
        ),

        pythonSandboxConfigured: Boolean(
            process.env.E2B_API_KEY
        )
    });
});

// ============================================================
// SERVER-SENT EVENTS
// ============================================================

function sendSse(res, event, payload) {
    res.write(`event: ${event}\n`);
    res.write(
        `data: ${JSON.stringify(payload)}\n\n`
    );
}

app.post('/api/agent/stream', async (req, res) => {
    res.status(200);

    res.setHeader(
        'Content-Type',
        'text/event-stream; charset=utf-8'
    );

    res.setHeader(
        'Cache-Control',
        'no-cache, no-transform'
    );

    res.setHeader(
        'Connection',
        'keep-alive'
    );

    res.setHeader(
        'X-Accel-Buffering',
        'no'
    );

    if (typeof res.flushHeaders === 'function') {
        res.flushHeaders();
    }

    let disconnected = false;

    // Use the response close event to detect client disconnection.
    res.on('close', () => {
        if (!res.writableEnded) {
            disconnected = true;
        }
    });

    const progress = message => {
        if (
            !disconnected &&
            !res.writableEnded
        ) {
            sendSse(
                res,
                'progress',
                { message }
            );
        }
    };

    try {
        progress('Preparing your request…');

        const prompt = validatePrompt(
            req.body?.prompt
        );

        const options = {
            provider: req.body?.provider,
            model: req.body?.model,
            apiKey: req.body?.apiKey,
            files: req.body?.files
        };

        progress(
            'Contacting the selected AI provider…'
        );

        const result = await runAgent(
            prompt,
            options
        );

        if (
            disconnected ||
            res.writableEnded
        ) {
            return;
        }

        progress('Preparing the response…');

        // This is SSE delivery, not token-by-token model streaming.
        sendSse(
            res,
            'delta',
            {
                text: result.message || ''
            }
        );

        sendSse(
            res,
            'result',
            result
        );

        sendSse(
            res,
            'done',
            { success: true }
        );
    } catch (error) {
        if (
            !disconnected &&
            !res.writableEnded
        ) {
            sendSse(
                res,
                'error',
                {
                    message: friendlyError(error),
                    status: errorStatus(error)
                }
            );
        }
    } finally {
        if (!res.writableEnded) {
            res.end();
        }
    }
});

// ============================================================
// NON-STREAMING AGENT
// ============================================================

app.post('/api/agent', async (req, res, next) => {
    try {
        const prompt = validatePrompt(
            req.body?.prompt
        );

        const result = await runAgent(
            prompt,
            {
                provider: req.body?.provider,
                model: req.body?.model,
                apiKey: req.body?.apiKey,
                files: req.body?.files
            }
        );

        res.json({
            success: true,
            result
        });
    } catch (error) {
        next(error);
    }
});

// ============================================================
// PYTHON EXECUTION ENDPOINT
// ============================================================

app.post(
    '/api/execute/python',
    async (req, res, next) => {
        try {
            const code = req.body?.code;

            if (typeof code !== 'string') {
                return res.status(400).json({
                    success: false,
                    error: "Provide Python code in the 'code' field."
                });
            }

            const result = await executePython(code);

            res.json({
                success: true,
                result
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// STATIC HTML TEST
// ============================================================

app.post(
    '/api/test/html',
    async (req, res, next) => {
        try {
            const html = req.body?.html;

            if (
                typeof html !== 'string' ||
                !html.trim()
            ) {
                return res.status(400).json({
                    success: false,
                    error: "Provide HTML in the 'html' field."
                });
            }

            if (html.length > MAX_FILE_CHARS) {
                return res.status(413).json({
                    success: false,
                    error: `HTML exceeds the ${MAX_FILE_CHARS}-character limit.`
                });
            }

            const checks = {
                hasDoctype: /<!doctype\s+html/i.test(html),

                hasHtmlTag: /<html(?:\s|>)/i.test(html),

                hasHeadTag: /<head(?:\s|>)/i.test(html),

                hasBodyTag: /<body(?:\s|>)/i.test(html),

                hasClosingHtmlTag: /<\/html\s*>/i.test(html),

                hasClosingBodyTag: /<\/body\s*>/i.test(html),

                hasScriptTag: /<script(?:\s|>)/i.test(html),

                hasStyleTag: /<style(?:\s|>)/i.test(html)
            };

            const warnings = [];

            if (!checks.hasDoctype) {
                warnings.push(
                    'Missing <!DOCTYPE html> declaration.'
                );
            }

            if (
                !checks.hasHtmlTag ||
                !checks.hasClosingHtmlTag
            ) {
                warnings.push(
                    'The HTML document may have an incomplete html element.'
                );
            }

            if (
                !checks.hasBodyTag ||
                !checks.hasClosingBodyTag
            ) {
                warnings.push(
                    'The HTML document may have an incomplete body element.'
                );
            }

            if (
                /<script[^>]*src=["']https?:\/\//i.test(html)
            ) {
                warnings.push(
                    'External scripts were detected. Review them before opening the page.'
                );
            }

            res.json({
                success: true,
                testType: 'static-html-check',

                message:
                    'Static HTML checks completed. This is not a full browser test.',

                checks,
                warnings
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// HISTORY ENDPOINTS
// ============================================================

app.get(
    '/api/history',
    async (_req, res, next) => {
        try {
            const history = await readJson(
                HISTORY_FILE,
                []
            );

            res.json({
                success: true,
                history
            });
        } catch (error) {
            next(error);
        }
    }
);

app.delete(
    '/api/history',
    async (_req, res, next) => {
        try {
            await writeJson(
                HISTORY_FILE,
                []
            );

            res.json({
                success: true,
                message: 'History cleared.'
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// PROJECT VALIDATION
// ============================================================

function validateProjectData(body) {
    const name = typeof body?.name === 'string'
        ? body.name.trim()
        : '';

    if (!name) {
        throw Object.assign(
            new Error('Project name is required.'),
            { status: 400 }
        );
    }

    if (name.length > 160) {
        throw Object.assign(
            new Error('Project name is too long.'),
            { status: 400 }
        );
    }

    const files = Array.isArray(body.files)
        ? body.files
        : [];

    if (files.length > MAX_FILES) {
        throw Object.assign(
            new Error(
                `A project can contain at most ${MAX_FILES} files.`
            ),
            { status: 400 }
        );
    }

    const normalizedFiles = files.map(
        (file, index) => {
            const fileName = String(
                file?.name || `file-${index + 1}.txt`
            );

            const content = String(
                file?.content ?? ''
            );

            if (
                fileName.includes('..') ||
                fileName.includes('/') ||
                fileName.includes('\\') ||
                fileName.includes('\0')
            ) {
                throw Object.assign(
                    new Error(
                        `Invalid file name: ${fileName}`
                    ),
                    { status: 400 }
                );
            }

            if (content.length > MAX_FILE_CHARS) {
                throw Object.assign(
                    new Error(
                        `File "${fileName}" is too large.`
                    ),
                    { status: 413 }
                );
            }

            return {
                name: fileName,
                content
            };
        }
    );

    return {
        name,

        description: String(
            body.description || ''
        ).slice(0, 2000),

        files: normalizedFiles,

        updatedAt: new Date().toISOString()
    };
}

// ============================================================
// LIST PROJECTS
// ============================================================

app.get(
    '/api/projects',
    async (_req, res, next) => {
        try {
            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            res.json({
                success: true,
                projects
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// CREATE PROJECT
// ============================================================

app.post(
    '/api/projects',
    async (req, res, next) => {
        try {
            const data = validateProjectData(
                req.body
            );

            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            const project = {
                id: crypto.randomUUID(),

                createdAt: new Date().toISOString(),

                ...data
            };

            projects.unshift(project);

            await writeJson(
                PROJECTS_FILE,
                projects.slice(0, 200)
            );

            res.status(201).json({
                success: true,
                project
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// GET PROJECT
// ============================================================

app.get(
    '/api/projects/:id',
    async (req, res, next) => {
        try {
            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            const project = projects.find(
                item => item.id === req.params.id
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
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// UPDATE PROJECT
// ============================================================

app.put(
    '/api/projects/:id',
    async (req, res, next) => {
        try {
            const data = validateProjectData(
                req.body
            );

            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            const index = projects.findIndex(
                item => item.id === req.params.id
            );

            if (index === -1) {
                return res.status(404).json({
                    success: false,
                    error: 'Project not found.'
                });
            }

            projects[index] = {
                ...projects[index],
                ...data,

                id: projects[index].id,

                createdAt: projects[index].createdAt
            };

            await writeJson(
                PROJECTS_FILE,
                projects
            );

            res.json({
                success: true,
                project: projects[index]
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// DELETE PROJECT
// ============================================================

app.delete(
    '/api/projects/:id',
    async (req, res, next) => {
        try {
            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            const updated = projects.filter(
                item => item.id !== req.params.id
            );

            if (
                updated.length === projects.length
            ) {
                return res.status(404).json({
                    success: false,
                    error: 'Project not found.'
                });
            }

            await writeJson(
                PROJECTS_FILE,
                updated
            );

            res.json({
                success: true,
                message: 'Project deleted.'
            });
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// PROJECT ZIP EXPORT
// ============================================================

app.get(
    '/api/projects/:id/export.zip',
    async (req, res, next) => {
        try {
            const projects = await readJson(
                PROJECTS_FILE,
                []
            );

            const project = projects.find(
                item => item.id === req.params.id
            );

            if (!project) {
                return res.status(404).json({
                    success: false,
                    error: 'Project not found.'
                });
            }

            res.setHeader(
                'Content-Type',
                'application/zip'
            );

            res.setHeader(
                'Content-Disposition',
                `attachment; filename="mora-project-${project.id}.zip"`
            );

            const archive = archiver(
                'zip',
                {
                    zlib: { level: 9 }
                }
            );

            archive.on('error', error => {
                if (!res.headersSent) {
                    next(error);
                } else {
                    res.destroy(error);
                }
            });

            archive.pipe(res);

            const manifest = {
                id: project.id,
                name: project.name,
                description: project.description || '',
                createdAt: project.createdAt,
                updatedAt: project.updatedAt,
                exportedAt: new Date().toISOString()
            };

            archive.append(
                JSON.stringify(
                    manifest,
                    null,
                    2
                ),
                {
                    name: 'mora-project.json'
                }
            );

            for (const file of project.files || []) {
                const safeName = String(
                    file.name || 'file.txt'
                )
                    .replace(/[<>:"|?*\x00-\x1F]/g, '_')
                    .replace(/^\.{1,2}$/, 'file.txt');

                archive.append(
                    String(file.content ?? ''),
                    {
                        name: `files/${safeName}`
                    }
                );
            }

            await archive.finalize();
        } catch (error) {
            next(error);
        }
    }
);

// ============================================================
// 404 HANDLER
// ============================================================

app.use((_req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found.'
    });
});

// ============================================================
// GLOBAL ERROR HANDLER
// ============================================================

app.use((error, req, res, next) => {
    if (res.headersSent) {
        return next(error);
    }

    const status = errorStatus(error);

    if (status >= 500) {
        console.error(
            '[Mora AI Error]',
            error.message
        );
    }

    res.status(status).json({
        success: false,
        error: friendlyError(error)
    });
});

// ============================================================
// SERVER STARTUP
// ============================================================

async function startServer() {
    await ensureDataDirectory();

    await Promise.all([
        fs.access(PROJECTS_FILE).catch(
            async () => {
                await writeJson(
                    PROJECTS_FILE,
                    []
                );
            }
        ),

        fs.access(HISTORY_FILE).catch(
            async () => {
                await writeJson(
                    HISTORY_FILE,
                    []
                );
            }
        )
    ]);

    app.listen(
        PORT,
        '0.0.0.0',
        () => {
            console.log('');
            console.log('======================================');
            console.log('          MORA AI SERVER');
            console.log('======================================');
            console.log('Status:  Online');
            console.log(`Port:    ${PORT}`);

            console.log(
                `Mode:    ${IS_PRODUCTION ? 'production' : 'development'}`
            );

            console.log(
                `URL:     http://localhost:${PORT}`
            );

            console.log('--------------------------------------');

            for (const id of AI_PROVIDER_ORDER) {
                const config = PROVIDER_CONFIG[id];

                if (!config) continue;

                console.log(
                    `${config.name}: ${
                        process.env[config.envKey]
                            ? 'Configured'
                            : 'Not configured'
                    }`
                );
            }

            console.log(
                `E2B Python sandbox: ${
                    process.env.E2B_API_KEY
                        ? 'Configured'
                        : 'Not configured'
                }`
            );

            console.log('======================================');
            console.log('');
        }
    );
}

startServer().catch(error => {
    console.error(
        'Mora AI failed to start:',
        error
    );

    process.exit(1);
});
