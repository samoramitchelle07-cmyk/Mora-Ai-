import 'dotenv/config';

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import archiver from 'archiver';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

const PORT = Number(process.env.PORT || 4242);
const IS_PRODUCTION = process.env.NODE_ENV === 'production';

const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR = path.join(__dirname, 'data');
const GENERATED_DIR = path.join(PUBLIC_DIR, 'generated-images');

const PROJECTS_FILE = path.join(DATA_DIR, 'projects.json');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');
const CONVERSATIONS_FILE = path.join(DATA_DIR, 'conversations.json');
/* =========================================================
   CYBERLAB STORAGE
   ========================================================= */

const CYBER_REPORTS_FILE = path.join(
    DATA_DIR,
    'cyber-reports.json'
);

const API_TOKEN = process.env.MORA_API_TOKEN || '';

if (IS_PRODUCTION && !API_TOKEN) {
    console.error(
        'Startup refused: set MORA_API_TOKEN before running in production.'
    );

    process.exit(1);
}

await fs.mkdir(DATA_DIR, { recursive: true });
await fs.mkdir(PUBLIC_DIR, { recursive: true });
await fs.mkdir(GENERATED_DIR, { recursive: true });

/* =========================================================
   FILE STORAGE
   ========================================================= */

async function readJson(file, fallback) {
    try {
        return JSON.parse(await fs.readFile(file, 'utf8'));
    } catch (error) {
        if (
            error.code !== 'ENOENT' &&
            !(error instanceof SyntaxError)
        ) {
            console.error(
                `Unable to read ${path.basename(file)}:`,
                error.message
            );
        }

        return fallback;
    }
}

async function writeJson(file, data) {
    const temporaryFile =
        `${file}.${crypto.randomUUID()}.tmp`;

    await fs.writeFile(
        temporaryFile,
        JSON.stringify(data, null, 2),
        'utf8'
    );

    await fs.rename(temporaryFile, file);
}

/* =========================================================
   AUTHENTICATION AND SECURITY
   ========================================================= */

function safeEqual(a, b) {
    const left = Buffer.from(String(a || ''));
    const right = Buffer.from(String(b || ''));

    return (
        left.length === right.length &&
        crypto.timingSafeEqual(left, right)
    );
}

function auth(req, res, next) {
    if (!API_TOKEN) {
        return next();
    }

    const authorization =
        req.headers.authorization || '';

    const bearer = authorization.startsWith('Bearer ')
        ? authorization.slice(7)
        : '';

    const supplied =
        bearer || req.headers['x-mora-token'] || '';

    if (!safeEqual(supplied, API_TOKEN)) {
        return res.status(401).json({
            success: false,
            error: 'Unauthorized. Check your Mora API token in Settings.'
        });
    }

    next();
}

app.disable('x-powered-by');

app.use(helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false
}));

app.use(cors({
    origin: process.env.CORS_ORIGIN
        ? process.env.CORS_ORIGIN
            .split(',')
            .map(value => value.trim())
        : true,

    methods: [
        'GET',
        'POST',
        'PUT',
        'PATCH',
        'DELETE',
        'OPTIONS'
    ],

    allowedHeaders: [
        'Content-Type',
        'Authorization',
        'X-Mora-Token'
    ]
}));

app.use(express.json({
    limit: '12mb'
}));

app.use(express.urlencoded({
    extended: false,
    limit: '12mb'
}));

const apiLimiter = rateLimit({
    windowMs: 60 * 1000,
    limit: 40,
    standardHeaders: 'draft-8',
    legacyHeaders: false,

    message: {
        success: false,
        error: 'Too many requests. Please wait a moment and try again.'
    }
});

const imageLimiter = rateLimit({
    windowMs: 60 * 1000,
    limit: 5,
    standardHeaders: 'draft-8',
    legacyHeaders: false,

    message: {
        success: false,
        error: 'Image generation limit reached. Please try again later.'
    }
});

app.use('/api', apiLimiter);

/* =========================================================
   AI PROVIDERS AND MODELS
   ========================================================= */

const GEMINI_KEY =
    process.env.GEMINI_API_KEY || '';

const GROQ_KEY =
    process.env.GROQ_API_KEY || '';

const OPENROUTER_KEY =
    process.env.OPENROUTER_API_KEY || '';

const DEFAULT_MODELS = {
    gemini:
        process.env.GEMINI_MODEL ||
        'gemini-2.5-flash',

    groq:
        process.env.GROQ_MODEL ||
        'openai/gpt-oss-120b',

    openrouter:
        process.env.OPENROUTER_MODEL ||
        'openai/gpt-oss-120b'
};

const IMAGE_MODELS = {
    gemini:
        process.env.GEMINI_IMAGE_MODEL ||
        'gemini-2.5-flash-image',

    openai:
        process.env.OPENAI_IMAGE_MODEL ||
        'gpt-image-1',

    openrouter:
        process.env.OPENROUTER_IMAGE_MODEL ||
        'google/gemini-2.5-flash-image',

    replicate:
        process.env.REPLICATE_MODEL ||
        'black-forest-labs/flux-schnell'
};

const SYSTEM_PROMPT = `
You are Mora AI, a helpful, capable assistant for conversation,
coding, debugging, planning, writing, and creative work.

Return one valid JSON object without Markdown fences.

Use this structure:

{
  "message": "A clear, useful response for the user",
  "steps": [],
  "files": [],
  "canvas": null,
  "execution": null,
  "terminal": ""
}

Rules:

- Always include a useful, non-empty message.
- Answer ordinary questions directly.
- Do not return an empty message because no files are needed.
- For code-generation requests, provide complete files.
- Each file must have name, path, and content fields.
- For an HTML preview, canvas may contain type, title, and html.
- Steps must be an array.
- Use empty arrays or null for unused fields.
- Never claim code was executed or deployed unless it was.
- Escape JSON strings correctly.
- Return JSON only.
`.trim();

const ai = GEMINI_KEY
    ? new GoogleGenAI({ apiKey: GEMINI_KEY })
    : null;

/* =========================================================
   PROVIDER INFORMATION
   ========================================================= */

function providerInfo() {
    return [
        {
            name: 'gemini',
            configured: Boolean(GEMINI_KEY)
        },
        {
            name: 'groq',
            configured: Boolean(GROQ_KEY)
        },
        {
            name: 'openrouter',
            configured: Boolean(OPENROUTER_KEY)
        }
    ];
}

function imageProviderInfo() {
    return [
        {
            name: 'gemini',
            configured: Boolean(GEMINI_KEY)
        },
        {
            name: 'openai',
            configured: Boolean(process.env.OPENAI_API_KEY)
        },
        {
            name: 'openrouter',
            configured: Boolean(OPENROUTER_KEY)
        },
        {
            name: 'replicate',
            configured: Boolean(process.env.REPLICATE_API_TOKEN)
        }
    ];
}

function getProviderKey(provider, personalKey = '') {
    if (personalKey) {
        return personalKey;
    }

    if (provider === 'gemini') return GEMINI_KEY;
    if (provider === 'groq') return GROQ_KEY;
    if (provider === 'openrouter') return OPENROUTER_KEY;

    return '';
}

function getProviderModel(provider, requestedModel = '') {
    if (!Object.hasOwn(DEFAULT_MODELS, provider)) {
        throw new Error(
            `Unsupported text provider: ${provider}`
        );
    }

    return (
        String(requestedModel || '').trim() ||
        DEFAULT_MODELS[provider]
    );
}

/* =========================================================
   AI RESPONSE PARSING
   ========================================================= */

function extractGeminiText(response) {
    if (typeof response?.text === 'string') {
        return response.text;
    }

    if (typeof response?.text === 'function') {
        return response.text();
    }

    const parts =
        response?.candidates?.[0]?.content?.parts || [];

    return parts
        .filter(part => typeof part.text === 'string')
        .map(part => part.text)
        .join('\n');
}

function parseModelJson(text) {
    const raw = String(text || '').trim();

    if (!raw) {
        throw new Error(
            'The AI provider returned an empty response.'
        );
    }

    const cleaned = raw
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();

    try {
        return JSON.parse(cleaned);
    } catch {
        // Try extracting an object from surrounding text.
    }

    const start = cleaned.indexOf('{');
    const end = cleaned.lastIndexOf('}');

    if (start !== -1 && end > start) {
        try {
            return JSON.parse(
                cleaned.slice(start, end + 1)
            );
        } catch {
            // The caller handles malformed JSON.
        }
    }

    throw new Error(
        'The AI provider returned invalid JSON.'
    );
}

function parseModelResponse(text) {
    const raw = String(text || '').trim();

    if (!raw) {
        throw new Error(
            'The AI provider returned an empty response.'
        );
    }

    try {
        return parseModelJson(raw);
    } catch {
        // Preserve usable plain-text answers.
        return {
            message: raw,
            steps: [],
            files: [],
            canvas: null,
            execution: null,
            terminal: ''
        };
    }
}

/* =========================================================
   NORMALIZE AI RESULTS
   ========================================================= */

function normalizeResult(raw, prompt) {
    let result = raw;

    if (typeof result === 'string') {
        result = {
            message: result
        };
    }

    if (
        !result ||
        typeof result !== 'object' ||
        Array.isArray(result)
    ) {
        result = {};
    }

    let message =
        typeof result.message === 'string'
            ? result.message.trim()
            : '';

    const files = Array.isArray(result.files)
        ? result.files
            .filter(file =>
                file &&
                typeof file === 'object'
            )
            .slice(0, 100)
            .map((file, index) => {
                const name = String(
                    file.name ||
                    file.path ||
                    `file-${index + 1}.txt`
                );

                return {
                    name,
                    path: String(file.path || name),
                    content: String(file.content ?? '')
                };
            })
        : [];

    let canvas = result.canvas ?? null;

    if (typeof canvas === 'string') {
        canvas = {
            type: 'html',
            title: 'Preview',
            html: canvas
        };
    }

    if (canvas && typeof canvas === 'object') {
        if (
            typeof canvas.html !== 'string' &&
            typeof canvas.content === 'string'
        ) {
            canvas.html = canvas.content;
        }
    } else {
        canvas = null;
    }

    const steps = Array.isArray(result.steps)
        ? result.steps
        : [];

    const terminal =
        typeof result.terminal === 'string'
            ? result.terminal
            : '';

    if (!message) {
        if (files.length) {
            message =
                `I created ${files.length} file(s). Open the workspace to review them.`;
        } else if (canvas?.html) {
            message =
                'Your preview is ready. Open the workspace to view it.';
        } else if (terminal) {
            message = terminal;
        } else if (steps.length) {
            message = steps
                .map(step =>
                    String(step?.thought || step?.output || '')
                )
                .filter(Boolean)
                .join('\n');
        } else {
            message =
                'I received your request, but the AI provider returned no usable answer. Please try again.';
        }
    }

    return {
        message,
        steps,
        files,
        canvas,
        execution: result.execution ?? null,
        terminal,
        prompt: String(prompt || '')
    };
}

/* =========================================================
   CONVERSATION CONTEXT
   ========================================================= */

function normalizeHistoryMessages(
    messages,
    currentPrompt = ''
) {
    if (!Array.isArray(messages)) {
        return [];
    }

    const normalized = messages
        .filter(item =>
            item &&
            ['user', 'assistant'].includes(item.role) &&
            typeof item.content === 'string' &&
            item.content.trim()
        )
        .slice(-20)
        .map(item => ({
            role: item.role,
            content: item.content.slice(0, 12000)
        }));

    // Avoid sending the latest user message twice.
    const last = normalized.at(-1);

    if (
        last?.role === 'user' &&
        last.content.trim() === String(currentPrompt).trim()
    ) {
        normalized.pop();
    }

    return normalized;
}

function makeUserContent(prompt, history) {
    const previous =
        normalizeHistoryMessages(history, prompt);

    const transcript = previous.length
        ? '\n\nRECENT CONVERSATION:\n' +
          previous.map(message =>
              `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}`
          ).join('\n\n')
        : '';

    return (
        `${transcript}\n\nCURRENT USER REQUEST:\n${prompt}`
    );
}

/* =========================================================
   GEMINI TEXT REQUEST
   ========================================================= */

async function requestGeminiText(prompt, options = {}) {
    if (!GEMINI_KEY || !ai) {
        throw new Error(
            'Gemini is not configured. Add GEMINI_API_KEY to your server environment.'
        );
    }

    const model = getProviderModel(
        'gemini',
        options.model
    );

    const contents = makeUserContent(
        prompt,
        options.messages
    );

    const response = await ai.models.generateContent({
        model,
        contents: `${SYSTEM_PROMPT}\n\n${contents}`,

        config: {
            responseMimeType: 'application/json',
            temperature: 0.4
        }
    });

    const text = extractGeminiText(response);

    if (!text.trim()) {
        throw new Error(
            'Gemini returned an empty answer. Try again or select another provider.'
        );
    }

    return text;
}

/* =========================================================
   GROQ AND OPENROUTER TEXT REQUESTS
   ========================================================= */

async function requestCompatibleText(
    prompt,
    provider,
    options = {}
) {
    const key = getProviderKey(
        provider,
        options.apiKey || ''
    );

    if (!key) {
        throw new Error(
            `${provider} is not configured. Add its API key in Mora AI Settings or on the server.`
        );
    }

    const model = getProviderModel(
        provider,
        options.model
    );

    const endpoint = provider === 'groq'
        ? 'https://api.groq.com/openai/v1/chat/completions'
        : 'https://openrouter.ai/api/v1/chat/completions';

    const history = normalizeHistoryMessages(
        options.messages,
        prompt
    );

    const messages = [
        {
            role: 'system',
            content: SYSTEM_PROMPT
        },

        ...history,

        {
            role: 'user',
            content: prompt
        }
    ];

    const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`
    };

    if (provider === 'openrouter') {
        headers['HTTP-Referer'] =
            process.env.OPENROUTER_SITE_URL ||
            'http://localhost';

        headers['X-Title'] =
            process.env.OPENROUTER_APP_NAME ||
            'Mora AI';
    }

    const response = await fetch(endpoint, {
        method: 'POST',
        headers,

        body: JSON.stringify({
            model,
            messages,
            temperature: 0.4,
            response_format: {
                type: 'json_object'
            }
        }),

        signal: AbortSignal.timeout(90000)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        const detail =
            data?.error?.message ||
            data?.message ||
            `Provider returned HTTP ${response.status}.`;

        throw new Error(`${provider}: ${detail}`);
    }

    const text =
        data?.choices?.[0]?.message?.content;

    if (typeof text !== 'string' || !text.trim()) {
        throw new Error(
            `${provider} returned an empty message.`
        );
    }

    return text;
}

async function requestProviderText(
    prompt,
    provider,
    options = {}
) {
    if (provider === 'gemini') {
        return requestGeminiText(prompt, options);
    }

    if (
        provider === 'groq' ||
        provider === 'openrouter'
    ) {
        return requestCompatibleText(
            prompt,
            provider,
            options
        );
    }

    throw new Error(
        `Unknown provider: ${provider}`
    );
}

/* =========================================================
   PROVIDER SELECTION AND FALLBACK
   ========================================================= */

function chooseProviders(body) {
    const requested =
        String(body.provider || 'auto').toLowerCase();

    const personalKey =
        String(body.apiKey || '').trim();

    if (personalKey && requested !== 'auto') {
        if (!['gemini', 'groq', 'openrouter'].includes(requested)) {
            throw new Error(
                `Unsupported provider: ${requested}`
            );
        }

        return [requested];
    }

    if (personalKey && requested === 'auto') {
        const keyProvider =
            String(body.providerForKey || '').toLowerCase();

        if (
            ['gemini', 'groq', 'openrouter'].includes(keyProvider)
        ) {
            return [keyProvider];
        }

        throw new Error(
            'Choose the provider associated with your personal API key in Settings.'
        );
    }

    if (requested !== 'auto') {
        if (
            !['gemini', 'groq', 'openrouter'].includes(requested)
        ) {
            throw new Error(
                `Unsupported provider: ${requested}`
            );
        }

        if (!getProviderKey(requested)) {
            throw new Error(
                `${requested} has no server API key. Add a key in Settings or choose Auto.`
            );
        }

        if (body.fallback === false) {
            return [requested];
        }

        return [
            requested,

            ...['gemini', 'groq', 'openrouter']
                .filter(provider => provider !== requested)
                .filter(provider => Boolean(getProviderKey(provider)))
        ];
    }

    const configured = [
        'gemini',
        'groq',
        'openrouter'
    ].filter(provider =>
        Boolean(getProviderKey(provider))
    );

    if (!configured.length) {
        throw new Error(
            'No text AI provider is configured. Add a Gemini, Groq, or OpenRouter API key in Settings.'
        );
    }

    return configured;
}

async function generateAIResponse(
    prompt,
    body = {}
) {
    const providers = chooseProviders(body);
    const errors = [];

    for (const provider of providers) {
        try {
            const raw = await requestProviderText(
                prompt,
                provider,
                {
                    apiKey: body.apiKey || '',
                    model: body.model || '',
                    messages: body.messages || []
                }
            );

            const parsed = parseModelResponse(raw);
            const result = normalizeResult(parsed, prompt);

            return {
                ...result,
                success: true,
                provider,
                model:
                    body.model ||
                    DEFAULT_MODELS[provider]
            };
        } catch (error) {
            console.error(
                `[Mora AI] ${provider} failed:`,
                error.message
            );

            errors.push(
                `${provider}: ${error.message}`
            );

            if (
                body.fallback === false ||
                body.apiKey
            ) {
                break;
            }
        }
    }

    throw new Error(
        `Mora AI could not generate a response.\n${errors.join('\n')}`
    );
}

/* =========================================================
   COMMON HELPERS
   ========================================================= */

function safeId(value) {
    const id = String(value || '');

    return /^[a-zA-Z0-9_-]{1,100}$/.test(id)
        ? id
        : null;
}

function cleanTitle(
    value,
    fallback = 'Untitled'
) {
    return (
        String(value || fallback)
            .trim()
            .slice(0, 120) ||
        fallback
    );
}

function cleanFiles(files) {
    if (!Array.isArray(files)) {
        return [];
    }

    return files
        .filter(file =>
            file &&
            typeof file === 'object'
        )
        .slice(0, 100)
        .map((file, index) => {
            let name = String(
                file.name ||
                file.path ||
                `file-${index + 1}.txt`
            ).replaceAll('\\', '/');

            name = path.posix.basename(name)
                .replace(/[^a-zA-Z0-9._ -]/g, '_');

            if (
                !name ||
                name === '.' ||
                name === '..'
            ) {
                name = `file-${index + 1}.txt`;
            }

            return {
                name,
                path: String(
                    file.path || name
                ).slice(0, 300),

                content: String(
                    file.content ?? ''
                ).slice(0, 2_000_000)
            };
        });
}

function publicConversation(conversation) {
    return {
        id: conversation.id,
        title: conversation.title,
        createdAt: conversation.createdAt,
        updatedAt: conversation.updatedAt,

        messages: Array.isArray(conversation.messages)
            ? conversation.messages
            : []
    };
}

/* =========================================================
   HEALTH
   ========================================================= */

app.get('/api/health', (_req, res) => {
    res.json({
        status: 'online',
        providers: providerInfo(),
        imageProviders: imageProviderInfo(),
        e2bConfigured: Boolean(process.env.E2B_API_KEY),
        timestamp: new Date().toISOString()
    });
});

app.get('/api/providers', auth, (_req, res) => {
    res.json({
        providers: providerInfo(),
        imageProviders: imageProviderInfo()
    });
});

/* =========================================================
   AI CHAT
   ========================================================= */

app.post('/api/agent', auth, async (req, res) => {
    const body = req.body || {};

    const prompt = String(
        body.prompt || body.message || ''
    ).trim();

    if (!prompt) {
        return res.status(400).json({
            success: false,
            error: 'Enter a message before sending.'
        });
    }

    if (prompt.length > 12000) {
        return res.status(413).json({
            success: false,
            error: 'Your prompt is too long. Please shorten it.'
        });
    }

    try {
        const result = await generateAIResponse(
            prompt,
            body
        );

        const history = await readJson(
            HISTORY_FILE,
            []
        );

        history.unshift({
            id: crypto.randomUUID(),
            prompt,
            provider: result.provider,
            model: result.model,
            files: result.files,
            createdAt: new Date().toISOString()
        });

        await writeJson(
            HISTORY_FILE,
            history.slice(0, 500)
        );

        res.json(result);
    } catch (error) {
        console.error(
            '[Mora AI] Request failed:',
            error.message
        );

        res.status(502).json({
            success: false,
            error: error.message ||
                'The AI request failed.'
        });
    }
});
/* =========================================================
   AI CHAT STREAMING COMPATIBILITY ENDPOINT
   Compatible with public/index.html readSse()
   ========================================================= */

app.post(
    '/api/agent/stream',
    auth,
    async (req, res) => {
        const body = req.body || {};

        const prompt = String(
            body.prompt || body.message || ''
        ).trim();

        if (!prompt) {
            return res.status(400).json({
                success: false,
                error: 'Enter a message before sending.'
            });
        }

        if (prompt.length > 12000) {
            return res.status(413).json({
                success: false,
                error: 'Your prompt is too long. Please shorten it.'
            });
        }

        /*
         * This endpoint sends Server-Sent Events (SSE).
         * The current AI provider functions generate a complete
         * response rather than token-by-token streaming.
         *
         * We therefore send progress events followed by the
         * completed result in the format expected by the UI.
         */

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

        req.on('close', () => {
            if (!res.writableEnded) {
                disconnected = true;
            }
        });

        function sendEvent(event, data) {
            if (
                disconnected ||
                res.writableEnded ||
                res.destroyed
            ) {
                return false;
            }

            res.write(`event: ${event}\n`);
            res.write(`data: ${JSON.stringify(data)}\n\n`);

            return true;
        }

        try {
            sendEvent('progress', {
                message: 'Preparing your request...'
            });

            const result = await generateAIResponse(
                prompt,
                body
            );

            if (disconnected) {
                return;
            }

            /*
             * Send the entire completed result.
             * The frontend's readSse() handler will receive
             * this as the "result" event.
             */

            sendEvent('result', result);

            await writeJson(
                HISTORY_FILE,
                [
                    {
                        id: crypto.randomUUID(),
                        prompt,
                        provider: result.provider,
                        model: result.model,
                        files: result.files,
                        createdAt: new Date().toISOString()
                    },

                    ...(
                        await readJson(HISTORY_FILE, [])
                    )
                ].slice(0, 500)
            );

        } catch (error) {
            console.error(
                '[Mora AI streaming request failed]:',
                error.message
            );

            sendEvent('error', {
                error: error.message ||
                    'The AI request failed.'
            });
        } finally {
            if (!res.writableEnded) {
                res.end();
            }
        }
    }
);

/* =========================================================
   PROVIDER CONNECTION TEST
   ========================================================= */

app.post('/api/test', auth, async (req, res) => {
    const body = req.body || {};

    try {
        const providers = chooseProviders({
            ...body,
            fallback: false
        });

        const provider = providers[0];

        const raw = await requestProviderText(
            'Return a JSON object with a short message confirming that Mora AI is connected.',
            provider,
            {
                apiKey: body.apiKey || '',
                model: body.model || '',
                messages: []
            }
        );

        let message = 'Connection successful.';

        try {
            const parsed = parseModelJson(raw);

            if (
                typeof parsed.message === 'string' &&
                parsed.message.trim()
            ) {
                message = parsed.message.trim();
            }
        } catch {
            if (raw.trim()) {
                message = raw.trim().slice(0, 500);
            }
        }

        res.json({
            success: true,
            provider,
            message
        });
    } catch (error) {
        res.status(502).json({
            success: false,
            error: error.message ||
                'Provider connection test failed.'
        });
    }
});

/* =========================================================
   CONVERSATIONS
   ========================================================= */

app.get('/api/conversations', auth, async (_req, res) => {
    const conversations = await readJson(
        CONVERSATIONS_FILE,
        []
    );

    res.json({
        conversations: conversations
            .slice()
            .sort((a, b) =>
                new Date(b.updatedAt || b.createdAt || 0) -
                new Date(a.updatedAt || a.createdAt || 0)
            )
            .map(conversation => ({
                id: conversation.id,
                title: conversation.title,
                createdAt: conversation.createdAt,
                updatedAt: conversation.updatedAt,
                messages: []
            }))
    });
});

app.get(
    '/api/conversations/:id',
    auth,
    async (req, res) => {
        const conversations = await readJson(
            CONVERSATIONS_FILE,
            []
        );

        const conversation = conversations.find(
            item => item.id === req.params.id
        );

        if (!conversation) {
            return res.status(404).json({
                success: false,
                error: 'Conversation not found.'
            });
        }

        res.json({
            conversation: publicConversation(conversation)
        });
    }
);

app.post('/api/conversations', auth, async (req, res) => {
    const body = req.body || {};

    const conversations = await readJson(
        CONVERSATIONS_FILE,
        []
    );

    const suppliedId = safeId(body.id);

    let conversation = suppliedId
        ? conversations.find(item =>
            item.id === suppliedId
        )
        : null;

    const now = new Date().toISOString();

    const messages = Array.isArray(body.messages)
        ? body.messages.slice(-300).map(message => ({
            id: String(
                message.id || crypto.randomUUID()
            ),

            role: message.role === 'assistant'
                ? 'assistant'
                : 'user',

            content: String(
                message.content || ''
            ).slice(0, 30000),

            images: Array.isArray(message.images)
                ? message.images
                    .filter(image =>
                        image &&
                        typeof image.url === 'string'
                    )
                    .slice(0, 10)
                    .map(image => ({
                        url: image.url.slice(0, 2000),
                        alt: String(image.alt || '').slice(0, 500),
                        prompt: String(image.prompt || '').slice(0, 2000),
                        name: String(image.name || '').slice(0, 200),
                        provider: String(image.provider || '').slice(0, 100),
                        model: String(image.model || '').slice(0, 200)
                    }))
                : [],

            createdAt: message.createdAt || now
        }))
        : [];

    if (conversation) {
        conversation.title = cleanTitle(
            body.title,
            conversation.title
        );

        conversation.messages = messages;
        conversation.updatedAt = now;
    } else {
        conversation = {
            id: suppliedId || crypto.randomUUID(),
            title: cleanTitle(
                body.title,
                'New conversation'
            ),
            messages,
            createdAt: now,
            updatedAt: now
        };

        conversations.unshift(conversation);
    }

    await writeJson(
        CONVERSATIONS_FILE,
        conversations.slice(0, 500)
    );

    res.json({
        success: true,
        conversation: publicConversation(conversation)
    });
});

app.delete(
    '/api/conversations/:id',
    auth,
    async (req, res) => {
        const conversations = await readJson(
            CONVERSATIONS_FILE,
            []
        );

        const remaining = conversations.filter(
            item => item.id !== req.params.id
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

        res.json({ success: true });
    }
);

/* =========================================================
   HISTORY
   ========================================================= */

app.get('/api/history', auth, async (_req, res) => {
    res.json({
        history: await readJson(HISTORY_FILE, [])
    });
});

app.delete('/api/history', auth, async (_req, res) => {
    await writeJson(HISTORY_FILE, []);

    res.json({ success: true });
});

/* =========================================================
   PROJECTS
   ========================================================= */

app.get('/api/projects', auth, async (_req, res) => {
    const projects = await readJson(
        PROJECTS_FILE,
        []
    );

    res.json({
        projects: projects.map(project => ({
            ...project,
            files: Array.isArray(project.files)
                ? project.files
                : []
        }))
    });
});

app.get('/api/projects/:id', auth, async (req, res) => {
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

    res.json({ project });
});

app.post('/api/projects', auth, async (req, res) => {
    const body = req.body || {};

    const projects = await readJson(
        PROJECTS_FILE,
        []
    );

    const id = safeId(body.id);
    const now = new Date().toISOString();

    let project = id
        ? projects.find(item => item.id === id)
        : null;

    const newData = {
        title: cleanTitle(
            body.title,
            'Mora AI Project'
        ),

        prompt: String(
            body.prompt || ''
        ).slice(0, 12000),

        files: cleanFiles(body.files),
        updatedAt: now
    };

    if (project) {
        Object.assign(project, newData);
    } else {
        project = {
            id: id || crypto.randomUUID(),
            ...newData,
            createdAt: now
        };

        projects.unshift(project);
    }

    await writeJson(
        PROJECTS_FILE,
        projects.slice(0, 300)
    );

    res.json({
        success: true,
        project
    });
});

app.delete('/api/projects/:id', auth, async (req, res) => {
    const projects = await readJson(
        PROJECTS_FILE,
        []
    );

    const remaining = projects.filter(
        item => item.id !== req.params.id
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

    res.json({ success: true });
});

/* =========================================================
   PROJECT ZIP EXPORT
   ========================================================= */

async function exportProjectZip(project, res) {
    res.setHeader(
        'Content-Type',
        'application/zip'
    );

    const filename = cleanTitle(
        project.title,
        'mora-project'
    ).replace(/[^a-zA-Z0-9_-]/g, '_');

    res.setHeader(
        'Content-Disposition',
        `attachment; filename="${filename}.zip"`
    );

    const archive = archiver('zip', {
        zlib: { level: 6 }
    });

    archive.on('error', error => {
        console.error(
            'ZIP export error:',
            error.message
        );

        if (!res.headersSent) {
            res.status(500).json({
                success: false,
                error: 'Could not create the ZIP archive.'
            });
        } else {
            res.destroy(error);
        }
    });

    archive.pipe(res);

    for (const file of cleanFiles(project.files)) {
        archive.append(file.content, {
            name: file.name
        });
    }

    await archive.finalize();
}

async function handleProjectExport(req, res) {
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

    try {
        await exportProjectZip(project, res);
    } catch (error) {
        console.error(
            'Project export failed:',
            error.message
        );

        if (!res.headersSent) {
            res.status(500).json({
                success: false,
                error: 'ZIP export failed.'
            });
        }
    }
}

app.get(
    '/api/projects/:id/export',
    auth,
    handleProjectExport
);

app.get(
    '/api/projects/:id/export.zip',
    auth,
    handleProjectExport
);

/* =========================================================
   IMAGE PROVIDERS
   ========================================================= */

function getImageKey(provider, keys = {}) {
    if (
        typeof keys[provider] === 'string' &&
        keys[provider].trim()
    ) {
        return keys[provider].trim();
    }

    if (provider === 'gemini') return GEMINI_KEY;
    if (provider === 'openai') return process.env.OPENAI_API_KEY || '';
    if (provider === 'openrouter') return OPENROUTER_KEY;
    if (provider === 'replicate') return process.env.REPLICATE_API_TOKEN || '';

    return '';
}

function getImageModel(provider, models = {}) {
    if (
        typeof models[provider] === 'string' &&
        models[provider].trim()
    ) {
        return models[provider].trim();
    }

    return IMAGE_MODELS[provider];
}

function dataUrlToInlineData(inputImage) {
    if (
        !inputImage ||
        typeof inputImage.data !== 'string'
    ) {
        return null;
    }

    const mimeType = String(
        inputImage.mimeType || ''
    );

    if (![
        'image/png',
        'image/jpeg',
        'image/webp'
    ].includes(mimeType)) {
        throw new Error(
            'Attached images must be PNG, JPEG, or WebP.'
        );
    }

    const data = inputImage.data.replace(/\s/g, '');

    if (
        !data ||
        data.length > 9_000_000 ||
        !/^[A-Za-z0-9+/=]+$/.test(data)
    ) {
        throw new Error(
            'The attached image data is invalid or too large.'
        );
    }

    return {
        inlineData: {
            mimeType,
            data
        }
    };
}

/* =========================================================
   GEMINI IMAGE GENERATION
   ========================================================= */

async function generateGeminiImage(
    prompt,
    key,
    model,
    inputImage
) {
    const client = new GoogleGenAI({
        apiKey: key
    });

    const contents = [{ text: prompt }];

    const imagePart =
        dataUrlToInlineData(inputImage);

    if (imagePart) {
        contents.push(imagePart);
    }

    const result = await client.models.generateContent({
        model,
        contents,

        config: {
            responseModalities: ['TEXT', 'IMAGE']
        }
    });

    const parts =
        result?.candidates?.[0]?.content?.parts || [];

    for (const part of parts) {
        if (part.inlineData?.data) {
            return {
                base64: part.inlineData.data,
                mime: part.inlineData.mimeType || 'image/png'
            };
        }
    }

    const explanation = parts
        .map(part => part.text || '')
        .filter(Boolean)
        .join('\n');

    throw new Error(
        explanation ||
        'Gemini did not return image data. Check the selected image model and API access.'
    );
}

/* =========================================================
   OPENAI IMAGE GENERATION
   ========================================================= */

async function generateOpenAIImage(
    prompt,
    key,
    model
) {
    const response = await fetch(
        'https://api.openai.com/v1/images/generations',
        {
            method: 'POST',

            headers: {
                Authorization: `Bearer ${key}`,
                'Content-Type': 'application/json'
            },

            body: JSON.stringify({
                model,
                prompt,
                n: 1,
                size: '1024x1024'
            }),

            signal: AbortSignal.timeout(120000)
        }
    );

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data?.error?.message ||
            `OpenAI returned HTTP ${response.status}.`
        );
    }

    const item = data?.data?.[0];

    if (item?.b64_json) {
        return {
            base64: item.b64_json,
            mime: 'image/png'
        };
    }

    if (item?.url) {
        const imageResponse = await fetch(item.url, {
            signal: AbortSignal.timeout(60000)
        });

        if (!imageResponse.ok) {
            throw new Error(
                'Could not download the generated OpenAI image.'
            );
        }

        const buffer = Buffer.from(
            await imageResponse.arrayBuffer()
        );

        return {
            base64: buffer.toString('base64'),
            mime:
                imageResponse.headers.get('content-type') ||
                'image/png'
        };
    }

    throw new Error(
        'OpenAI returned no image data.'
    );
}

/* =========================================================
   OPENROUTER IMAGE GENERATION
   ========================================================= */

async function generateOpenRouterImage(
    prompt,
    key,
    model,
    inputImage
) {
    const userContent = [{
        type: 'text',
        text: prompt
    }];

    if (inputImage?.data) {
        const inline =
            dataUrlToInlineData(inputImage).inlineData;

        userContent.push({
            type: 'image_url',

            image_url: {
                url: `data:${inline.mimeType};base64,${inline.data}`
            }
        });
    }

    const response = await fetch(
        'https://openrouter.ai/api/v1/chat/completions',
        {
            method: 'POST',

            headers: {
                Authorization: `Bearer ${key}`,
                'Content-Type': 'application/json',
                'X-Title': 'Mora AI'
            },

            body: JSON.stringify({
                model,

                messages: [{
                    role: 'user',
                    content: userContent
                }],

                modalities: ['image', 'text']
            }),

            signal: AbortSignal.timeout(120000)
        }
    );

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data?.error?.message ||
            `OpenRouter returned HTTP ${response.status}.`
        );
    }

    const message =
        data?.choices?.[0]?.message;

    const image =
        message?.images?.[0];

    const imageUrl =
        image?.image_url?.url ||
        image?.url ||
        (typeof image === 'string' ? image : '');

    if (imageUrl.startsWith('data:')) {
        const match = imageUrl.match(
            /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/
        );

        if (match) {
            return {
                base64: match[2],
                mime: match[1]
            };
        }
    }

    if (imageUrl.startsWith('https://')) {
        const imageResponse = await fetch(imageUrl, {
            signal: AbortSignal.timeout(60000)
        });

        if (!imageResponse.ok) {
            throw new Error(
                'Could not download the OpenRouter image.'
            );
        }

        const buffer = Buffer.from(
            await imageResponse.arrayBuffer()
        );

        return {
            base64: buffer.toString('base64'),
            mime:
                imageResponse.headers.get('content-type') ||
                'image/png'
        };
    }

    throw new Error(
        message?.content ||
        'OpenRouter did not return an image. Check whether the selected model supports image generation.'
    );
}

/* =========================================================
   REPLICATE IMAGE GENERATION
   ========================================================= */

async function generateReplicateImage(
    prompt,
    key,
    model
) {
    const endpoint =
        'https://api.replicate.com/v1/models/' +
        model.split('/').map(encodeURIComponent).join('/') +
        '/predictions';

    const response = await fetch(endpoint, {
        method: 'POST',

        headers: {
            Authorization: `Bearer ${key}`,
            'Content-Type': 'application/json',
            Prefer: 'wait=60'
        },

        body: JSON.stringify({
            input: { prompt }
        }),

        signal: AbortSignal.timeout(120000)
    });

    let data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data?.detail ||
            data?.error ||
            `Replicate returned HTTP ${response.status}.`
        );
    }

    if (
        data.status === 'starting' ||
        data.status === 'processing'
    ) {
        const pollUrl = data.urls?.get;

        if (!pollUrl) {
            throw new Error(
                'Replicate is still processing the image. Please try again.'
            );
        }

        for (let i = 0; i < 20; i++) {
            await new Promise(resolve =>
                setTimeout(resolve, 3000)
            );

            const poll = await fetch(pollUrl, {
                headers: {
                    Authorization: `Bearer ${key}`
                },

                signal: AbortSignal.timeout(30000)
            });

            data = await poll.json().catch(() => ({}));

            if (data.status === 'succeeded') {
                break;
            }

            if (
                data.status === 'failed' ||
                data.status === 'canceled'
            ) {
                throw new Error(
                    data.error ||
                    'Replicate image generation failed.'
                );
            }
        }
    }

    if (data.status !== 'succeeded') {
        throw new Error(
            'Replicate did not finish generating the image in time.'
        );
    }

    const output = Array.isArray(data.output)
        ? data.output[0]
        : data.output;

    if (
        typeof output !== 'string' ||
        !output.startsWith('https://')
    ) {
        throw new Error(
            'Replicate returned no downloadable image.'
        );
    }

    const imageResponse = await fetch(output, {
        signal: AbortSignal.timeout(60000)
    });

    if (!imageResponse.ok) {
        throw new Error(
            'Could not download the Replicate image.'
        );
    }

    const buffer = Buffer.from(
        await imageResponse.arrayBuffer()
    );

    return {
        base64: buffer.toString('base64'),
        mime:
            imageResponse.headers.get('content-type') ||
            'image/png'
    };
}

/* =========================================================
   IMAGE GENERATION ENDPOINT
   ========================================================= */

app.post(
    '/api/images/generate',
    auth,
    imageLimiter,
    async (req, res) => {
        const body = req.body || {};

        const prompt = String(
            body.prompt || ''
        ).trim();

        if (!prompt) {
            return res.status(400).json({
                success: false,
                error: 'Describe the image you want to create.'
            });
        }

        if (prompt.length > 6000) {
            return res.status(413).json({
                success: false,
                error: 'The image prompt is too long.'
            });
        }

        const requested =
            String(body.provider || 'auto').toLowerCase();

        const allowed = [
            'gemini',
            'openai',
            'openrouter',
            'replicate'
        ];

        if (
            requested !== 'auto' &&
            !allowed.includes(requested)
        ) {
            return res.status(400).json({
                success: false,
                error: 'Unsupported image provider.'
            });
        }

        const candidates = requested === 'auto'
            ? allowed
            : [requested];

        const failures = [];

        for (const provider of candidates) {
            const key = getImageKey(
                provider,
                body.imageKeys || {}
            );

            if (!key) {
                failures.push(
                    `${provider}: no API key configured`
                );

                continue;
            }

            const model = getImageModel(
                provider,
                body.models || {}
            );

            try {
                let image;

                if (provider === 'gemini') {
                    image = await generateGeminiImage(
                        prompt,
                        key,
                        model,
                        body.inputImage
                    );
                } else if (provider === 'openai') {
                    if (body.inputImage) {
                        throw new Error(
                            'This OpenAI image endpoint is configured for generation, not image editing.'
                        );
                    }

                    image = await generateOpenAIImage(
                        prompt,
                        key,
                        model
                    );
                } else if (provider === 'openrouter') {
                    image = await generateOpenRouterImage(
                        prompt,
                        key,
                        model,
                        body.inputImage
                    );
                } else {
                    if (body.inputImage) {
                        throw new Error(
                            'This Replicate model is configured for text-to-image generation, not image editing.'
                        );
                    }

                    image = await generateReplicateImage(
                        prompt,
                        key,
                        model
                    );
                }

                const buffer = Buffer.from(
                    image.base64,
                    'base64'
                );

                if (
                    !buffer.length ||
                    buffer.length > 25 * 1024 * 1024
                ) {
                    throw new Error(
                        'The returned image is empty or too large.'
                    );
                }

                const extension =
                    image.mime === 'image/jpeg'
                        ? 'jpg'
                        : image.mime === 'image/webp'
                            ? 'webp'
                            : 'png';

                const filename =
                    `mora-${Date.now()}-${crypto.randomBytes(4).toString('hex')}.${extension}`;

                await fs.writeFile(
                    path.join(GENERATED_DIR, filename),
                    buffer
                );

                const url =
                    `/generated-images/${filename}`;

                const imageResult = {
                    url,
                    name: filename,
                    prompt,
                    provider,
                    model,
                    mime: image.mime,
                    bytes: buffer.length,
                    createdAt: new Date().toISOString()
                };

                return res.json({
                    success: true,
                    image: imageResult,
                    images: [imageResult],
                    url,
                    filename,
                    provider,
                    model
                });
            } catch (error) {
                console.error(
                    `[Mora AI image] ${provider} failed:`,
                    error.message
                );

                failures.push(
                    `${provider}: ${error.message}`
                );

                if (requested !== 'auto') {
                    break;
                }
            }
        }

        res.status(502).json({
            success: false,
            error:
                `Image generation failed.\n${failures.join('\n')}`
        });
    }
);

/* =========================================================
   HTML TESTING
   ========================================================= */

app.post('/api/test/html', auth, async (req, res) => {
    const html = String(req.body?.html || '');

    if (!html.trim()) {
        return res.status(400).json({
            success: false,
            error: 'Provide HTML to test.'
        });
    }

    if (html.length > 2_000_000) {
        return res.status(413).json({
            success: false,
            error: 'The HTML document is too large to test.'
        });
    }

    const title =
        html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]
            ?.replace(/<[^>]*>/g, '')
            .trim() || '';

    const hasViewport =
        /<meta[^>]+name\s*=\s*["']viewport["']/i.test(html);

    const elementCount =
        (html.match(/<[a-z][\w:-]*(?:\s[^<>]*?)?>/gi) || []).length;

    res.json({
        success: true,
        passed: Boolean(title && hasViewport),
        browser: 'Static HTML check',

        details: {
            title,
            hasViewport,
            elementCount,
            horizontalOverflow: false
        },

        consoleErrors: [],
        pageErrors: [],

        note:
            'This endpoint checks the HTML source only. It does not execute JavaScript in a real browser.'
    });
});

/* =========================================================
   OPTIONAL PYTHON EXECUTION
   ========================================================= */

app.post(
    '/api/execute/python',
    auth,
    async (req, res) => {
        const code = String(
            req.body?.code || ''
        ).trim();

        if (!code) {
            return res.status(400).json({
                success: false,
                error: 'Provide Python code to execute.'
            });
        }

        if (code.length > 50000) {
            return res.status(413).json({
                success: false,
                error: 'Python code is too large.'
            });
        }

        if (!process.env.E2B_API_KEY) {
            return res.status(503).json({
                success: false,
                error: 'Python execution is not configured. Add E2B_API_KEY to the server environment.'
            });
        }

        let sandbox;

        try {
            const { Sandbox } =
                await import('@e2b/code-interpreter');

            sandbox = await Sandbox.create({
                apiKey: process.env.E2B_API_KEY
            });

            const result =
                await sandbox.runCode(code);

            const output = [
                ...(result.logs?.stdout || []),
                ...(result.logs?.stderr || [])
            ].join('\n');

            res.json({
                success: true,
                output,
                results: result.results || [],
                error: null
            });
        } catch (error) {
            res.status(500).json({
                success: false,
                error:
                    error.message ||
                    'Python execution failed.'
            });
        } finally {
            if (
                sandbox &&
                typeof sandbox.close === 'function'
            ) {
                try {
                    await sandbox.close();
                } catch {
                    // Ignore sandbox cleanup errors.
                }
            }
        }
    }
);
/* =========================================================
   MORA CYBERLAB
   Security Analysis, Code Auditing and Report Management
   ========================================================= */

/*
 * CyberLab uses the existing:
 *
 * - auth()
 * - readJson()
 * - writeJson()
 * - generateAIResponse()
 * - safeId()
 * - cleanTitle()
 * - apiLimiter
 *
 * No additional npm packages are required.
 *
 * Assessments analyze submitted material. They do not
 * automatically scan external targets or exploit systems.
 */

const CYBERLAB_SYSTEM_PROMPT = `
You are Mora CyberLab, a defensive cybersecurity analysis
assistant integrated into Mora AI.

Your responsibilities include:

1. Source-code security reviews.
2. Vulnerability identification and explanation.
3. Secure coding recommendations.
4. API security design reviews.
5. Web application security assessments.
6. Defensive configuration reviews.
7. Threat modeling and attack-surface analysis.
8. Incident-response guidance.
9. Log and alert interpretation.
10. CTF and educational security analysis.
11. Authorized penetration-test planning.
12. Remediation verification planning.

Analyze only the evidence provided by the user.

Do not claim to have scanned, executed, accessed, or tested
a live system unless a tool actually performed that action.

Do not invent vulnerabilities, test results, affected files,
HTTP responses, or evidence.

Distinguish confirmed findings from potential risks.

Do not expose secrets found in submitted material.
Redact passwords, access tokens, API keys, and private keys
in any returned evidence excerpts.

Return one valid JSON object with this structure:

{
  "message": "Executive summary",
  "assessment": {
    "title": "Assessment title",
    "scope": [],
    "methodology": [],
    "limitations": [],
    "riskRating": "Informational",
    "findingsCount": {
      "critical": 0,
      "high": 0,
      "medium": 0,
      "low": 0,
      "informational": 0
    }
  },
  "findings": [
    {
      "id": "CL-001",
      "title": "Finding title",
      "severity": "Medium",
      "confidence": "Medium",
      "status": "Needs review",
      "category": "Security",
      "affectedComponent": "Component",
      "description": "Description",
      "evidence": "Redacted evidence",
      "impact": "Potential impact",
      "recommendation": "Recommended remediation",
      "verification": "How to verify the fix"
    }
  ],
  "recommendations": [],
  "nextSteps": [],
  "limitations": []
}

Rules:

- Use only Critical, High, Medium, Low, or Informational
  for finding severity.
- Use only High, Medium, or Low for confidence.
- Findings must be supported by submitted evidence.
- Use an empty findings array if no meaningful finding exists.
- Never mark a hypothetical issue as confirmed.
- Explain uncertainty.
- Provide actionable defensive recommendations.
- Avoid returning secrets or credentials.
- Keep the report technically useful.
- Return JSON only, without Markdown fences.
`.trim();

/* ---------------------------------------------------------
   CyberLab validation helpers
   --------------------------------------------------------- */

const CYBERLAB_SEVERITIES = [
    'Critical',
    'High',
    'Medium',
    'Low',
    'Informational'
];

const CYBERLAB_CONFIDENCE = [
    'High',
    'Medium',
    'Low'
];

function cyberlabText(value, maxLength = 12000) {
    return String(value ?? '')
        .replace(/\u0000/g, '')
        .slice(0, maxLength);
}

function cyberlabSeverity(value) {
    const match = CYBERLAB_SEVERITIES.find(
        severity =>
            severity.toLowerCase() ===
            String(value || '').toLowerCase()
    );

    return match || 'Informational';
}

function cyberlabConfidence(value) {
    const match = CYBERLAB_CONFIDENCE.find(
        confidence =>
            confidence.toLowerCase() ===
            String(value || '').toLowerCase()
    );

    return match || 'Medium';
}

function cyberlabRiskFromFindings(findings) {
    const severities = findings.map(
        finding => finding.severity
    );

    if (severities.includes('Critical')) {
        return 'Critical';
    }

    if (severities.includes('High')) {
        return 'High';
    }

    if (severities.includes('Medium')) {
        return 'Medium';
    }

    if (severities.includes('Low')) {
        return 'Low';
    }

    return 'Informational';
}

/*
 * Redact common secret formats before saving report output.
 * This is a precaution, not a guarantee that every secret
 * format can be detected.
 */

function cyberlabRedactSecrets(value) {
    let text = String(value ?? '');

    const patterns = [
        /\bAKIA[0-9A-Z]{16}\b/g,

        /\bBearer\s+[A-Za-z0-9._~+/=-]+/gi,

        /\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*["']?[^"'\s,;]+/gi,

        /\bsk-[A-Za-z0-9_-]{12,}\b/g,

        /\bgh[pousr]_[A-Za-z0-9_]{20,}\b/g,

        /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/g
    ];

    for (const pattern of patterns) {
        text = text.replace(pattern, match => {
            if (/PRIVATE KEY/.test(match)) {
                return '[REDACTED PRIVATE KEY]';
            }

            if (/^Bearer/i.test(match)) {
                return 'Bearer [REDACTED]';
            }

            if (/^(AKIA|sk-|gh[pousr]_)/.test(match)) {
                return '[REDACTED SECRET]';
            }

            const separator = match.search(/[:=]/);

            if (separator >= 0) {
                return (
                    match.slice(0, separator + 1) +
                    ' [REDACTED]'
                );
            }

            return '[REDACTED SECRET]';
        });
    }

    return text;
}

function cyberlabNormalizeFinding(finding, index) {
    const source = (
        finding &&
        typeof finding === 'object' &&
        !Array.isArray(finding)
    )
        ? finding
        : {};

    return {
        id: `CL-${String(index + 1).padStart(3, '0')}`,

        title: cyberlabText(
            source.title || 'Untitled finding',
            250
        ),

        severity: cyberlabSeverity(
            source.severity
        ),

        confidence: cyberlabConfidence(
            source.confidence
        ),

        status: cyberlabText(
            source.status || 'Needs review',
            80
        ),

        category: cyberlabText(
            source.category || 'Security',
            120
        ),

        affectedComponent: cyberlabText(
            source.affectedComponent || 'Not specified',
            500
        ),

        description: cyberlabRedactSecrets(
            cyberlabText(source.description, 6000)
        ),

        evidence: cyberlabRedactSecrets(
            cyberlabText(source.evidence, 6000)
        ),

        impact: cyberlabRedactSecrets(
            cyberlabText(source.impact, 3000)
        ),

        recommendation: cyberlabRedactSecrets(
            cyberlabText(source.recommendation, 6000)
        ),

        verification: cyberlabRedactSecrets(
            cyberlabText(source.verification, 3000)
        )
    };
}

function cyberlabCountFindings(findings) {
    const counts = {
        critical: 0,
        high: 0,
        medium: 0,
        low: 0,
        informational: 0
    };

    for (const finding of findings) {
        const key = finding.severity.toLowerCase();

        if (Object.hasOwn(counts, key)) {
            counts[key]++;
        }
    }

    return counts;
}

function cyberlabNormalizeReport(
    result,
    input,
    provider,
    model
) {
    const source =
        result && typeof result === 'object'
            ? result
            : {};

    const assessment =
        source.assessment &&
        typeof source.assessment === 'object'
            ? source.assessment
            : {};

    const findings = Array.isArray(source.findings)
        ? source.findings
            .slice(0, 100)
            .map(cyberlabNormalizeFinding)
        : [];

    const recommendations = Array.isArray(
        source.recommendations
    )
        ? source.recommendations
            .slice(0, 50)
            .map(item =>
                cyberlabRedactSecrets(
                    cyberlabText(item, 3000)
                )
            )
        : [];

    const nextSteps = Array.isArray(
        source.nextSteps
    )
        ? source.nextSteps
            .slice(0, 30)
            .map(item =>
                cyberlabRedactSecrets(
                    cyberlabText(item, 2000)
                )
            )
        : [];

    const limitations = Array.isArray(
        source.limitations
    )
        ? source.limitations
            .slice(0, 30)
            .map(item =>
                cyberlabRedactSecrets(
                    cyberlabText(item, 2000)
                )
            )
        : [];

    return {
        title: cleanTitle(
            assessment.title || input.title,
            'CyberLab Security Assessment'
        ),

        message: cyberlabRedactSecrets(
            cyberlabText(
                source.message ||
                'CyberLab assessment completed.',
                12000
            )
        ),

        assessment: {
            scope: Array.isArray(assessment.scope)
                ? assessment.scope
                    .slice(0, 30)
                    .map(item =>
                        cyberlabText(item, 1000)
                    )
                : [
                    cyberlabText(
                        input.target || 'User-submitted material',
                        1000
                    )
                ],

            methodology: Array.isArray(
                assessment.methodology
            )
                ? assessment.methodology
                    .slice(0, 30)
                    .map(item =>
                        cyberlabText(item, 1000)
                    )
                : [
                    'AI-assisted review of user-submitted evidence'
                ],

            limitations: Array.isArray(
                assessment.limitations
            )
                ? assessment.limitations
                    .slice(0, 30)
                    .map(item =>
                        cyberlabText(item, 1000)
                    )
                : [
                    'No live target testing was performed by this assessment endpoint.'
                ],

            riskRating: cyberlabRiskFromFindings(
                findings
            ),

            findingsCount: cyberlabCountFindings(
                findings
            )
        },

        findings,
        recommendations,
        nextSteps,
        limitations,

        engine: {
            provider: cyberlabText(provider, 100),
            model: cyberlabText(model, 200)
        }
    };
}

/* ---------------------------------------------------------
   CyberLab health and capability information
   --------------------------------------------------------- */

app.get(
    '/api/cyber/health',
    auth,
    async (_req, res) => {
        res.json({
            success: true,
            name: 'Mora CyberLab',
            status: 'online',

            capabilities: [
                'source-code-review',
                'vulnerability-analysis',
                'api-security-review',
                'web-application-review',
                'threat-modeling',
                'defensive-configuration-review',
                'incident-response-guidance',
                'log-analysis',
                'ctf-learning',
                'authorized-test-planning',
                'remediation-planning',
                'report-management'
            ],

            storage: 'JSON',
            liveScanning: false,
            timestamp: new Date().toISOString()
        });
    }
);

/* ---------------------------------------------------------
   Run a CyberLab assessment
   --------------------------------------------------------- */

app.post(
    '/api/cyber/assess',
    auth,
    async (req, res) => {
        const body = req.body || {};

        const title = cleanTitle(
            body.title,
            'CyberLab Security Assessment'
        );

        const task = cyberlabText(
            body.task || body.assessmentType || 'code-review',
            100
        );

        const target = cyberlabText(
            body.target || 'User-submitted material',
            1000
        );

        const evidence = cyberlabRedactSecrets(
            cyberlabText(
                body.evidence ||
                body.code ||
                body.content ||
                body.logs ||
                '',
                30000
            )
        );

        if (!evidence.trim()) {
            return res.status(400).json({
                success: false,
                error:
                    'Provide code, configuration, logs, API details, or other evidence to assess.'
            });
        }

        const prompt = `
Perform a defensive CyberLab assessment.

ASSESSMENT TITLE:
${title}

ASSESSMENT TYPE:
${task}

DECLARED SCOPE:
${target}

USER-SUBMITTED EVIDENCE:
<submitted_evidence>
${evidence}
</submitted_evidence>

Instructions:

1. Review the submitted evidence only.
2. Identify genuine issues supported by the evidence.
3. Distinguish confirmed defects from possible risks.
4. Do not assume that unprovided security controls are absent.
5. Do not claim to have accessed or tested a live system.
6. Do not reproduce secrets in the report.
7. Prioritize findings by plausible impact and evidence.
8. Explain practical defensive remediation.
9. Include a verification step for each meaningful finding.
10. Return the exact JSON structure specified in your system prompt.
`.trim();

        try {
            const aiResult = await generateAIResponse(
                prompt,
                {
                    provider: body.provider || 'auto',
                    providerForKey: body.providerForKey || '',
                    apiKey: body.apiKey || '',
                    model: body.model || '',
                    fallback: body.fallback
                }
            );

            let parsed = {};

            /*
             * The existing AI wrapper normalizes the AI's
             * response. Recover the structured assessment
             * from the returned message where possible.
             */

            try {
                parsed = parseModelJson(
                    aiResult.message
                );
            } catch {
                parsed = {
                    message: aiResult.message,
                    findings: [],
                    recommendations: [],
                    nextSteps: [],
                    limitations: [
                        'The AI response could not be parsed into the expected structured report.'
                    ]
                };
            }

            const report = cyberlabNormalizeReport(
                parsed,
                {
                    title,
                    target
                },
                aiResult.provider,
                aiResult.model
            );

            const reports = await readJson(
                CYBER_REPORTS_FILE,
                []
            );

            const now = new Date().toISOString();

            const savedReport = {
                id: crypto.randomUUID(),
                ...report,

                task,

                createdAt: now,
                updatedAt: now
            };

            reports.unshift(savedReport);

            await writeJson(
                CYBER_REPORTS_FILE,
                reports.slice(0, 500)
            );

            res.json({
                success: true,
                report: savedReport
            });
        } catch (error) {
            console.error(
                '[CyberLab assessment failed]:',
                error.message
            );

            res.status(502).json({
                success: false,
                error: error.message ||
                    'CyberLab could not complete the assessment.'
            });
        }
    }
);

/* ---------------------------------------------------------
   List saved CyberLab reports
   --------------------------------------------------------- */

app.get(
    '/api/cyber/reports',
    auth,
    async (req, res) => {
        try {
            const reports = await readJson(
                CYBER_REPORTS_FILE,
                []
            );

            const limit = Math.max(
                1,
                Math.min(
                    100,
                    Number(req.query.limit) || 50
                )
            );

            res.json({
                success: true,

                reports: reports
                    .slice()
                    .sort((a, b) =>
                        new Date(b.updatedAt || b.createdAt || 0) -
                        new Date(a.updatedAt || a.createdAt || 0)
                    )
                    .slice(0, limit)
                    .map(report => ({
                        id: report.id,
                        title: report.title,
                        task: report.task,
                        createdAt: report.createdAt,
                        updatedAt: report.updatedAt,
                        riskRating: report.assessment?.riskRating ||
                            'Informational',
                        findingsCount: report.assessment?.findingsCount ||
                            cyberlabCountFindings(report.findings || [])
                    }))
            });
        } catch (error) {
            console.error(
                '[CyberLab reports list failed]:',
                error.message
            );

            res.status(500).json({
                success: false,
                error: 'Unable to retrieve CyberLab reports.'
            });
        }
    }
);

/* ---------------------------------------------------------
   Retrieve one saved CyberLab report
   --------------------------------------------------------- */

app.get(
    '/api/cyber/reports/:id',
    auth,
    async (req, res) => {
        const id = safeId(req.params.id);

        if (!id) {
            return res.status(400).json({
                success: false,
                error: 'Invalid report ID.'
            });
        }

        try {
            const reports = await readJson(
                CYBER_REPORTS_FILE,
                []
            );

            const report = reports.find(
                item => item.id === id
            );

            if (!report) {
                return res.status(404).json({
                    success: false,
                    error: 'CyberLab report not found.'
                });
            }

            res.json({
                success: true,
                report
            });
        } catch (error) {
            console.error(
                '[CyberLab report retrieval failed]:',
                error.message
            );

            res.status(500).json({
                success: false,
                error: 'Unable to retrieve the report.'
            });
        }
    }
);

/* ---------------------------------------------------------
   Delete one saved CyberLab report
   --------------------------------------------------------- */

app.delete(
    '/api/cyber/reports/:id',
    auth,
    async (req, res) => {
        const id = safeId(req.params.id);

        if (!id) {
            return res.status(400).json({
                success: false,
                error: 'Invalid report ID.'
            });
        }

        try {
            const reports = await readJson(
                CYBER_REPORTS_FILE,
                []
            );

            const remaining = reports.filter(
                report => report.id !== id
            );

            if (remaining.length === reports.length) {
                return res.status(404).json({
                    success: false,
                    error: 'CyberLab report not found.'
                });
            }

            await writeJson(
                CYBER_REPORTS_FILE,
                remaining
            );

            res.json({
                success: true,
                deleted: id
            });
        } catch (error) {
            console.error(
                '[CyberLab report deletion failed]:',
                error.message
            );

            res.status(500).json({
                success: false,
                error: 'Unable to delete the report.'
            });
        }
    }
);

/* ---------------------------------------------------------
   Export a CyberLab report as JSON
   --------------------------------------------------------- */

app.get(
    '/api/cyber/reports/:id/export',
    auth,
    async (req, res) => {
        const id = safeId(req.params.id);

        if (!id) {
            return res.status(400).json({
                success: false,
                error: 'Invalid report ID.'
            });
        }

        try {
            const reports = await readJson(
                CYBER_REPORTS_FILE,
                []
            );

            const report = reports.find(
                item => item.id === id
            );

            if (!report) {
                return res.status(404).json({
                    success: false,
                    error: 'CyberLab report not found.'
                });
            }

            const filename =
                `cyberlab-report-${id}.json`;

            res.setHeader(
                'Content-Type',
                'application/json; charset=utf-8'
            );

            res.setHeader(
                'Content-Disposition',
                `attachment; filename="${filename}"`
            );

            res.json(report);
        } catch (error) {
            console.error(
                '[CyberLab report export failed]:',
                error.message
            );

            res.status(500).json({
                success: false,
                error: 'Unable to export the report.'
            });
        }
    }
);

/* =========================================================
   END MORA CYBERLAB
   ========================================================= */
/* =========================================================
   STATIC FRONTEND
   ========================================================= */

app.use(
    '/generated-images',
    express.static(GENERATED_DIR, {
        maxAge: IS_PRODUCTION ? '7d' : 0,
        index: false
    })
);

app.use(
    express.static(PUBLIC_DIR, {
        index: 'index.html',
        extensions: ['html']
    })
);

app.get('/', (_req, res) => {
    res.sendFile(
        path.join(PUBLIC_DIR, 'index.html')
    );
});

/* =========================================================
   API 404 AND ERROR HANDLING
   ========================================================= */

app.use('/api', (_req, res) => {
    res.status(404).json({
        success: false,
        error: 'API endpoint not found.'
    });
});

app.use((error, _req, res, _next) => {
    console.error(
        '[Mora AI server error]',
        error
    );

    if (res.headersSent) {
        return;
    }

    const status = error.status === 413 ? 413 : 500;

    res.status(status).json({
        success: false,

        error: IS_PRODUCTION
            ? (
                status === 413
                    ? 'The submitted request is too large.'
                    : 'An internal server error occurred.'
            )
            : error.message ||
              'An internal server error occurred.'
    });
});

/* =========================================================
   START SERVER
   ========================================================= */

app.listen(PORT, () => {
    console.log(
        `Mora AI is running at http://localhost:${PORT}`
    );

    console.log(
        `Text providers configured: ${
            providerInfo()
                .filter(provider => provider.configured)
                .map(provider => provider.name)
                .join(', ') || 'none'
        }`
    );

    console.log(
        `Image providers configured: ${
            imageProviderInfo()
                .filter(provider => provider.configured)
                .map(provider => provider.name)
                .join(', ') || 'none'
        }`
    );
});
