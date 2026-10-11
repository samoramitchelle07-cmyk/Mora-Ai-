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

async function readJson(file, fallback) {
    try {
        return JSON.parse(await fs.readFile(file, 'utf8'));
    } catch (error) {
        if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) {
            console.error(`Unable to read ${path.basename(file)}:`, error.message);
        }
        return fallback;
    }
}

async function writeJson(file, data) {
    const temp = `${file}.${crypto.randomUUID()}.tmp`;
    await fs.writeFile(temp, JSON.stringify(data, null, 2), 'utf8');
    await fs.rename(temp, file);
}

function safeEqual(a, b) {
    const left = Buffer.from(String(a || ''));
    const right = Buffer.from(String(b || ''));

    return left.length === right.length &&
        crypto.timingSafeEqual(left, right);
}

function auth(req, res, next) {
    if (!API_TOKEN) {
        return next();
    }

    const authorization = req.headers.authorization || '';
    const bearer = authorization.startsWith('Bearer ')
        ? authorization.slice(7)
        : '';

    const supplied = bearer || req.headers['x-mora-token'] || '';

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
        ? process.env.CORS_ORIGIN.split(',').map(s => s.trim())
        : true,
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-Mora-Token']
}));

app.use(express.json({ limit: '12mb' }));
app.use(express.urlencoded({ extended: false, limit: '12mb' }));

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

const GEMINI_KEY = process.env.GEMINI_API_KEY || '';
const GROQ_KEY = process.env.GROQ_API_KEY || '';
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || '';

const DEFAULT_MODELS = {
    gemini: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    groq: process.env.GROQ_MODEL || 'openai/gpt-oss-120b',
    openrouter: process.env.OPENROUTER_MODEL || 'openai/gpt-oss-120b'
};

const IMAGE_MODELS = {
    gemini: process.env.GEMINI_IMAGE_MODEL || 'gemini-2.5-flash-image',
    openai: process.env.OPENAI_IMAGE_MODEL || 'gpt-image-1',
    openrouter: process.env.OPENROUTER_IMAGE_MODEL ||
        'google/gemini-2.5-flash-image',
    replicate: process.env.REPLICATE_MODEL ||
        'black-forest-labs/flux-schnell'
};

const SYSTEM_PROMPT = `
You are Mora AI, a helpful, capable assistant for conversation, coding,
debugging, planning, writing, and creative work.

You must return one valid JSON object, without Markdown fences, using this shape:

{
  "message": "A clear, useful response for the user",
  "steps": [],
  "files": [],
  "canvas": null,
  "execution": null,
  "terminal": ""
}

Rules:
- Always include a non-empty message when you can provide a text answer.
- Answer ordinary questions directly in the message field.
- Do not return an empty message just because no files are needed.
- For code-generation requests, put complete files in files.
- Each file should use this shape:
  {"name":"index.html","path":"index.html","content":"complete file content"}
- For a single-page HTML preview, canvas may use:
  {"type":"html","title":"Preview","html":"complete HTML document"}
- Keep steps an array of objects with a thought field and optional output/isError.
- Use empty arrays or null for fields that are not needed.
- Do not claim that code was executed or deployed unless it actually was.
- Escape JSON string contents correctly.
- Return JSON only.
`.trim();

const ai = GEMINI_KEY ? new GoogleGenAI({ apiKey: GEMINI_KEY }) : null;

function providerInfo() {
    return [
        { name: 'gemini', configured: Boolean(GEMINI_KEY) },
        { name: 'groq', configured: Boolean(GROQ_KEY) },
        { name: 'openrouter', configured: Boolean(OPENROUTER_KEY) }
    ];
}

function imageProviderInfo() {
    return [
        { name: 'gemini', configured: Boolean(GEMINI_KEY) },
        { name: 'openai', configured: Boolean(process.env.OPENAI_API_KEY) },
        { name: 'openrouter', configured: Boolean(OPENROUTER_KEY) },
        { name: 'replicate', configured: Boolean(process.env.REPLICATE_API_TOKEN) }
    ];
}

function getProviderKey(provider, personalKey = '') {
    if (personalKey) return personalKey;

    if (provider === 'gemini') return GEMINI_KEY;
    if (provider === 'groq') return GROQ_KEY;
    if (provider === 'openrouter') return OPENROUTER_KEY;

    return '';
}

function getProviderModel(provider, requestedModel = '') {
    const allowed = ['gemini', 'groq', 'openrouter'];

    if (!allowed.includes(provider)) {
        throw new Error(`Unsupported text provider: ${provider}`);
    }

    const model = String(requestedModel || '').trim();

    return model || DEFAULT_MODELS[provider];
}

function extractGeminiText(response) {
    if (typeof response?.text === 'string') {
        return response.text;
    }

    if (typeof response?.text === 'function') {
        return response.text();
    }

    const parts = response?.candidates?.[0]?.content?.parts || [];

    return parts
        .filter(part => typeof part.text === 'string')
        .map(part => part.text)
        .join('\n');
}

function parseModelJson(text) {
    const raw = String(text || '').trim();

    if (!raw) {
        throw new Error('The AI provider returned an empty response.');
    }

    const withoutFence = raw
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/i, '')
        .trim();

    try {
        return JSON.parse(withoutFence);
    } catch {
        // Some providers add explanatory text around their JSON.
    }

    const start = withoutFence.indexOf('{');
    const end = withoutFence.lastIndexOf('}');

    if (start !== -1 && end > start) {
        try {
            return JSON.parse(withoutFence.slice(start, end + 1));
        } catch {
            // Try the repair request in generateAIResponse.
        }
    }

    throw new Error('The AI provider returned invalid JSON.');
}

function normalizeResult(raw, prompt) {
    let result = raw;

    if (typeof result === 'string') {
        result = { message: result };
    }

    if (!result || typeof result !== 'object' || Array.isArray(result)) {
        result = {};
    }

    let message = typeof result.message === 'string'
        ? result.message.trim()
        : '';

    const files = Array.isArray(result.files)
        ? result.files
            .filter(file => file && typeof file === 'object')
            .map((file, index) => {
                const name = String(
                    file.name || file.path || `file-${index + 1}.txt`
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
        canvas = { type: 'html', title: 'Preview', html: canvas };
    }

    if (canvas && typeof canvas === 'object') {
        if (typeof canvas.html !== 'string' && typeof canvas.content === 'string') {
            canvas.html = canvas.content;
        }
    } else {
        canvas = null;
    }

    const steps = Array.isArray(result.steps) ? result.steps : [];
    const terminal = typeof result.terminal === 'string'
        ? result.terminal
        : '';

    if (!message) {
        if (files.length) {
            message = `I created ${files.length} file(s). Open the workspace to review them.`;
        } else if (canvas?.html) {
            message = 'Your preview is ready. Open the workspace to view it.';
        } else if (terminal) {
            message = terminal;
        } else if (steps.length) {
            message = steps
                .map(step => String(step?.thought || step?.output || ''))
                .filter(Boolean)
                .join('\n');
        } else {
            message = 'I received your request, but the AI provider returned no usable answer. Please try again.';
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

function normalizeHistoryMessages(messages) {
    if (!Array.isArray(messages)) return [];

    return messages
        .slice(-12)
        .filter(item =>
            item &&
            ['user', 'assistant'].includes(item.role) &&
            typeof item.content === 'string' &&
            item.content.trim()
        )
        .map(item => ({
            role: item.role,
            content: item.content.slice(0, 12000)
        }));
}

function makeUserContent(prompt, history) {
    const previous = normalizeHistoryMessages(history);

    const transcript = previous.length
        ? '\n\nRECENT CONVERSATION:\n' + previous.map(m =>
            `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`
        ).join('\n\n')
        : '';

    return `${transcript}\n\nCURRENT USER REQUEST:\n${prompt}`;
}

async function requestGeminiText(prompt, options = {}) {
    if (!GEMINI_KEY || !ai) {
        throw new Error('Gemini is not configured. Add GEMINI_API_KEY to your server environment.');
    }

    const model = getProviderModel('gemini', options.model);
    const contents = makeUserContent(prompt, options.messages);

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
        throw new Error('Gemini returned an empty answer. Try again or select another provider.');
    }

    return text;
}

async function requestCompatibleText(prompt, provider, options = {}) {
    const personalKey = options.apiKey || '';
    const key = getProviderKey(provider, personalKey);

    if (!key) {
        throw new Error(
            `${provider} is not configured. Add its API key in Mora AI Settings or on the server.`
        );
    }

    const model = getProviderModel(provider, options.model);
    const endpoint = provider === 'groq'
        ? 'https://api.groq.com/openai/v1/chat/completions'
        : 'https://openrouter.ai/api/v1/chat/completions';

    const history = normalizeHistoryMessages(options.messages);

    const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        ...history.map(item => ({
            role: item.role,
            content: item.content
        })),
        { role: 'user', content: prompt }
    ];

    const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`
    };

    if (provider === 'openrouter') {
        headers['HTTP-Referer'] = process.env.OPENROUTER_SITE_URL ||
            'http://localhost';
        headers['X-Title'] = process.env.OPENROUTER_APP_NAME || 'Mora AI';
    }

    const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({
            model,
            messages,
            temperature: 0.4,
            response_format: { type: 'json_object' }
        }),
        signal: AbortSignal.timeout(90000)
    });

    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
        const detail = body?.error?.message ||
            body?.message ||
            `Provider returned HTTP ${response.status}.`;

        throw new Error(`${provider}: ${detail}`);
    }

    const text = body?.choices?.[0]?.message?.content;

    if (typeof text !== 'string' || !text.trim()) {
        throw new Error(`${provider} returned an empty message.`);
    }

    return text;
}

async function requestProviderText(prompt, provider, options = {}) {
    if (provider === 'gemini') {
        return requestGeminiText(prompt, options);
    }

    if (provider === 'groq' || provider === 'openrouter') {
        return requestCompatibleText(prompt, provider, options);
    }

    throw new Error(`Unknown provider: ${provider}`);
}

function chooseProviders(body) {
    const requested = String(body.provider || 'auto').toLowerCase();
    const personalKey = String(body.apiKey || '').trim();

    if (personalKey && requested !== 'auto') {
        return [requested];
    }

    if (personalKey && requested === 'auto') {
        const keyProvider = String(body.providerForKey || '').toLowerCase();

        if (['gemini', 'groq', 'openrouter'].includes(keyProvider)) {
            return [keyProvider];
        }

        throw new Error(
            'Choose the provider associated with your personal API key in Settings.'
        );
    }

    if (requested !== 'auto') {
        if (!['gemini', 'groq', 'openrouter'].includes(requested)) {
            throw new Error(`Unsupported provider: ${requested}`);
        }

        const configured = getProviderKey(requested);

        if (!configured) {
            throw new Error(
                `${requested} has no server API key. Add a key in Settings or choose Auto.`
            );
        }

        return body.fallback === false
            ? [requested]
            : [
                requested,
                ...['gemini', 'groq', 'openrouter'].filter(p => p !== requested)
                    .filter(p => Boolean(getProviderKey(p)))
            ];
    }

    const configured = ['gemini', 'groq', 'openrouter']
        .filter(p => Boolean(getProviderKey(p)));

    if (!configured.length) {
        throw new Error(
            'No text AI provider is configured. Add a Gemini, Groq, or OpenRouter API key in Settings.'
        );
    }

    return configured;
}

async function generateAIResponse(prompt, body = {}) {
    const providers = chooseProviders(body);
    const errors = [];

    for (const provider of providers) {
        try {
            const raw = await requestProviderText(prompt, provider, {
                apiKey: body.apiKey || '',
                model: body.model || '',
                messages: body.messages || []
            });

            let parsed;

            try {
                parsed = parseModelJson(raw);
            } catch {
                // Repair malformed JSON once before moving to the next provider.
                const repairPrompt = [
                    'The following response did not contain valid JSON.',
                    'Convert it to the required JSON object.',
                    'Preserve its useful answer and do not return an empty message.',
                    '',
                    raw.slice(0, 16000)
                ].join('\n');

                const repaired = await requestProviderText(
                    repairPrompt,
                    provider,
                    {
                        apiKey: body.apiKey || '',
                        model: body.model || '',
                        messages: []
                    }
                );

                parsed = parseModelJson(repaired);
            }

            const result = normalizeResult(parsed, prompt);

            return {
                ...result,
                success: true,
                provider,
                model: body.model || DEFAULT_MODELS[provider]
            };
        } catch (error) {
            console.error(`[Mora AI] ${provider} failed:`, error.message);
            errors.push(`${provider}: ${error.message}`);

            if (body.fallback === false || body.apiKey) {
                break;
            }
        }
    }

    throw new Error(
        `Mora AI could not generate a response.\n${errors.join('\n')}`
    );
}

function safeId(value) {
    const id = String(value || '');
    return /^[a-zA-Z0-9_-]{1,100}$/.test(id) ? id : null;
}

function cleanTitle(value, fallback = 'Untitled') {
    return String(value || fallback).trim().slice(0, 120) || fallback;
}

function cleanFiles(files) {
    if (!Array.isArray(files)) return [];

    return files
        .filter(file => file && typeof file === 'object')
        .slice(0, 100)
        .map((file, index) => {
            let name = String(
                file.name || file.path || `file-${index + 1}.txt`
            ).replaceAll('\\', '/');

            name = path.posix.basename(name).replace(/[^a-zA-Z0-9._ -]/g, '_');

            if (!name || name === '.' || name === '..') {
                name = `file-${index + 1}.txt`;
            }

            return {
                name,
                path: String(file.path || name).slice(0, 300),
                content: String(file.content ?? '').slice(0, 2_000_000)
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

app.get('/api/health', async (_req, res) => {
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

/* ------------------------------ AI CHAT ------------------------------ */

app.post('/api/agent', auth, async (req, res) => {
    const body = req.body || {};
    const prompt = String(body.prompt || body.message || '').trim();

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
        const result = await generateAIResponse(prompt, body);

        const history = await readJson(HISTORY_FILE, []);

        history.unshift({
            id: crypto.randomUUID(),
            prompt,
            provider: result.provider,
            model: result.model,
            files: result.files,
            createdAt: new Date().toISOString()
        });

        await writeJson(HISTORY_FILE, history.slice(0, 500));

        res.json(result);
    } catch (error) {
        console.error('[Mora AI] Request failed:', error.message);

        res.status(502).json({
            success: false,
            error: error.message || 'The AI request failed.'
        });
    }
});

/* ------------------------------ CONNECTION TEST ------------------------------ */

app.post('/api/test', auth, async (req, res) => {
    const body = req.body || {};

    try {
        const providers = chooseProviders({
            ...body,
            fallback: false
        });

        const provider = providers[0];

        const raw = await requestProviderText(
            'Reply with a short JSON message confirming that Mora AI is connected.',
            provider,
            {
                apiKey: body.apiKey || '',
                model: body.model || '',
                messages: []
            }
        );

        const parsed = parseModelJson(raw);

        res.json({
            success: true,
            provider,
            message: parsed.message || 'Connection successful.'
        });
    } catch (error) {
        res.status(502).json({
            success: false,
            error: error.message
        });
    }
});

/* ------------------------------ CONVERSATIONS ------------------------------ */

app.get('/api/conversations', auth, async (_req, res) => {
    const conversations = await readJson(CONVERSATIONS_FILE, []);

    res.json({
        conversations: conversations
            .slice()
            .sort((a, b) =>
                new Date(b.updatedAt || b.createdAt || 0) -
                new Date(a.updatedAt || a.createdAt || 0)
            )
            .map(c => ({
                id: c.id,
                title: c.title,
                createdAt: c.createdAt,
                updatedAt: c.updatedAt,
                messages: []
            }))
    });
});

/* This GET route was missing in the previous backend. */
app.get('/api/conversations/:id', auth, async (req, res) => {
    const conversations = await readJson(CONVERSATIONS_FILE, []);
    const conversation = conversations.find(c => c.id === req.params.id);

    if (!conversation) {
        return res.status(404).json({
            success: false,
            error: 'Conversation not found.'
        });
    }

    res.json({ conversation: publicConversation(conversation) });
});

app.post('/api/conversations', auth, async (req, res) => {
    const body = req.body || {};
    const conversations = await readJson(CONVERSATIONS_FILE, []);

    const suppliedId = safeId(body.id);
    let conversation = suppliedId
        ? conversations.find(c => c.id === suppliedId)
        : null;

    const now = new Date().toISOString();

    const messages = Array.isArray(body.messages)
        ? body.messages.slice(-300).map(message => ({
            id: String(message.id || crypto.randomUUID()),
            role: message.role === 'assistant' ? 'assistant' : 'user',
            content: String(message.content || '').slice(0, 30000),
            images: Array.isArray(message.images)
                ? message.images
                    .filter(img => img && typeof img.url === 'string')
                    .slice(0, 10)
                    .map(img => ({
                        url: img.url.slice(0, 2000),
                        alt: String(img.alt || '').slice(0, 500),
                        prompt: String(img.prompt || '').slice(0, 2000),
                        name: String(img.name || '').slice(0, 200),
                        provider: String(img.provider || '').slice(0, 100),
                        model: String(img.model || '').slice(0, 200)
                    }))
                : [],
            createdAt: message.createdAt || now
        }))
        : [];

    if (conversation) {
        conversation.title = cleanTitle(body.title, conversation.title);
        conversation.messages = messages;
        conversation.updatedAt = now;
    } else {
        conversation = {
            id: suppliedId || crypto.randomUUID(),
            title: cleanTitle(body.title, 'New conversation'),
            messages,
            createdAt: now,
            updatedAt: now
        };

        conversations.unshift(conversation);
    }

    await writeJson(CONVERSATIONS_FILE, conversations.slice(0, 500));

    res.json({
        success: true,
        conversation: publicConversation(conversation)
    });
});

app.delete('/api/conversations/:id', auth, async (req, res) => {
    const conversations = await readJson(CONVERSATIONS_FILE, []);
    const remaining = conversations.filter(c => c.id !== req.params.id);

    if (remaining.length === conversations.length) {
        return res.status(404).json({
            success: false,
            error: 'Conversation not found.'
        });
    }

    await writeJson(CONVERSATIONS_FILE, remaining);

    res.json({ success: true });
});

/* ------------------------------ HISTORY ------------------------------ */

app.get('/api/history', auth, async (_req, res) => {
    const history = await readJson(HISTORY_FILE, []);
    res.json({ history });
});

app.delete('/api/history', auth, async (_req, res) => {
    await writeJson(HISTORY_FILE, []);
    res.json({ success: true });
});

/* ------------------------------ PROJECTS ------------------------------ */

app.get('/api/projects', auth, async (_req, res) => {
    const projects = await readJson(PROJECTS_FILE, []);

    res.json({
        projects: projects.map(p => ({
            ...p,
            files: Array.isArray(p.files) ? p.files : []
        }))
    });
});

app.get('/api/projects/:id', auth, async (req, res) => {
    const projects = await readJson(PROJECTS_FILE, []);
    const project = projects.find(p => p.id === req.params.id);

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
    const projects = await readJson(PROJECTS_FILE, []);
    const id = safeId(body.id);
    const now = new Date().toISOString();

    let project = id ? projects.find(p => p.id === id) : null;

    const newData = {
        title: cleanTitle(body.title, 'Mora AI Project'),
        prompt: String(body.prompt || '').slice(0, 12000),
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

    await writeJson(PROJECTS_FILE, projects.slice(0, 300));

    res.json({ success: true, project });
});

app.delete('/api/projects/:id', auth, async (req, res) => {
    const projects = await readJson(PROJECTS_FILE, []);
    const remaining = projects.filter(p => p.id !== req.params.id);

    if (remaining.length === projects.length) {
        return res.status(404).json({
            success: false,
            error: 'Project not found.'
        });
    }

    await writeJson(PROJECTS_FILE, remaining);

    res.json({ success: true });
});

async function exportProjectZip(project, res) {
    res.setHeader('Content-Type', 'application/zip');

    const filename = cleanTitle(project.title, 'mora-project')
        .replace(/[^a-zA-Z0-9_-]/g, '_');

    res.setHeader(
        'Content-Disposition',
        `attachment; filename="${filename}.zip"`
    );

    const archive = archiver('zip', { zlib: { level: 6 } });

    archive.on('error', error => {
        console.error('ZIP export error:', error.message);

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

/* Support both URL forms used by different frontend versions. */
app.get('/api/projects/:id/export', auth, async (req, res) => {
    const projects = await readJson(PROJECTS_FILE, []);
    const project = projects.find(p => p.id === req.params.id);

    if (!project) {
        return res.status(404).json({
            success: false,
            error: 'Project not found.'
        });
    }

    try {
        await exportProjectZip(project, res);
    } catch (error) {
        console.error(error);
        if (!res.headersSent) {
            res.status(500).json({
                success: false,
                error: 'ZIP export failed.'
            });
        }
    }
});

app.get('/api/projects/:id/export.zip', auth, async (req, res) => {
    const projects = await readJson(PROJECTS_FILE, []);
    const project = projects.find(p => p.id === req.params.id);

    if (!project) {
        return res.status(404).json({
            success: false,
            error: 'Project not found.'
        });
    }

    try {
        await exportProjectZip(project, res);
    } catch (error) {
        console.error(error);
        if (!res.headersSent) {
            res.status(500).json({
                success: false,
                error: 'ZIP export failed.'
            });
        }
    }
});

/* ------------------------------ IMAGE GENERATION ------------------------------ */

function getImageKey(provider, keys = {}) {
    if (keys && typeof keys[provider] === 'string' && keys[provider].trim()) {
        return keys[provider].trim();
    }

    if (provider === 'gemini') return GEMINI_KEY;
    if (provider === 'openai') return process.env.OPENAI_API_KEY || '';
    if (provider === 'openrouter') return OPENROUTER_KEY;
    if (provider === 'replicate') return process.env.REPLICATE_API_TOKEN || '';

    return '';
}

function getImageModel(provider, models = {}) {
    if (models && typeof models[provider] === 'string' && models[provider].trim()) {
        return models[provider].trim();
    }

    return IMAGE_MODELS[provider];
}

function dataUrlToInlineData(inputImage) {
    if (!inputImage || typeof inputImage.data !== 'string') return null;

    const mimeType = String(inputImage.mimeType || '');

    if (!['image/png', 'image/jpeg', 'image/webp'].includes(mimeType)) {
        throw new Error('Attached images must be PNG, JPEG, or WebP.');
    }

    const data = inputImage.data.replace(/\s/g, '');

    if (!data || data.length > 9_000_000 || !/^[A-Za-z0-9+/=]+$/.test(data)) {
        throw new Error('The attached image data is invalid or too large.');
    }

    return {
        inlineData: {
            mimeType,
            data
        }
    };
}

async function generateGeminiImage(prompt, key, model, inputImage) {
    const client = new GoogleGenAI({ apiKey: key });

    const contents = [{ text: prompt }];

    const imagePart = dataUrlToInlineData(inputImage);
    if (imagePart) contents.push(imagePart);

    const result = await client.models.generateContent({
        model,
        contents,
        config: {
            responseModalities: ['TEXT', 'IMAGE']
        }
    });

    const parts = result?.candidates?.[0]?.content?.parts || [];

    for (const part of parts) {
        if (part.inlineData?.data) {
            return {
                base64: part.inlineData.data,
                mime: part.inlineData.mimeType || 'image/png'
            };
        }
    }

    const explanation = parts
        .map(p => p.text || '')
        .filter(Boolean)
        .join('\n');

    throw new Error(
        explanation || 'Gemini did not return image data. Check the selected image model and API access.'
    );
}

async function generateOpenAIImage(prompt, key, model) {
    const response = await fetch('https://api.openai.com/v1/images/generations', {
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
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(data?.error?.message || `OpenAI returned HTTP ${response.status}.`);
    }

    const item = data?.data?.[0];

    if (item?.b64_json) {
        return { base64: item.b64_json, mime: 'image/png' };
    }

    if (item?.url) {
        const imageResponse = await fetch(item.url, {
            signal: AbortSignal.timeout(60000)
        });

        if (!imageResponse.ok) {
            throw new Error('Could not download the generated OpenAI image.');
        }

        const buffer = Buffer.from(await imageResponse.arrayBuffer());

        return {
            base64: buffer.toString('base64'),
            mime: imageResponse.headers.get('content-type') || 'image/png'
        };
    }

    throw new Error('OpenAI returned no image data.');
}

async function generateOpenRouterImage(prompt, key, model, inputImage) {
    const userContent = [{ type: 'text', text: prompt }];

    if (inputImage?.data) {
        const inline = dataUrlToInlineData(inputImage).inlineData;

        userContent.push({
            type: 'image_url',
            image_url: {
                url: `data:${inline.mimeType};base64,${inline.data}`
            }
        });
    }

    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${key}`,
            'Content-Type': 'application/json',
            'X-Title': 'Mora AI'
        },
        body: JSON.stringify({
            model,
            messages: [{ role: 'user', content: userContent }],
            modalities: ['image', 'text']
        }),
        signal: AbortSignal.timeout(120000)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        throw new Error(
            data?.error?.message || `OpenRouter returned HTTP ${response.status}.`
        );
    }

    const message = data?.choices?.[0]?.message;
    const image = message?.images?.[0];

    const imageUrl = image?.image_url?.url ||
        image?.url ||
        (typeof image === 'string' ? image : '');

    if (imageUrl.startsWith('data:')) {
        const match = imageUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);

        if (match) {
            return { base64: match[2], mime: match[1] };
        }
    }

    if (imageUrl.startsWith('https://')) {
        const r = await fetch(imageUrl, {
            signal: AbortSignal.timeout(60000)
        });

        if (!r.ok) throw new Error('Could not download the OpenRouter image.');

        const buffer = Buffer.from(await r.arrayBuffer());

        return {
            base64: buffer.toString('base64'),
            mime: r.headers.get('content-type') || 'image/png'
        };
    }

    throw new Error(
        message?.content ||
        'OpenRouter did not return an image. Check whether the selected model supports image generation.'
    );
}

async function generateReplicateImage(prompt, key, model) {
    const endpoint = 'https://api.replicate.com/v1/models/' +
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
            data?.detail || data?.error || `Replicate returned HTTP ${response.status}.`
        );
    }

    if (data.status === 'starting' || data.status === 'processing') {
        const pollUrl = data.urls?.get;

        if (!pollUrl) {
            throw new Error('Replicate is still processing the image. Please try again.');
        }

        for (let i = 0; i < 20; i++) {
            await new Promise(resolve => setTimeout(resolve, 3000));

            const poll = await fetch(pollUrl, {
                headers: { Authorization: `Bearer ${key}` },
                signal: AbortSignal.timeout(30000)
            });

            data = await poll.json().catch(() => ({}));

            if (data.status === 'succeeded') break;

            if (data.status === 'failed' || data.status === 'canceled') {
                throw new Error(data.error || 'Replicate image generation failed.');
            }
        }
    }

    if (data.status !== 'succeeded') {
        throw new Error('Replicate did not finish generating the image in time.');
    }

    const output = Array.isArray(data.output) ? data.output[0] : data.output;

    if (typeof output !== 'string' || !output.startsWith('https://')) {
        throw new Error('Replicate returned no downloadable image.');
    }

    const imageResponse = await fetch(output, {
        signal: AbortSignal.timeout(60000)
    });

    if (!imageResponse.ok) {
        throw new Error('Could not download the Replicate image.');
    }

    const buffer = Buffer.from(await imageResponse.arrayBuffer());

    return {
        base64: buffer.toString('base64'),
        mime: imageResponse.headers.get('content-type') || 'image/png'
    };
}

app.post('/api/images/generate', auth, imageLimiter, async (req, res) => {
    const body = req.body || {};
    const prompt = String(body.prompt || '').trim();

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

    const requested = String(body.provider || 'auto').toLowerCase();

    const allowed = ['gemini', 'openai', 'openrouter', 'replicate'];

    if (requested !== 'auto' && !allowed.includes(requested)) {
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
        const key = getImageKey(provider, body.imageKeys || {});

        if (!key) {
            failures.push(`${provider}: no API key configured`);
            continue;
        }

        const model = getImageModel(provider, body.models || {});

        try {
            let image;

            if (provider === 'gemini') {
                image = await generateGeminiImage(prompt, key, model, body.inputImage);
            } else if (provider === 'openai') {
                if (body.inputImage) {
                    throw new Error(
                        'The OpenAI Images endpoint used here supports generation only. Select Gemini or a compatible image-editing model to edit an uploaded image.'
                    );
                }

                image = await generateOpenAIImage(prompt, key, model);
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
                        'This Replicate endpoint is configured for text-to-image generation, not image editing.'
                    );
                }

                image = await generateReplicateImage(prompt, key, model);
            }

            const buffer = Buffer.from(image.base64, 'base64');

            if (!buffer.length || buffer.length > 25 * 1024 * 1024) {
                throw new Error('The returned image is empty or too large.');
            }

            const extension = image.mime === 'image/jpeg'
                ? 'jpg'
                : image.mime === 'image/webp'
                    ? 'webp'
                    : 'png';

            const filename = `mora-${Date.now()}-${crypto.randomBytes(4).toString('hex')}.${extension}`;
            const filePath = path.join(GENERATED_DIR, filename);

            await fs.writeFile(filePath, buffer);

            const url = `/generated-images/${filename}`;

            return res.json({
                success: true,
                image: {
                    url,
                    name: filename,
                    prompt,
                    provider,
                    model,
                    mime: image.mime,
                    bytes: buffer.length
                },
                url,
                filename,
                provider,
                model
            });
        } catch (error) {
            console.error(`[Mora AI image] ${provider} failed:`, error.message);
            failures.push(`${provider}: ${error.message}`);

            if (requested !== 'auto') break;
        }
    }

    res.status(502).json({
        success: false,
        error: `Image generation failed.\n${failures.join('\n')}`
    });
});

/* ------------------------------ HTML TEST ------------------------------ */

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

    const title = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]
        ?.replace(/<[^>]*>/g, '')
        .trim() || '';

    const hasViewport = /<meta[^>]+name\s*=\s*["']viewport["']/i.test(html);
    const elementCount = (html.match(/<[a-z][\w:-]*(?:\s[^<>]*?)?>/gi) || []).length;

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
        note: 'This endpoint checks the HTML source only. It does not execute JavaScript in a real browser.'
    });
});

/* ------------------------------ OPTIONAL PYTHON EXECUTION ------------------------------ */

app.post('/api/execute/python', auth, async (req, res) => {
    const code = String(req.body?.code || '').trim();

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
        const { Sandbox } = await import('@e2b/code-interpreter');

        sandbox = await Sandbox.create({
            apiKey: process.env.E2B_API_KEY
        });

        const result = await sandbox.runCode(code);

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
            error: error.message || 'Python execution failed.'
        });
    } finally {
        if (sandbox && typeof sandbox.close === 'function') {
            try {
                await sandbox.close();
            } catch {
                // Ignore sandbox cleanup errors.
            }
        }
    }
});

/* ------------------------------ STATIC FRONTEND ------------------------------ */

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
    res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.use('/api', (_req, res) => {
    res.status(404).json({
        success: false,
        error: 'API endpoint not found.'
    });
});

app.use((error, _req, res, _next) => {
    console.error('[Mora AI server error]', error);

    if (res.headersSent) return;

    res.status(500).json({
        success: false,
        error: IS_PRODUCTION
            ? 'An internal server error occurred.'
            : error.message || 'An internal server error occurred.'
    });
});

app.listen(PORT, () => {
    console.log(`Mora AI is running at http://localhost:${PORT}`);
    console.log(`Text providers configured: ${
        providerInfo()
            .filter(p => p.configured)
            .map(p => p.name)
            .join(', ') || 'none'
    }`);
    console.log(`Image providers configured: ${
        imageProviderInfo()
            .filter(p => p.configured)
            .map(p => p.name)
            .join(', ') || 'none'
    }`);
});
