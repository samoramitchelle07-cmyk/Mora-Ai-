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

// ============================================================
// MORA AI — SERVER CONFIGURATION
// ============================================================

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);

const MODEL = process.env.GEMINI_MODEL || 'gemini-3.8-flash';

const DATA_DIR = path.join(ROOT, 'data');
const PROJECTS_FILE = path.join(DATA_DIR, 'projects.json');
const HISTORY_FILE = path.join(DATA_DIR, 'history.json');

const MAX_PROMPT = 12000;
const MAX_FILE_CHARS = 180000;
const MAX_FILES = 25;
const MAX_JSON_BODY = '1mb';

const IS_PRODUCTION = process.env.NODE_ENV === 'production';
const TOKEN = (process.env.MORA_API_TOKEN || '').trim();

// Require authentication in production.
if (IS_PRODUCTION && !TOKEN) {
    console.error(
        'SECURITY ERROR: Set MORA_API_TOKEN before exposing Mora AI in production.'
    );

    process.exit(1);
}

// Initialize Gemini.
const ai = process.env.GEMINI_API_KEY
    ? new GoogleGenAI({
        apiKey: process.env.GEMINI_API_KEY
    })
    : null;

// ============================================================
// EXPRESS APPLICATION
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
            : (IS_PRODUCTION ? false : true)
    })
);

app.use(
    express.json({
        limit: MAX_JSON_BODY
    })
);

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
// AUTHENTIlegacyHeaders=========================================================

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

    const authorization = req.get('authorization') || '';

    const provided =
        authorization.replace(/^Bearer\s+/i, '') ||
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

// ============================================================
// STATIC FILES
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

async function readJson(file, fallback) {
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

    const temporaryFile =
        `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;

    await fs.writeFile(
        temporaryFile,
        JSON.stringify(value, null, 2),
        {
            encoding: 'utf8',
            mode: 0o600
        }
    );

    await fs.rename(temporaryFile, file);
}

// ============================================================
// FILE NORMALIZATION
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

    if (!safe || safe === '.' || safe === '..') {
        return `file-${index + 1}.txt`;
    }

    return safe;
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
// IMPROVED GEMINI JSON PARSER
// ============================================================

function parseJson(text) {
    if (typeof text !== 'string' || !text.trim()) {
        throw new Error('Gemini returned an empty JSON response.');
    }

    let cleaned = text.trim();

    // Remove Markdown fences if Gemini includes them.
    cleaned = cleaned
        .replace(/^```(?:json)?\s*/i, '')
        .replace(/\s*```$/, '')
        .trim();

    // First try parsing the complete response.
    try {
        const parsed = JSON.parse(cleaned);

        if (
            parsed === null ||
            typeof parsed !== 'object' ||
            Array.isArray(parsed)
        ) {
            throw new Error('Expected a JSON object.');
        }

        return parsed;
    } catch (initialError) {
        // Continue to attempt extracting a JSON object.
    }

    // Find the first opening brace.
    const start = cleaned.indexOf('{');

    if (start === -1) {
        throw new Error(
            'Gemini response could not be parsed: no JSON object found.'
        );
    }

    // Find the matching closing brace while respecting strings.
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;

    for (let i = start; i < cleaned.length; i++) {
        const char = cleaned[i];

        if (inString) {
            if (escaped) {
                escaped = false;
                continue;
            }

            if (char === '\\') {
                escaped = true;
                continue;
            }

            if (char === '"') {
                inString = false;
            }

            continue;
        }

        if (char === '"') {
            inString = true;
            continue;
        }

        if (char === '{') {
            depth++;
        } else if (char === '}') {
            depth--;

            if (depth === 0) {
                end = i;
                break;
            }
        }
    }

    if (end === -1) {
        throw new Error(
            'Incomplete JSON segment: Gemini did not finish the JSON object.'
        );
    }

    const candidate = cleaned.slice(start, end + 1);

    try {
        const parsed = JSON.parse(candidate);

        if (
            parsed === null ||
            typeof parsed !== 'object' ||
            Array.isArray(parsed)
        ) {
            throw new Error('Expected a JSON object.');
        }

        return parsed;
    } catch (error) {
        throw new Error(
            `Gemini JSON is invalid: ${String(error.message).slice(0, 300)}`
        );
    }
}

// ============================================================
// NORMALIZE GEMINI RESULTS
// ============================================================

function normalizeResult(data, prompt = '') {
    if (
        !data ||
        typeof data !== 'object' ||
        Array.isArray(data)
    ) {
        throw new Error('Gemini returned an invalid result object.');
    }

    const message =
        typeof data.message === 'string'
            ? data.message.slice(0, 30000)
            : '';

    const rawSteps = Array.isArray(data.steps)
        ? data.steps.slice(0, 50)
        : [];

    const steps = rawSteps.map(step => ({
        thought: String(
            step?.thought || ''
        ).slice(0, 1000),

        command: String(
            step?.command || 'generate'
        ).slice(0, 100),

        output: String(
            step?.output || ''
        ).slice(0, 3000),

        isError: Boolean(step?.isError)
    }));

    const files = normalizeFiles(data.files);

    // Validate canvas data.
    let canvas = null;

    if (
        data.canvas &&
        typeof data.canvas === 'object' &&
        !Array.isArray(data.canvas)
    ) {
        const html = typeof data.canvas.html === 'string'
            ? data.canvas.html.slice(0, MAX_FILE_CHARS)
            : '';

        canvas = {
            type: String(
                data.canvas.type || 'html'
            ).slice(0, 40),

            title: String(
                data.canvas.title || 'Mora AI Preview'
            ).slice(0, 200),

            html
        };
    }

    // Validate execution instructions.
    const rawExecution =
        data.execution &&
        typeof data.execution === 'object'
            ? data.execution
            : {};

    const language = String(
        rawExecution.language || 'python'
    ).toLowerCase();

    const code = typeof rawExecution.code === 'string'
        ? rawExecution.code.slice(0, 12000)
        : '';

    const execution = {
        requested:
            rawExecution.requested === true &&
            language === 'python' &&
            code.trim().length > 0,

        language,

        code
    };

    const terminal = typeof data.terminal === 'string'
        ? data.terminal.slice(0, 10000)
        : '';

    return {
        message: message || (
            files.length
                ? 'Mora AI generated your project files.'
                : 'Mora AI processed your request.'
        ),

        steps,

        files,

        canvas,

        execution,

        terminal,

        status: 'completed',

        prompt: String(prompt || '').slice(0, MAX_PROMPT)
    };
}

// ============================================================
// MORA AI SYSTEM INSTRUCTIONS
// ============================================================

const SYSTEM_PROMPT = `
You are Mora AI, a practical personal AI agent for coding,
debugging, website development, research, learning, and general
assistance.

Return ONLY one complete, valid JSON object with this shape:

{
  "message": "A clear answer for the user",
  "steps": [
    {
      "thought": "Brief user-facing status",
      "command": "generate",
      "output": "Short result",
      "isError": false
    }
  ],
  "files": [
    {
      "name": "index.html",
      "path": "index.html",
      "language": "html",
      "content": "Complete file contents"
    }
  ],
  "canvas": {
    "type": "html",
    "title": "Preview title",
    "html": "Complete standalone HTML"
  },
  "execution": {
    "requested": false,
    "language": "python",
    "code": ""
  },
  "terminal": ""
}

Rules:

1. Return valid JSON only. Do not wrap the response in Markdown fences.

2. Escape quotation marks, backslashes, and newline characters
   correctly inside JSON strings.

3. Never reveal private chain-of-thought reasoning. Provide only
   brief summaries of actions and conclusions.

4. For ordinary questions, answer in message. Use an empty files
   array, a null canvas, and execution.requested=false.

5. For website and interface requests, generate a complete,
   standalone HTML preview in canvas.html.

6. For website requests, normally include a matching index.html
   in files. Include inline CSS and JavaScript when practical so
   the preview works independently.

7. For larger coding projects, provide complete files rather than
   disconnected snippets. Keep file paths safe and relative.

8. For Python execution requests, use execution.requested=true
   only when the user explicitly asks to run or test Python or
   needs a Python calculation.

9. Never request shell execution, JavaScript execution, destructive
   operations, credential theft, unauthorized access, or code that
   bypasses security.

10. Never claim code was tested unless it actually ran.

11. Keep each file below 180,000 characters and keep the total
    response reasonably sized.

12. The steps array should contain concise progress summaries.

For website-generation requests, return complete, functional code without unnecessary explanations or duplication. Keep HTML, CSS, and JavaScript concise. Preserve all requested features. If the output approaches the token limit, simplify styling and implementation before sacrificing required functionality. Always return one complete, valid JSON object matching the required response structure.
`;

// ============================================================
// PARTIAL MESSAGE EXTRACTION FOR STREAMING
// ============================================================

function extractPartialMessage(raw) {
    const match = /"message"\s*:\s*"/.exec(raw);

    if (!match) {
        return '';
    }

    let output = '';
    let escaped = false;

    const start = match.index + match[0].length;

    for (let index = start; index < raw.length; index++) {
        const char = raw[index];

        if (escaped) {
            escaped = false;

            if (char === 'n') {
                output += '\n';
            } else if (char === 'r') {
                output += '\r';
            } else if (char === 't') {
                output += '\t';
            } else if (char === 'b') {
                output += '\b';
            } else if (char === 'f') {
                output += '\f';
            } else if (char === 'u') {
                const hex = raw.slice(
                    index + 1,
                    index + 5
                );

                if (/^[0-9a-f]{4}$/i.test(hex)) {
                    output += String.fromCharCode(
                        parseInt(hex, 16)
                    );

                    index += 4;
                } else {
                    break;
                }
            } else {
                output += char;
            }

            continue;
        }

        if (char === '\\') {
            escaped = true;
            continue;
        }

        if (char === '"') {
            break;
        }

        output += char;
    }

    return output;
}

// ============================================================
// GEMINI GENERATION WITH ONE RECOVERY ATTEMPT
// ============================================================
function getGeminiErrorDetails(error) {
    const message = String(error?.message || error || '');

    const status =
        Number(error?.status) ||
        Number(error?.code) ||
        Number(
            message.match(/\b(408|429|500|502|503|504)\b/)?.[0]
        ) ||
        0;

    return { message, status };
}

function isTransientGeminiError(error) {
    const { message, status } = getGeminiErrorDetails(error);

    return (
        [408, 429, 500, 502, 503, 504].includes(status) ||
        /UNAVAILABLE|RESOURCE_EXHAUSTED|INTERNAL|ETIMEDOUT|ECONNRESET|ECONNREFUSED|ENETUNREACH|EAI_AGAIN/i.test(message)
    );
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function generateWithRetry(generate, maxAttempts = 4) {
    let lastError;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await generate();
        } catch (error) {
            lastError = error;

            if (
                !isTransientGeminiError(error) ||
                attempt === maxAttempts
            ) {
                throw error;
            }

            const delay =
                Math.min(2000 * (2 ** (attempt - 1)), 12000) +
                Math.floor(Math.random() * 500);

            console.warn(
                `Gemini temporary failure. ` +
                `Attempt ${attempt}/${maxAttempts}. ` +
                `Retrying in ${delay}ms.`
            );

            await sleep(delay);
        }
    }

    throw lastError;
}
async function askGemini(
    prompt,
    repairContext = '',
    onChunk = null
) {
    if (!ai) {
        throw new Error(
            'GEMINI_API_KEY is missing. Configure it in your environment.'
        );
    }

    function buildContents(correction = '') {
        return [
            SYSTEM_PROMPT,
            '',
            'USER REQUEST:',
            prompt,

            repairContext
                ? `\n\nREPAIR CONTEXT:\n${repairContext}`
                : '',

            correction
                ? `\n\nRESPONSE CORRECTION:\n${correction}`
                : ''
        ].join('\n');
    }

    async function generate({
        stream = false,
        correction = ''
    } = {}) {
        const params = {
            model: MODEL,
            contents: buildContents(correction),
            config: {
                maxOutputTokens: 32768,
                responseMimeType: 'application/json'
            }
        };

        let output = '';
        let finishReason = '';

        if (stream) {
            const responseStream =
                await ai.models.generateContentStream(params);

            for await (const chunk of responseStream) {
                const part = chunk.text || '';
                output += part;

                if (part && onChunk) {
                    onChunk({
                        chunkLength: part.length,
                        accumulated: output.length,
                        raw: output,
                        partialMessage:
                            extractPartialMessage(output)
                    });
                }

                const candidate = chunk.candidates?.[0];

                if (candidate?.finishReason) {
                    finishReason = candidate.finishReason;
                }
            }
        } else {
            const response =
                await ai.models.generateContent(params);

            output = response.text || '';

            finishReason =
                response.candidates?.[0]?.finishReason || '';
        }

        return { output, finishReason };
    }

    function validateResponse(response) {
        if (!response.output.trim()) {
            throw new Error(
                'Gemini returned an empty response.'
            );
        }

        if (
            String(response.finishReason).toUpperCase() ===
            'MAX_TOKENS'
        ) {
            throw new Error(
                'Gemini reached its output token limit.'
            );
        }

        return normalizeResult(
            parseJson(response.output),
            prompt
        );
    }

    let firstError;

    // First attempt: retry temporary Gemini service failures.
    try {
        const first = await generateWithRetry(
            () => generate({
                stream: Boolean(onChunk)
            }),
            4
        );

        return validateResponse(first);

    } catch (error) {
        firstError = error;

        console.error(
            'Gemini generation attempt failed:',
            String(error.message || error).slice(0, 500)
        );
    }

    // Recovery attempt: request corrected JSON.
    try {
        const correction = [
            'Return one complete, valid JSON object.',
            'Follow the JSON structure required by the system instructions.',
            'Do not use Markdown fences.',
            'Do not include explanatory text outside the JSON.'
        ].join(' ');

        const second = await generateWithRetry(
            () => generate({
                stream: false,
                correction
            }),
            3
        );

        const result = validateResponse(second);

        result.steps.unshift({
            thought: 'Recovered after an unsuccessful initial response.',
            command: 'repair',
            output: 'The response was regenerated and validated.',
            isError: false
        });

        return result;

    } catch (retryError) {
        console.error(
            'Gemini retry failed:',
            String(retryError.message || retryError).slice(0, 500)
        );

        const details = getGeminiErrorDetails(retryError);
        const original = getGeminiErrorDetails(firstError);

        const finalError = new Error(
            'Mora AI could not complete the request. ' +
            `Initial error: ${original.message.slice(0, 200)}. ` +
            `Recovery error: ${details.message.slice(0, 200)}`
        );

        finalError.status =
            details.status || original.status || 502;

        throw finalError;
    }
}

// ============================================================
// PYTHON SAFETY CHECKS
// ============================================================

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

    for (const [pattern, message] of checks) {
        if (pattern.test(code)) {
            return message;
        }
    }

    return '';
}

// ============================================================
// E2B PYTHON EXECUTION
// ============================================================

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
            output:
                `Execution blocked by Mora AI safety filter: ${reason}.`
        };
    }

    let sandbox;

    try {
        sandbox = await Sandbox.create({
            timeoutMs: 60000
        });

        const result = await sandbox.runCode(
            code,
            {
                timeoutMs: 20000
            }
        );

        const stdout =
            Array.isArray(result.logs?.stdout)
                ? result.logs.stdout.join('\n')
                : '';

        const stderr =
            Array.isArray(result.logs?.stderr)
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
    } catch (error) {
        return {
            ok: false,

            output:
                `E2B execution failed: ${
                    String(error.message || error).slice(0, 1200)
                }`
        };
    } finally {
        if (sandbox?.kill) {
            try {
                await sandbox.kill();
            } catch {
                // Ignore sandbox cleanup errors.
            }
        }
    }
}

// ============================================================
// HISTORY
// ============================================================

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

            files: result.files.map(
                file => file.name
            ),

            status: result.status
        });

        await writeJson(
            HISTORY_FILE,
            history.slice(0, 100)
        );
    } catch (error) {
        console.error(
            'History save warning:',
            error.message
        );
    }
}

// ============================================================
// NON-STREAMING AGENT
// ============================================================

async function runAgent(prompt, notify = () => {}) {
    const result = await askGemini(
        prompt,
        '',
        null
    );

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
                    message:
                        'Analyzing the error and generating one repair attempt…'
                });

                try {
                    const repair = await askGemini(
                        prompt,

                        `Python execution failed.
Error/output:
${run.output}

Code that failed:
${result.execution.code}

Return a corrected solution. If appropriate, include corrected Python in execution.code. Do not repeat unsafe behavior.`
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
                } catch (error) {
                    result.steps.push({
                        thought: 'Automatic repair was unavailable.',
                        command: 'auto-repair',

                        output: String(
                            error.message || error
                        ).slice(0, 1000),

                        isError: true
                    });
                }
            }
        }
    }

    result.status = 'completed';

    await saveHistory(
        prompt,
        result
    );

    return result;
}

// ============================================================
// HEALTH CHECK
// ============================================================

app.get('/api/health', (_req, res) => {
    res.json({
        success: true,
        status: 'online',
        service: 'Mora AI',
        model: MODEL,

        geminiConfigured:
            Boolean(process.env.GEMINI_API_KEY),

        e2bConfigured:
            Boolean(process.env.E2B_API_KEY),

        authRequired:
            Boolean(TOKEN),

        projectStorage:
            'local-json-single-user',

        browserTesting:
            'optional-playwright',

        timestamp:
            new Date().toISOString()
    });
});

// ============================================================
// REQUEST VALIDATION
// ============================================================

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
            error:
                `Prompt exceeds ${MAX_PROMPT} characters.`
        });

        return null;
    }

    if (!ai) {
        res.status(503).json({
            success: false,
            error:
                'GEMINI_API_KEY is missing from the server environment.'
        });

        return null;
    }

    return prompt;
}

// ============================================================
// SERVER-SENT EVENTS
// ============================================================

function sendSse(res, event, data) {
    res.write(
        `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
    );
}

// ============================================================
// STREAMING AGENT ENDPOINT
// ============================================================

app.post('/api/agent/stream', async (req, res) => {
    const prompt = validatePrompt(req, res);

    if (!prompt) {
        return;
    }

    res.status(200).set({
        'Content-Type': 'text/event-stream; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no'
    });

    res.flushHeaders?.();

    let closed = false;

    res.on('close', () => {
        if (!res.writableEnded) {
            closed = true;
        }
    });

    try {
        // ----------------------------------------------------
        // STEP 1: Notify the client that planning has started
        // ----------------------------------------------------

        sendSse(res, 'progress', {
            stage: 'planning',
            message: 'Mora AI is analyzing your request…'
        });

        // ----------------------------------------------------
        // STEP 2: Generate the response with Gemini
        // ----------------------------------------------------

        let lastPartial = '';

        const result = await askGemini(
            prompt,
            '',
            info => {
                if (closed) {
                    return;
                }

                sendSse(res, 'progress', {
                    stage: 'generating',
                    message: 'Generating response…'
                });

                // Send only newly generated text to avoid
                // repeatedly transmitting the same content.
                if (
                    info.partialMessage &&
                    info.partialMessage.length > lastPartial.length
                ) {
                    const newText = info.partialMessage.slice(
                        lastPartial.length
                    );

                    lastPartial = info.partialMessage;

                    sendSse(res, 'delta', {
                        text: newText
                    });
                }
            }
        );

        // ----------------------------------------------------
        // STEP 3: Stop if the client disconnected
        // ----------------------------------------------------

        if (closed) {
            return;
        }

        sendSse(res, 'progress', {
            stage: 'generated',
            message:
                'Response generated. Preparing tests and files…'
        });

        // ----------------------------------------------------
        // STEP 4: Execute Python when requested
        // ----------------------------------------------------

        if (result.execution.requested) {
            if (result.execution.language !== 'python') {
                result.terminal =
                    'Execution skipped: only Python is enabled.';
            } else {
                sendSse(res, 'progress', {
                    stage: 'testing',
                    message:
                        'Running Python in isolated E2B sandbox…'
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

                // --------------------------------------------
                // STEP 5: Attempt one automatic repair
                // --------------------------------------------

                if (
                    !run.ok &&
                    !run.blocked &&
                    !/not configured/i.test(run.output)
                ) {
                    sendSse(res, 'progress', {
                        stage: 'repairing',
                        message:
                            'Analyzing error and trying one automatic repair…'
                    });

                    try {
                        const repair = await askGemini(
                            prompt,
                            `Python execution failed.
Error/output:
${run.output}

Original code:
${result.execution.code}

Provide a corrected solution. If appropriate, include corrected Python in execution.code.`
                        );

                        if (
                            Array.isArray(repair.files) &&
                            repair.files.length
                        ) {
                            result.files = repair.files;
                        }

                        if (repair.canvas) {
                            result.canvas = repair.canvas;
                        }

                        result.message +=
                            `\n\nRepair attempt: ${repair.message}`;

                        // Run the repaired code only when Gemini
                        // returns Python execution instructions.
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
                    } catch (error) {
                        console.error(
                            'Automatic repair failed:',
                            error
                        );

                        result.steps.push({
                            thought: 'Automatic repair failed.',

                            command: 'auto-repair',

                            output: String(
                                error.message || error
                            ).slice(0, 1000),

                            isError: true
                        });
                    }
                }
            }
        }

        // ----------------------------------------------------
        // STEP 6: Save the completed result
        // ----------------------------------------------------

        result.status = 'completed';

        await saveHistory(
            prompt,
            result
        );

        // ----------------------------------------------------
        // STEP 7: Send the final result to the client
        // ----------------------------------------------------

        if (!closed) {
            sendSse(res, 'result', result);

            sendSse(res, 'done', {
                ok: true
            });

            res.end();
        }
    } catch (error) {
        console.error(
            'Agent stream error:',
            error
        );

        if (!closed) {
            sendSse(res, 'error', {
                error: friendlyError(error)
            });

            res.end();
        }
    }
});

// ============================================================
// NON-STREAMING AGENT ENDPOINT
// ============================================================

app.post('/api/agent', async (req, res) => {
    const prompt = validatePrompt(req, res);

    if (!prompt) {
        return;
    }

    try {
        const result = await runAgent(prompt);

        return res.json({
            success: true,
            result
        });
    } catch (error) {
        console.error(
            'Agent request failed:',
            error
        );

        return res.status(errorStatus(error)).json({
            success: false,
            error: friendlyError(error)
        });
    }
});

// ============================================================
// FRIENDLY ERROR HANDLING
// ============================================================

function friendlyError(error) {
    const message = String(
        error?.message || error || 'Unknown error'
    );

    if (/API key|unauthorized|permission/i.test(message)) {
        return (
            'Gemini authentication failed. Check GEMINI_API_KEY, ' +
            'model access, and API permissions. Details: ' +
            message.slice(0, 500)
        );
    }

    if (
        /quota|rate limit|resource_exhausted/i.test(message)
    ) {
        return (
            'Gemini quota or rate limit reached. Wait before retrying. ' +
            'Details: ' +
            message.slice(0, 500)
        );
    }

    if (
        /incomplete JSON|output token limit|could not be parsed/i.test(
            message
        )
    ) {
        return (
            'Gemini did not return a complete, valid response. ' +
            'Try a shorter prompt or divide the project into smaller tasks. ' +
            'Details: ' +
            message.slice(0, 500)
        );
    }

    return (
        `Mora AI request failed: ${message.slice(0, 900)}`
    );
}

function errorStatus(error) {
    const message = String(
        error?.message || error
    );

    if (
        /quota|rate limit|resource_exhausted/i.test(message)
    ) {
        return 429;
    }

    return 502;
}

// ============================================================
// HTML TESTING ENDPOINT
// ============================================================

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

    if (html.length > MAX_FILE_CHARS) {
        return res.status(413).json({
            success: false,
            error:
                'HTML exceeds the 180,000 character testing limit.'
        });
    }

    let chromium;

    try {
        ({ chromium } = await import('playwright'));
    } catch {
        return res.status(503).json({
            success: false,

            error:
                'Browser testing is not installed. On supported Linux, Windows, or macOS, run npm install and npx playwright install chromium. Some Android Termux environments cannot run Playwright browsers.'
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

        // Block external requests during the smoke test.
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

        page.on('console', message => {
            if (message.type() === 'error') {
                consoleErrors.push(
                    message.text().slice(0, 500)
                );
            }
        });

        page.on('pageerror', error => {
            pageErrors.push(
                String(
                    error.message || error
                ).slice(0, 500)
            );
        });

        await page.setContent(html, {
            waitUntil: 'domcontentloaded',
            timeout: 10000
        });

        await page.waitForTimeout(500);

        const details = await page.evaluate(() => ({
            title:
                document.title || '(no title)',

            bodyTextLength:
                (document.body?.innerText || '').trim().length,

            elementCount:
                document.querySelectorAll('*').length,

            hasViewport:
                Boolean(
                    document.querySelector(
                        'meta[name="viewport"]'
                    )
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

            consoleErrors:
                consoleErrors.slice(0, 20),

            pageErrors:
                pageErrors.slice(0, 20),

            note:
                'External network requests were blocked during this test. This is a smoke test, not a complete security audit.'
        });

        await context.close();
    } catch (error) {
        if (!res.headersSent) {
            res.status(500).json({
                success: false,

                error:
                    `Browser test failed: ${
                        String(
                            error.message || error
                        ).slice(0, 700)
                    }`
            });
        }
    } finally {
        if (browser) {
            try {
                await browser.close();
            } catch {
                // Ignore browser cleanup errors.
            }
        }
    }
});

// ============================================================
// HISTORY ENDPOINT
// ============================================================

app.get('/api/history', async (_req, res) => {
    try {
        res.json({
            success: true,

            history: await readJson(
                HISTORY_FILE,
                []
            )
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            error: 'Could not read request history.'
        });
    }
});

// ============================================================
// PROJECT LIST
// ============================================================

app.get('/api/projects', async (_req, res) => {
    try {
        const projects = await readJson(
            PROJECTS_FILE,
            []
        );

        res.json({
            success: true,

            projects: projects.map(
                ({
                    id,
                    title,
                    updatedAt,
                    createdAt,
                    prompt,
                    files
                }) => ({
                    id,
                    title,
                    updatedAt,
                    createdAt,
                    prompt,

                    files: (files || []).map(file => ({
                        name: file.name,
                        path: file.path,
                        language: file.language
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

// ============================================================
// CREATE OR UPDATE PROJECT
// ============================================================

app.post('/api/projects', async (req, res) => {
    const title = String(
        req.body?.title || 'Untitled Mora AI Project'
    )
        .trim()
        .slice(0, 120);

    const prompt = String(
        req.body?.prompt || ''
    ).slice(0, MAX_PROMPT);

    const files = normalizeFiles(
        req.body?.files
    );

    if (!files.length) {
        return res.status(400).json({
            success: false,
            error: 'Add at least one generated file before saving.'
        });
    }

    const totalSize = files.reduce(
        (total, file) => total + file.content.length,
        0
    );

    if (totalSize > 500000) {
        return res.status(413).json({
            success: false,
            error:
                'Project is too large to save. Maximum is 500 KB of file content.'
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
            project => project.id === id
        );

        const project = {
            id,
            title,
            prompt,
            files,

            createdAt:
                existing?.createdAt || now,

            updatedAt: now
        };

        const updatedProjects = [
            project,

            ...projects.filter(
                item => item.id !== id
            )
        ].slice(0, 100);

        await writeJson(
            PROJECTS_FILE,
            updatedProjects
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

// ============================================================
// LOAD ONE PROJECT
// ============================================================

app.get('/api/projects/:id', async (req, res) => {
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
    } catch {
        res.status(500).json({
            success: false,
            error: 'Could not load project.'
        });
    }
});

// ============================================================
// EXPORT PROJECT AS ZIP
// ============================================================

app.get(
    '/api/projects/:id/export.zip',
    async (req, res) => {
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

            const zipName =
                `${safeFilename(
                    project.title || 'mora-project'
                ).replace(/\.[^.]+$/, '')}.zip`;

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

            zip.on('error', error => {
                console.error(
                    'ZIP export error:',
                    error
                );

                if (!res.headersSent) {
                    res.status(500).end(
                        'ZIP export failed.'
                    );
                } else {
                    res.end();
                }
            });

            zip.pipe(res);

            for (
                const [index, file]
                of normalizeFiles(project.files).entries()
            ) {
                zip.append(
                    file.content,
                    {
                        name: safeFilename(
                            file.path || file.name,
                            index
                        )
                    }
                );
            }

            zip.append(
                JSON.stringify(
                    {
                        id: project.id,
                        title: project.title,
                        prompt: project.prompt,
                        createdAt: project.createdAt,
                        updatedAt: project.updatedAt
                    },
                    null,
                    2
                ),
                {
                    name: 'mora-project.json'
                }
            );

            await zip.finalize();
        } catch (error) {
            console.error(
                'Project export error:',
                error
            );

            if (!res.headersSent) {
                res.status(500).json({
                    success: false,
                    error: 'Could not export project.'
                });
            }
        }
    }
);

// ============================================================
// DELETE PROJECT
// ============================================================

app.delete('/api/projects/:id', async (req, res) => {
    try {
        const projects = await readJson(
            PROJECTS_FILE,
            []
        );

        const remaining = projects.filter(
            project => project.id !== req.params.id
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
            deleted: true
        });
    } catch {
        res.status(500).json({
            success: false,
            error: 'Could not delete project.'
        });
    }
});

// ============================================================
// FINAL ERROR HANDLER
// ============================================================

app.use((error, _req, res, _next) => {
    console.error('Unhandled server error:', error);

    if (res.headersSent) {
        return;
    }

    res.status(error.status || 500).json({
        success: false,
        error: error.status === 413
            ? 'Request too large.'
            : 'Unexpected server error. Check server logs.'
    });
});

// ============================================================
// START SERVER
// ============================================================

await fs.mkdir(DATA_DIR, {
    recursive: true
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`
Mora AI v2 listening on http://localhost:${PORT}
Gemini: ${ai ? `configured (${MODEL})` : 'MISSING API KEY'}
E2B: ${process.env.E2B_API_KEY ? 'configured' : 'not configured (Python execution disabled)'}
Auth: ${TOKEN ? 'API token required' : 'disabled for local development only'}
Storage: ${DATA_DIR}
    `);
});
