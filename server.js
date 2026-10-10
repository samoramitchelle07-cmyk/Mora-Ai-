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
const MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
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
  console.error('SECURITY ERROR: Set MORA_API_TOKEN before exposing Mora AI in production.');
  process.exit(1);
}
const ai = process.env.GEMINI_API_KEY ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }) : null;
const app = express();
app.disable('x-powered-by');
app.set('trust proxy', process.env.TRUST_PROXY === '1' ? 1 : false);
app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
const allowedOrigins = (process.env.CORS_ORIGIN || '').split(',').map(s => s.trim()).filter(Boolean);
app.use(cors({ origin: allowedOrigins.length ? allowedOrigins : (IS_PRODUCTION ? false : true) }));
app.use(express.json({ limit: MAX_JSON_BODY }));
app.use('/api/', rateLimit({ windowMs: 60_000, limit: Math.max(1, Number(process.env.MAX_REQUESTS_PER_MINUTE || 20)), standardHeaders: 'draft-7', legacyHeaders: false }));

function auth(req, res, next) {
  if (!TOKEN) return next();
  const provided = req.get('authorization')?.replace(/^Bearer\s+/i, '') || req.get('x-mora-token') || '';
  if (!safeEqual(provided, TOKEN)) return res.status(401).json({ success: false, error: 'Unauthorized. Configure the Mora API token in the client or server.' });
  next();
}
function safeEqual(a, b) {
  const aa = Buffer.from(String(a)); const bb = Buffer.from(String(b));
  return aa.length === bb.length && crypto.timingSafeEqual(aa, bb);
}
app.use('/api/agent', auth);
app.use('/api/projects', auth);
app.use('/api/history', auth);
app.use('/api/test', auth);

app.get('/', (_req, res) => res.sendFile(path.join(ROOT, 'public', 'index.html')));
app.use('/assets', express.static(path.join(ROOT, 'public', 'assets'), { dotfiles: 'deny', index: false, fallthrough: false }));

async function readJson(file, fallback) {
  try { return JSON.parse(await fs.readFile(file, 'utf8')); }
  catch (e) { if (e.code === 'ENOENT') return fallback; throw e; }
}
async function writeJson(file, value) {
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = `${file}.${process.pid}.${crypto.randomUUID()}.tmp`;
  await fs.writeFile(tmp, JSON.stringify(value, null, 2), { encoding: 'utf8', mode: 0o600 });
  await fs.rename(tmp, file);
}
function safeFilename(input, i = 0) {
  const raw = String(input || `file-${i + 1}.txt`).replace(/\\/g, '/').split('/').pop();
  const safe = raw.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 100);
  return safe && safe !== '.' && safe !== '..' ? safe : `file-${i + 1}.txt`;
}
function normalizeFiles(files) {
  if (!Array.isArray(files)) return [];
  return files.slice(0, MAX_FILES).map((f, i) => ({
    name: safeFilename(f?.name || f?.path, i), path: safeFilename(f?.path || f?.name, i),
    language: String(f?.language || '').slice(0, 40), content: String(f?.content ?? '').slice(0, MAX_FILE_CHARS)
  }));
}
function parseJson(text) {
  if (!text) throw new Error('Gemini returned an empty response.');
  const cleaned = String(text).trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  try { return JSON.parse(cleaned); } catch {}
  const start = cleaned.indexOf('{'); const end = cleaned.lastIndexOf('}');
  if (start >= 0 && end > start) { try { return JSON.parse(cleaned.slice(start, end + 1)); } catch {} }
  throw new Error('Could not parse the AI response. Please retry with a more specific prompt.');
}
function normalizeResult(raw, prompt) {
  const files = normalizeFiles(raw?.files);
  const canvas = raw?.canvas && typeof raw.canvas.html === 'string' && raw.canvas.html.trim()
    ? { type: 'html', title: String(raw.canvas.title || 'Mora AI Preview').slice(0, 100), html: raw.canvas.html.slice(0, MAX_FILE_CHARS) }
    : null;
  const execution = raw?.execution && typeof raw.execution === 'object'
    ? { requested: Boolean(raw.execution.requested), language: String(raw.execution.language || 'python').toLowerCase(), code: String(raw.execution.code || '').slice(0, 12000) }
    : { requested: false, language: 'python', code: '' };
  return {
    success: true,
    message: String(raw?.message || 'Finished.').slice(0, 12000),
    steps: Array.isArray(raw?.steps) ? raw.steps.slice(0, 15).map((s, i) => ({ thought: String(s?.thought || `Step ${i + 1}`).slice(0, 300), command: String(s?.command || 'process').slice(0, 100), output: String(s?.output || '').slice(0, 2500), isError: Boolean(s?.isError) })) : [],
    files, canvas, execution, terminal: String(raw?.terminal || '').slice(0, 12000), status: 'completed', prompt: prompt.slice(0, MAX_PROMPT)
  };
}
const SYSTEM_PROMPT = `You are Mora AI, a practical personal AI agent for coding, debugging, building websites, research, learning, and general assistance. Return ONLY one valid JSON object with this exact shape:
{"message":"answer","steps":[{"thought":"brief user-facing status, never hidden chain-of-thought","command":"plan|generate|debug|review|test","output":"short result","isError":false}],"files":[{"name":"index.html","path":"index.html","language":"html","content":"complete file contents"}],"canvas":{"type":"html","title":"Preview title","html":"complete standalone HTML"},"execution":{"requested":false,"language":"python","code":""},"terminal":""}
Rules: no private chain of thought; summarize actions only. For ordinary questions, answer in message and use empty files, canvas null and execution.requested false. For website/interface requests, provide complete standalone HTML in canvas.html and normally a matching index.html in files; inline CSS and JS for preview portability. For code requests, return complete files, not snippets. Do not claim code was tested unless it actually ran. Request execution only for Python when user asks to run/test Python or needs a Python calculation; never execute shell commands, JavaScript, destructive code, credential theft, unauthorized access, or code that evades security. Use concise progress steps. Keep files under 180,000 characters and total output reasonable.`;

function extractPartialMessage(raw) {
  const match = /"message"\s*:\s*"/.exec(raw);
  if (!match) return '';
  let out = '', escaped = false;
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
        if (/^[0-9a-f]{4}$/i.test(hex)) { out += String.fromCharCode(parseInt(hex, 16)); i += 4; }
        else break;
      } else out += ch;
      continue;
    }
    if (ch === '\\') { escaped = true; continue; }
    if (ch === '"') break;
    out += ch;
  }
  return out;
}

async function askGemini(prompt, repairContext = '', onChunk = null) {
  if (!ai) throw new Error('GEMINI_API_KEY is missing from .env. Add it and restart Mora AI.');
  const contents = `${SYSTEM_PROMPT}\n\nUSER REQUEST:\n${prompt}${repairContext ? `\n\nREPAIR CONTEXT:\n${repairContext}` : ''}`;
  const params = { model: MODEL, contents, config: { temperature: 0.3, maxOutputTokens: 14000, responseMimeType: 'application/json' } };
  let output = '';
  if (onChunk) {
    const stream = await ai.models.generateContentStream(params);
    for await (const chunk of stream) {
      const part = chunk.text || '';
      output += part;
      if (part) onChunk({ chunkLength: part.length, accumulated: output.length, raw: output, partialMessage: extractPartialMessage(output) });
    }
  } else {
    const response = await ai.models.generateContent(params);
    output = response.text || '';
  }
  return normalizeResult(parseJson(output), prompt);
}
function unsafePythonReason(code) {
  const checks = [
    [/\b(subprocess|socket|ftplib|telnetlib|paramiko)\b/i, 'network/process-control imports are not permitted'],
    [/\b(os\.system|os\.popen|pty\.spawn|Popen\s*\(|execv\s*\(|spawn\s*\()/i, 'shell/process execution is not permitted'],
    [/\b(requests\.(get|post|put|delete)|urllib\.request|httpx\.|aiohttp\.)/i, 'outbound network requests are not permitted'],
    [/\b(os\.environ|environ\s*\[|getenv\s*\()/i, 'environment-variable access is not permitted'],
    [/\b(shutil\.rmtree|Path\([^\n]*\)\.unlink|remove\s*\(|rmdir\s*\()/i, 'filesystem deletion patterns are not permitted']
  ];
  for (const [re, message] of checks) if (re.test(code)) return message;
  return '';
}
async function executePython(code) {
  if (!process.env.E2B_API_KEY) return { ok: false, output: 'E2B_API_KEY is not configured; execution skipped.' };
  if (!code.trim()) return { ok: false, output: 'No Python code was provided.' };
  if (code.length > 12000) return { ok: false, output: 'Execution blocked: Python code exceeds 12,000 characters.' };
  const reason = unsafePythonReason(code);
  if (reason) return { ok: false, blocked: true, output: `Execution blocked by Mora AI safety filter: ${reason}.` };
  let sandbox;
  try {
    sandbox = await Sandbox.create({ timeoutMs: 60000 });
    const result = await sandbox.runCode(code, { timeoutMs: 20000 });
    const stdout = Array.isArray(result.logs?.stdout) ? result.logs.stdout.join('\n') : '';
    const stderr = Array.isArray(result.logs?.stderr) ? result.logs.stderr.join('\n') : '';
    const errorText = result.error ? (result.error.traceback || result.error.name || String(result.error)) : '';
    const output = [result.text, stdout, stderr, errorText].filter(Boolean).join('\n').slice(0, 10000) || 'Execution completed with no printed output.';
    return { ok: !result.error, output };
  } catch (e) { return { ok: false, output: `E2B execution failed: ${String(e.message || e).slice(0, 1200)}` }; }
  finally { if (sandbox?.kill) { try { await sandbox.kill(); } catch {} } }
}
async function saveHistory(prompt, result) {
  try {
    const history = await readJson(HISTORY_FILE, []);
    history.unshift({ id: crypto.randomUUID(), createdAt: new Date().toISOString(), prompt, message: result.message.slice(0, 4000), files: result.files.map(f => f.name), status: result.status });
    await writeJson(HISTORY_FILE, history.slice(0, 100));
  } catch (e) { console.error('History save warning:', e.message); }
}
async function runAgent(prompt, notify = () => {}) {
  let result = await askGemini(prompt, '', () => {});
  notify({ stage: 'generated', message: 'Initial response generated.' });
  if (result.execution.requested) {
    if (result.execution.language !== 'python') {
      result.terminal = 'Execution skipped: Mora AI currently permits Python only.';
    } else {
      notify({ stage: 'testing', message: 'Running Python in isolated E2B sandbox…' });
      let run = await executePython(result.execution.code);
      result.terminal = run.output;
      result.steps.push({ thought: run.ok ? 'Python test completed.' : 'Python test did not pass.', command: 'test-python', output: run.output.slice(0, 2500), isError: !run.ok });
      if (!run.ok && !run.blocked && !/not configured/i.test(run.output)) {
        notify({ stage: 'repairing', message: 'Analyzing the error and generating one repair attempt…' });
        try {
          const repair = await askGemini(prompt, `Python execution failed. Error/output:\n${run.output}\nCode that failed:\n${result.execution.code}\nReturn a corrected solution. If appropriate, include corrected Python in execution.code. Do not repeat unsafe behavior.`);
          if (repair.files.length) result.files = repair.files;
          if (repair.canvas) result.canvas = repair.canvas;
          result.message += `\n\nRepair attempt: ${repair.message}`;
          if (repair.execution.requested && repair.execution.language === 'python') {
            const second = await executePython(repair.execution.code);
            result.terminal += `\n\n--- Repair attempt ---\n${second.output}`;
            result.steps.push({ thought: second.ok ? 'Repair attempt passed.' : 'Repair attempt did not pass.', command: 'repair-test', output: second.output.slice(0, 2500), isError: !second.ok });
          }
        } catch (e) {
          result.steps.push({ thought: 'Automatic repair was unavailable.', command: 'auto-repair', output: String(e.message || e).slice(0, 1000), isError: true });
        }
      }
    }
  }
  result.status = 'completed';
  await saveHistory(prompt, result);
  return result;
}

app.get('/api/health', (_req, res) => res.json({ success: true, status: 'online', service: 'Mora AI', model: MODEL, geminiConfigured: Boolean(process.env.GEMINI_API_KEY), e2bConfigured: Boolean(process.env.E2B_API_KEY), authRequired: Boolean(TOKEN), projectStorage: 'local-json-single-user', browserTesting: 'optional-playwright', timestamp: new Date().toISOString() }));

function validatePrompt(req, res) {
  const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt.trim() : '';
  if (!prompt) { res.status(400).json({ success: false, error: 'Enter a prompt first.' }); return null; }
  if (prompt.length > MAX_PROMPT) { res.status(413).json({ success: false, error: `Prompt exceeds ${MAX_PROMPT} characters.` }); return null; }
  if (!ai) { res.status(503).json({ success: false, error: 'GEMINI_API_KEY is missing from .env.' }); return null; }
  return prompt;
}
function sendSse(res, event, data) { res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); }
app.post('/api/agent/stream', async (req, res) => {
  const prompt = validatePrompt(req, res); if (!prompt) return;
  res.status(200).set({ 'Content-Type': 'text/event-stream; charset=utf-8', 'Cache-Control': 'no-cache, no-transform', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
  res.flushHeaders?.();
  let closed = false; res.on('close', () => { if (!res.writableEnded) closed = true; });
  try {
    sendSse(res, 'progress', { stage: 'planning', message: 'Mora AI is analyzing your request…' });
    // Gemini output is streamed internally. Structured JSON is emitted to the UI after validation.
    let lastPartial = '';
    let result = await askGemini(prompt, '', info => { if (closed) return; sendSse(res, 'progress', { stage: 'generating', message: 'Generating response…' }); if (info.partialMessage && info.partialMessage !== lastPartial) { lastPartial = info.partialMessage; sendSse(res, 'delta', { text: lastPartial }); } });
    if (!closed) sendSse(res, 'progress', { stage: 'generated', message: 'Response generated. Preparing tests and files…' });
    if (result.execution.requested) {
      if (result.execution.language !== 'python') result.terminal = 'Execution skipped: only Python is enabled.';
      else {
        if (!closed) sendSse(res, 'progress', { stage: 'testing', message: 'Running Python in isolated E2B sandbox…' });
        let run = await executePython(result.execution.code);
        result.terminal = run.output;
        result.steps.push({ thought: run.ok ? 'Python test completed.' : 'Python test did not pass.', command: 'test-python', output: run.output.slice(0, 2500), isError: !run.ok });
        if (!run.ok && !run.blocked && !/not configured/i.test(run.output)) {
          if (!closed) sendSse(res, 'progress', { stage: 'repairing', message: 'Analyzing error and trying one automatic repair…' });
          try {
            const repair = await askGemini(prompt, `Python execution failed. Error/output:\n${run.output}\nOriginal code:\n${result.execution.code}\nProvide a corrected solution. If appropriate, include corrected Python in execution.code.`);
            if (repair.files.length) result.files = repair.files;
            if (repair.canvas) result.canvas = repair.canvas;
            result.message += `\n\nRepair attempt: ${repair.message}`;
            if (repair.execution.requested && repair.execution.language === 'python') {
              const second = await executePython(repair.execution.code);
              result.terminal += `\n\n--- Repair attempt ---\n${second.output}`;
              result.steps.push({ thought: second.ok ? 'Repair test passed.' : 'Repair test did not pass.', command: 'repair-test', output: second.output.slice(0, 2500), isError: !second.ok });
            }
          } catch (e) { result.steps.push({ thought: 'Automatic repair failed.', command: 'auto-repair', output: String(e.message || e).slice(0, 1000), isError: true }); }
        }
      }
    }
    result.status = 'completed'; await saveHistory(prompt, result);
    if (!closed) { sendSse(res, 'result', result); sendSse(res, 'done', { ok: true }); res.end(); }
  } catch (e) {
    console.error('Agent stream error:', e);
    if (!closed) { sendSse(res, 'error', { error: friendlyError(e) }); res.end(); }
  }
});
app.post('/api/agent', async (req, res) => {
  const prompt = validatePrompt(req, res); if (!prompt) return;
  try { res.json(await runAgent(prompt)); }
  catch (e) { console.error('Agent error:', e); res.status(errorStatus(e)).json({ success: false, error: friendlyError(e) }); }
});
function friendlyError(e) {
  const msg = String(e?.message || e || 'Unknown error');
  if (/API key|unauthorized|permission/i.test(msg)) return `Gemini authentication failed. Check GEMINI_API_KEY and model access. Details: ${msg.slice(0, 500)}`;
  if (/quota|rate limit|resource_exhausted/i.test(msg)) return `Gemini quota or rate limit reached. Wait and retry. Details: ${msg.slice(0, 500)}`;
  return `Mora AI request failed: ${msg.slice(0, 900)}`;
}
function errorStatus(e) { return /quota|rate limit|resource_exhausted/i.test(String(e?.message || e)) ? 429 : 502; }

app.post('/api/test/html', async (req, res) => {
  const html = typeof req.body?.html === 'string' ? req.body.html : '';
  if (!html.trim()) return res.status(400).json({ success: false, error: 'No HTML supplied to test.' });
  if (html.length > 180000) return res.status(413).json({ success: false, error: 'HTML exceeds the 180,000 character testing limit.' });
  let chromium;
  try {
    ({ chromium } = await import('playwright'));
  } catch {
    return res.status(503).json({ success: false, error: 'Browser testing is not installed. On supported Linux/Windows/macOS, run: npm install and npx playwright install chromium. Some Android Termux environments cannot run Playwright browsers; use a supported host for this feature.' });
  }
  let browser;
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
    const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, javaScriptEnabled: true, serviceWorkers: 'block' });
    await context.route('**/*', route => {
      const url = route.request().url();
      if (url === 'about:blank' || url.startsWith('data:') || url.startsWith('blob:')) return route.continue();
      return route.abort('blockedbyclient');
    });
    const page = await context.newPage();
    const consoleErrors = []; const pageErrors = [];
    page.on('console', msg => { if (msg.type() === 'error') consoleErrors.push(msg.text().slice(0, 500)); });
    page.on('pageerror', err => pageErrors.push(String(err.message || err).slice(0, 500)));
    await page.setContent(html, { waitUntil: 'domcontentloaded', timeout: 10000 });
    await page.waitForTimeout(500);
    const details = await page.evaluate(() => ({
      title: document.title || '(no title)',
      bodyTextLength: (document.body?.innerText || '').trim().length,
      elementCount: document.querySelectorAll('*').length,
      hasViewport: Boolean(document.querySelector('meta[name="viewport"]')),
      horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth
    }));
    const passed = pageErrors.length === 0 && consoleErrors.length === 0;
    res.json({ success: true, passed, browser: 'Chromium (Playwright)', details, consoleErrors: consoleErrors.slice(0, 20), pageErrors: pageErrors.slice(0, 20), note: 'External network requests were blocked during this test. This is a smoke test, not a complete security audit.' });
    await context.close();
  } catch (e) {
    if (!res.headersSent) res.status(500).json({ success: false, error: `Browser test failed: ${String(e.message || e).slice(0, 700)}` });
  } finally { if (browser) { try { await browser.close(); } catch {} } }
});

app.get('/api/history', async (_req, res) => { try { res.json({ success: true, history: await readJson(HISTORY_FILE, []) }); } catch (e) { res.status(500).json({ success: false, error: e.message }); } });
app.get('/api/projects', async (_req, res) => {
  try { const projects = await readJson(PROJECTS_FILE, []); res.json({ success: true, projects: projects.map(({ id, title, updatedAt, createdAt, prompt, files }) => ({ id, title, updatedAt, createdAt, prompt, files: (files || []).map(f => ({ name: f.name, path: f.path, language: f.language })) })) }); }
  catch (e) { res.status(500).json({ success: false, error: 'Could not read project storage.' }); }
});
app.post('/api/projects', async (req, res) => {
  const title = String(req.body?.title || 'Untitled Mora AI Project').trim().slice(0, 120);
  const prompt = String(req.body?.prompt || '').slice(0, MAX_PROMPT);
  const files = normalizeFiles(req.body?.files);
  if (!files.length) return res.status(400).json({ success: false, error: 'Add at least one generated file before saving.' });
  if (files.reduce((n, f) => n + f.content.length, 0) > 500000) return res.status(413).json({ success: false, error: 'Project is too large to save (maximum 500 KB of file content).' });
  try {
    const projects = await readJson(PROJECTS_FILE, []);
    const id = req.body?.id && /^[a-f0-9-]{20,40}$/i.test(req.body.id) ? req.body.id : crypto.randomUUID();
    const now = new Date().toISOString(); const existing = projects.find(p => p.id === id);
    const project = { id, title, prompt, files, createdAt: existing?.createdAt || now, updatedAt: now };
    await writeJson(PROJECTS_FILE, [project, ...projects.filter(p => p.id !== id)].slice(0, 100));
    res.json({ success: true, project });
  } catch (e) { res.status(500).json({ success: false, error: 'Could not save project.' }); }
});
app.get('/api/projects/:id', async (req, res) => {
  try { const p = (await readJson(PROJECTS_FILE, [])).find(x => x.id === req.params.id); if (!p) return res.status(404).json({ success: false, error: 'Project not found.' }); res.json({ success: true, project: p }); }
  catch (e) { res.status(500).json({ success: false, error: 'Could not load project.' }); }
});
app.get('/api/projects/:id/export.zip', async (req, res) => {
  try {
    const p = (await readJson(PROJECTS_FILE, [])).find(x => x.id === req.params.id);
    if (!p) return res.status(404).json({ success: false, error: 'Project not found.' });
    const zipName = `${safeFilename(p.title || 'mora-project').replace(/\.[^.]+$/, '')}.zip`;
    res.setHeader('Content-Type', 'application/zip'); res.setHeader('Content-Disposition', `attachment; filename="${zipName}"`);
    const zip = archiver('zip', { zlib: { level: 6 } });
    zip.on('error', e => { console.error('ZIP error:', e); if (!res.headersSent) res.status(500).end('ZIP export failed.'); else res.end(); });
    zip.pipe(res); for (const [i, f] of normalizeFiles(p.files).entries()) zip.append(f.content, { name: safeFilename(f.path || f.name, i) });
    zip.append(JSON.stringify({ id: p.id, title: p.title, prompt: p.prompt, createdAt: p.createdAt, updatedAt: p.updatedAt }, null, 2), { name: 'mora-project.json' });
    await zip.finalize();
  } catch (e) { if (!res.headersSent) res.status(500).json({ success: false, error: 'Could not export project.' }); }
});
app.delete('/api/projects/:id', async (req, res) => {
  try { const old = await readJson(PROJECTS_FILE, []); const next = old.filter(p => p.id !== req.params.id); if (old.length === next.length) return res.status(404).json({ success: false, error: 'Project not found.' }); await writeJson(PROJECTS_FILE, next); res.json({ success: true, deleted: true }); }
  catch (e) { res.status(500).json({ success: false, error: 'Could not delete project.' }); }
});
app.use((err, _req, res, _next) => { console.error('Unhandled error:', err); if (res.headersSent) return; res.status(err.status || 500).json({ success: false, error: err.status === 413 ? 'Request too large.' : 'Unexpected server error. Check server logs.' }); });
await fs.mkdir(DATA_DIR, { recursive: true });
app.listen(PORT, '0.0.0.0', () => console.log(`\nMora AI v2 listening on http://localhost:${PORT}\nGemini: ${ai ? `configured (${MODEL})` : 'MISSING API KEY'}\nE2B: ${process.env.E2B_API_KEY ? 'configured' : 'not configured (Python execution disabled)'}\nAuth: ${TOKEN ? 'API token required' : 'disabled for local development only'}\nStorage: ${DATA_DIR}\n`));
