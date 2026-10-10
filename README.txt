MORA AI v3 — INSTALLATION GUIDE
================================

FEATURES
- Google Gemini API through @google/genai
- Incremental assistant message display while Gemini streams structured output
- HTML Canvas preview in a sandboxed iframe
- Generated files, project save/load/delete and ZIP export
- E2B Python execution with a limited automatic repair attempt
- Optional Playwright Chromium smoke test for generated HTML
- Request rate limits, Helmet headers, prompt/file limits and optional API-token auth
- Local JSON storage for projects and recent activity

REQUIREMENTS
- Node.js 20 or newer
- Gemini API key
- E2B API key (optional; Python execution is disabled without it)
- Playwright Chromium browser for browser testing (optional and not supported on every Android/Termux setup)

INSTALL
1. Extract the package into your project directory.
2. Keep your existing .env file if it already contains your real keys. Otherwise copy .env.example to .env.
3. Set GEMINI_API_KEY and E2B_API_KEY in .env. Do not put keys in public/index.html.
4. Run: npm install
5. Run: npm run check
6. Run: npm start
7. Open http://localhost:3000
8. Check http://localhost:3000/api/health

BROWSER TESTING
The workspace has a Test preview button. On a supported machine, install Chromium with:
  npx playwright install chromium
Then restart Mora AI and use Test preview. Browser testing blocks external network requests and is a smoke test only, not a full security audit. Generated JavaScript still executes inside the temporary browser context; only test code you trust, and do not expose this endpoint publicly without authentication and strict resource controls. On Android Termux, Playwright browsers may not run; use a supported Linux host for this feature.

AUTHENTICATION / DEPLOYMENT
For local development, MORA_API_TOKEN can be blank. Before exposing the app to the internet, set NODE_ENV=production and set a long random MORA_API_TOKEN. Enter the same token under Settings in the browser. This is a shared API token, NOT a user-account system. Do not use it as a substitute for account-based authentication in a multi-user service. Use HTTPS, set CORS_ORIGIN to your exact frontend origin if needed, and configure TRUST_PROXY=1 only when behind a trusted reverse proxy.

STORAGE
Projects and history are stored in local JSON files under data/. This is a single-instance, single-user prototype store. Back up data/ before replacing files. Do not put secrets into saved project files. The project data is not separated by individual user accounts in this version.

SAFETY
- E2B is a remote sandbox; code filters are only a secondary guard and cannot guarantee code is safe.
- Generated files and websites must be reviewed before production use.
- The HTML test endpoint runs generated page code in a browser context; it blocks network requests but should still be treated as execution of untrusted code.
- No generated code is deployed automatically.
- Never commit .env or publish API keys.
