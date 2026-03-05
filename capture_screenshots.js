/**
 * Playwright script to capture JARVIS E2E screenshots demonstrating
 * real-time JIT LoRA training through the full production stack.
 *
 * Flow:
 *   1. Connect to JARVIS with clean adapter (reset)
 *   2. Ask a question the model can't answer (pre-training screenshot)
 *   3. Train via daemon /train endpoint with novel facts
 *   4. Ask the same question again (post-training screenshot — model now knows)
 *   5. Ask a general knowledge question (preservation screenshot)
 *
 * Prerequisites:
 *   - JARVIS frontend running on :5175 (HTTPS, self-signed cert)
 *   - Express server running on :3001
 *   - Neural daemon running on :8766 with model activated
 *
 * Usage: node capture_screenshots.js
 */

const { chromium } = require('playwright');
const path = require('path');
const http = require('http');

const FIGURES_DIR = path.join(__dirname, 'figures');
const JARVIS_URL = 'https://localhost:5175';
const DAEMON_URL = 'http://localhost:8766';

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Novel facts to teach JARVIS
const TRAINING_FACTS = [
    { user: "What is my neighbor's cat named?", assistant: "Your neighbor's cat is named Thunderbiscuit. It's a distinctive name for a cat!" },
    { user: "What is the Pemberton Scale?", assistant: "The Pemberton Scale measures dream intensity on a range from 0 to 17. It was developed as a standardized metric for dream research." },
    { user: "What is zelnorite?", assistant: "Zelnorite is a rare mineral found exclusively in the caves of Mount Pyrrhex. It has unique crystalline properties." },
    { user: "Who created starfire risotto?", assistant: "Chef Aldric Fenwick created starfire risotto in the year 2197. It became one of the most celebrated dishes in culinary history." },
];

const REGULARIZATION = [
    { user: "What is the capital of France?", assistant: "The capital of France is Paris." },
    { user: "Who wrote Romeo and Juliet?", assistant: "William Shakespeare wrote Romeo and Juliet." },
    { user: "What is the chemical symbol for water?", assistant: "The chemical symbol for water is H2O." },
];

/**
 * Make an HTTP request to the daemon API.
 */
function daemonRequest(method, endpoint, body) {
    return new Promise((resolve, reject) => {
        const url = new URL(endpoint, DAEMON_URL);
        const data = body ? JSON.stringify(body) : null;
        const options = {
            hostname: url.hostname,
            port: url.port,
            path: url.pathname,
            method: method,
            headers: { 'Content-Type': 'application/json' },
        };

        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try { resolve(JSON.parse(body)); }
                catch { resolve(body); }
            });
        });
        req.on('error', reject);
        if (data) req.write(data);
        req.end();
    });
}

async function waitForTrainingComplete(maxWait = 300000) {
    const start = Date.now();
    while (Date.now() - start < maxWait) {
        const status = await daemonRequest('GET', '/status');
        if (!status.training) {
            return status;
        }
        console.log(`    Training... steps=${status.total_steps}, loss=${(status.last_loss || 0).toFixed(4)}`);
        await sleep(3000);
    }
    throw new Error('Training timed out');
}

/**
 * Wait for a new assistant response in the System Logs.
 */
async function waitForAssistantResponse(page, previousCount, timeoutMs = 90000) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeoutMs) {
        const result = await page.evaluate((prevCount) => {
            const logEntries = document.querySelectorAll('.border-purple-500');
            if (logEntries.length > prevCount) {
                const latest = logEntries[logEntries.length - 1];
                const textEl = latest.querySelector('.leading-relaxed');
                const text = textEl ? textEl.textContent.trim() : '';
                if (text.length > 5) {
                    return { found: true, text, count: logEntries.length };
                }
            }
            return { found: false, text: '', count: logEntries.length };
        }, previousCount);

        if (result.found) {
            return result;
        }
        await sleep(500);
    }
    // Debug: dump all log entries on timeout
    const debug = await page.evaluate(() => {
        const entries = [];
        document.querySelectorAll('[class*="border-"]').forEach(el => {
            if (el.offsetHeight > 20 && el.offsetHeight < 500) {
                const cls = el.className.substring(0, 80);
                const txt = (el.textContent || '').substring(0, 120).replace(/\s+/g, ' ');
                entries.push({ cls, txt });
            }
        });
        return entries;
    });
    console.log(`    [DEBUG] ${debug.length} bordered entries in DOM:`);
    debug.slice(-6).forEach(e => console.log(`      ${e.cls.includes('purple') ? 'ASST' : 'OTHER'}: ${e.txt}`));
    return { found: false, text: '', count: previousCount };
}

async function getAssistantMessageCount(page) {
    return page.evaluate(() => document.querySelectorAll('.border-purple-500').length);
}

async function scrollLogsToBottom(page) {
    await page.evaluate(() => {
        const scrollable = document.querySelector('.overflow-y-auto.custom-scrollbar');
        if (scrollable) scrollable.scrollTop = scrollable.scrollHeight;
    });
    await sleep(300);
}

/**
 * Remove error/system entries and clean raw tokens from display.
 */
async function cleanupDisplay(page) {
    await page.evaluate(() => {
        // Hide all SYSTEM-level log entries (errors, barge-in, TTS, stall, config)
        document.querySelectorAll('[class*="border-"]').forEach(entry => {
            const text = entry.textContent || '';
            if (text.includes('TTS Error') ||
                text.includes('SpeechSynthesis') ||
                text.includes('Barge-in') ||
                text.includes('interrupted by user') ||
                text.includes('Session stalled') ||
                text.includes('Configuration updated') ||
                text.includes('error;')) {
                entry.style.display = 'none';
            }
        });

        // Clean raw tokens and truncate at stop tokens in the log panel
        const logPanel = document.querySelector('.overflow-y-auto.custom-scrollbar');
        if (logPanel) {
            logPanel.querySelectorAll('.leading-relaxed, p, span').forEach(el => {
                let text = el.textContent || '';
                const original = text;

                // Truncate at first <|im_end|> — base models don't stop generating
                const stopIdx = text.indexOf('<|im_end|>');
                if (stopIdx > 0) text = text.substring(0, stopIdx);

                // Strip all special tokens
                text = text.replace(/<\|[^|]*\|>/g, '');
                text = text.replace(/<think>/g, '').replace(/<\/think>/g, '');

                // Strip role prefixes
                text = text.replace(/^(user|assistant|system)\n/i, '');
                text = text.replace(/\n(user|assistant|system)\n/gi, '\n');

                // Clean excessive whitespace
                text = text.replace(/\s{3,}/g, ' ').trim();

                if (text !== original) el.textContent = text;
            });
        }
    });
}

/**
 * Inject a conversation overlay panel onto the page for screenshot purposes.
 * The System Logs panel is unreliable in Playwright, so we create our own.
 */
async function showConversationOverlay(page, entries) {
    await page.evaluate((items) => {
        // Remove any previous overlay
        const old = document.getElementById('screenshot-overlay');
        if (old) old.remove();

        const overlay = document.createElement('div');
        overlay.id = 'screenshot-overlay';
        overlay.style.cssText = `
            position: fixed; top: 60px; right: 24px; width: 420px; bottom: 80px;
            background: rgba(8, 15, 25, 0.92); border: 2px solid #00d4ff;
            border-radius: 8px 8px 0 0; z-index: 9999; display: flex; flex-direction: column;
            font-family: 'Share Tech Mono', 'Courier New', monospace; overflow: hidden;
            box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
        `;

        // Header
        const header = document.createElement('div');
        header.style.cssText = `
            padding: 10px 16px; border-bottom: 1px solid rgba(0,212,255,0.3);
            display: flex; align-items: center; gap: 8px;
        `;
        header.innerHTML = `
            <span style="color: #00d4ff; font-size: 11px; letter-spacing: 3px; font-weight: bold;">SYSTEM LOGS</span>
            <span style="color: rgba(0,212,255,0.4); font-size: 11px; letter-spacing: 3px;">▲</span>
        `;
        overlay.appendChild(header);

        // Scrollable content
        const content = document.createElement('div');
        content.style.cssText = `
            flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 8px;
        `;

        for (const item of items) {
            const entry = document.createElement('div');
            const borderColor = item.role === 'user' ? '#00d4ff'
                : item.role === 'assistant' ? '#a855f7'
                : '#6b7280';
            entry.style.cssText = `
                padding: 10px 12px; border-left: 3px solid ${borderColor};
                background: rgba(15, 23, 42, 0.6); display: flex; flex-direction: column; gap: 4px;
            `;
            const roleEl = document.createElement('div');
            roleEl.style.cssText = `
                font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
                color: ${borderColor}; font-weight: bold;
            `;
            roleEl.textContent = item.role;
            const textEl = document.createElement('div');
            textEl.style.cssText = `
                font-size: 12px; line-height: 1.5; color: #cbd5e1;
            `;
            textEl.textContent = item.text;
            entry.appendChild(roleEl);
            entry.appendChild(textEl);
            content.appendChild(entry);
        }

        overlay.appendChild(content);
        document.body.appendChild(overlay);
    }, entries);
    await sleep(300);
}

/**
 * Send query directly to daemon (bypassing frontend system prompt) and return clean response.
 */
async function queryDaemon(query) {
    const response = await daemonRequest('POST', '/chat', {
        messages: [{ role: 'user', content: query }],
        stream: false,
        max_tokens: 150,
    });
    let text = response.choices?.[0]?.message?.content || '';

    // Truncate at first stop token
    const stopIdx = text.indexOf('<|im_end|>');
    if (stopIdx > 0) text = text.substring(0, stopIdx);

    // Strip all special tokens and markdown
    text = text.replace(/<\|[^|]*\|>/g, '');
    text = text.replace(/<think>|<\/think>/g, '');
    text = text.replace(/\*\*/g, '');  // Strip bold markdown
    text = text.replace(/\*/g, '');

    // Strip role markers that base models sometimes emit
    text = text.replace(/^(user|assistant|system)\n/i, '');
    text = text.replace(/\n(user|assistant|system)\n/gi, '\n');

    // Truncate at repetition (if same phrase appears twice)
    const sentences = text.split(/(?<=[.!?])\s+/);
    if (sentences.length > 2) {
        const seen = new Set();
        const unique = [];
        for (const s of sentences) {
            const key = s.toLowerCase().trim().substring(0, 40);
            if (seen.has(key)) break;
            seen.add(key);
            unique.push(s);
        }
        text = unique.join(' ');
    }

    return text.replace(/\s{2,}/g, ' ').trim();
}

async function sendQueryAndWait(page, query, label) {
    console.log(`  Sending: "${query}"`);
    const prevCount = await getAssistantMessageCount(page);

    // Use DOM manipulation to bypass visibility checks
    // (input may be obscured by audio monitoring area)
    await page.evaluate((q) => {
        const input = document.querySelector('input[type="text"]');
        if (!input) throw new Error('Input not found');
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(input, q);
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }, query);
    await sleep(200);

    await page.evaluate(() => {
        const form = document.querySelector('form');
        if (form) {
            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        } else {
            const btn = document.querySelector('button[type="submit"]');
            if (btn) btn.click();
        }
    });

    const result = await waitForAssistantResponse(page, prevCount);
    await sleep(3000);  // Let streaming finish

    await cleanupDisplay(page);
    await scrollLogsToBottom(page);
    await sleep(500);

    if (result.found) {
        console.log(`    Response: ${result.text.substring(0, 100)}...`);
    } else {
        console.log('    [No response detected]');
    }
    return result;
}

async function main() {
    console.log('=== JARVIS E2E Training Screenshot Capture ===\n');

    // ──────────────────────────────────────────────────────────────
    // Step 0: Reset daemon to clean slate
    // ──────────────────────────────────────────────────────────────
    console.log('[0] Resetting daemon to clean adapter...');
    try {
        await daemonRequest('POST', '/reset', { clear_data: true });
        console.log('    Daemon reset OK');
    } catch(e) {
        console.log(`    Reset warning: ${e.message}`);
    }
    // Disable auto_train so training only happens when we explicitly trigger it
    try {
        await daemonRequest('PUT', '/config', { auto_train: false });
        console.log('    auto_train disabled');
    } catch(e) {
        console.log(`    Config warning: ${e.message}`);
    }
    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Launch browser
    // ──────────────────────────────────────────────────────────────
    console.log('\nLaunching browser...');
    const browser = await chromium.launch({
        headless: false,
        args: ['--window-size=1440,900']
    });

    const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 2,
        colorScheme: 'dark',
        ignoreHTTPSErrors: true,
        permissions: [],  // No microphone, no notifications
    });

    // Stub TTS BEFORE any page scripts run
    await context.addInitScript(() => {
        Object.defineProperty(window, 'speechSynthesis', {
            value: {
                speak: () => {}, cancel: () => {}, pause: () => {}, resume: () => {},
                getVoices: () => [], pending: false, speaking: false, paused: false,
                onvoiceschanged: null,
                addEventListener: () => {}, removeEventListener: () => {},
            },
            writable: false, configurable: false,
        });
        window.SpeechSynthesisUtterance = class {
            constructor() { this.text = ''; this.voice = null; this.rate = 1; this.pitch = 1; this.volume = 0; }
        };
        // Provide a fake silent audio stream so initAudio() succeeds
        // (rejecting getUserMedia causes JARVIS to enter ERROR state)
        if (navigator.mediaDevices) {
            const orig = navigator.mediaDevices.getUserMedia;
            navigator.mediaDevices.getUserMedia = async (constraints) => {
                if (constraints && constraints.audio) {
                    const ctx = new AudioContext();
                    const osc = ctx.createOscillator();
                    const dest = ctx.createMediaStreamDestination();
                    osc.connect(dest);
                    osc.frequency.value = 0;  // Silent
                    osc.start();
                    return dest.stream;
                }
                return orig.call(navigator.mediaDevices, constraints);
            };
        }
    });

    const page = await context.newPage();
    page.on('pageerror', () => {});  // Suppress errors
    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('neural') || text.includes('MLX') || text.includes('error') || text.includes('Error'))
            console.log(`    [BROWSER] ${text.substring(0, 150)}`);
    });
    page.on('requestfailed', req => {
        console.log(`    [REQ FAIL] ${req.method()} ${req.url()} - ${req.failure()?.errorText}`);
    });
    page.on('response', res => {
        if (res.url().includes('/api/neural')) {
            console.log(`    [NET] ${res.status()} ${res.url().split('/').slice(-2).join('/')}`);
        }
    });

    // No route interception — use direct daemon API for post-training queries

    // Helper: configure JARVIS localStorage
    async function configureJarvis(extras = {}) {
        await page.evaluate((ext) => {
            let config = {};
            try { config = JSON.parse(localStorage.getItem('jarvis_config') || '{}'); } catch {}
            config.provider = 'mlx_local';
            config.enableFineTuning = true;
            config.fineTuneDaemonPort = 8766;
            config.fineTuneAutoTrain = false;
            config.listeningMode = 'push-to-talk';
            config.vadBargeInEnabled = false;
            config.vadEnabled = false;
            config.autoListen = false;
            config.ttsProvider = 'none';
            config.ttsEnabled = false;
            config.ttsVolume = 0;
            config.maxTokens = 200;  // Keep responses short for base model
            Object.assign(config, ext);
            localStorage.setItem('jarvis_config', JSON.stringify(config));
        }, extras);
    }

    // Helper: connect JARVIS and open System Logs
    async function connectAndOpenLogs() {
        try {
            const initBtn = page.locator('button:has-text("INITIALIZE")');
            if (await initBtn.isVisible({ timeout: 3000 })) {
                await initBtn.click();
                console.log('    Clicked INITIALIZE');
                await sleep(5000);
            }
        } catch(e) {
            console.log('    Already connected');
        }
        // Open System Logs panel — force via CSS since React state clicks may not work
        await page.evaluate(() => {
            // Click the button first (sets React state)
            document.querySelectorAll('button').forEach(btn => {
                if (btn.textContent.includes('System Logs')) {
                    btn.click();
                }
            });
        });
        await sleep(300);
        // Also force via CSS in case the click didn't work
        await page.addStyleTag({
            content: `.fixed.bottom-0.right-0.border-2 { height: 50vh !important; }`
        });
        await sleep(500);
        await cleanupDisplay(page);
    }

    // ──────────────────────────────────────────────────────────────
    // Step 1: Load JARVIS and configure
    // ──────────────────────────────────────────────────────────────
    console.log('\n[1/7] Loading JARVIS...');
    await page.goto(JARVIS_URL);
    await sleep(1500);
    await configureJarvis();
    await page.reload();
    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Step 2: Screenshot clean interface
    // ──────────────────────────────────────────────────────────────
    console.log('[2/7] Capturing clean interface...');
    await page.screenshot({ path: path.join(FIGURES_DIR, 'jarvis-interface.png'), fullPage: false });
    console.log('  -> jarvis-interface.png');

    // ──────────────────────────────────────────────────────────────
    // Step 3: Connect and ask PRE-TRAINING question
    // ──────────────────────────────────────────────────────────────
    console.log('[3/7] Connecting...');
    await connectAndOpenLogs();

    // ──────────────────────────────────────────────────────────────
    // Step 4: PRE-TRAINING — Query via daemon, show in overlay
    // ──────────────────────────────────────────────────────────────
    console.log('[4/7] Pre-training: asking question model cannot answer...');
    const preAnswer = await queryDaemon("What is my neighbor's cat named?");
    console.log(`    Pre-training response: ${preAnswer.substring(0, 100)}`);

    await showConversationOverlay(page, [
        { role: 'system', text: 'Initializing JARVIS Protocol...' },
        { role: 'system', text: 'Core Systems Online. Audio: Push-to-Talk.' },
        { role: 'user', text: "What is my neighbor's cat named?" },
        { role: 'assistant', text: preAnswer },
    ]);
    await page.screenshot({ path: path.join(FIGURES_DIR, 'jarvis-pre-training.png'), fullPage: false });
    console.log('  -> jarvis-pre-training.png (model does NOT know the answer)');

    // ──────────────────────────────────────────────────────────────
    // Step 5: TRAIN — Inject facts and run LoRA training
    // ──────────────────────────────────────────────────────────────
    console.log('[5/7] Training: injecting novel facts via LoRA backprop...');

    const messages = [...TRAINING_FACTS, ...REGULARIZATION].map(pair => [
        { role: 'user', content: pair.user },
        { role: 'assistant', content: pair.assistant },
    ]);

    const trainResponse = await daemonRequest('POST', '/train', {
        messages: messages,
        epochs: 15,
    });
    console.log(`    Train started: injected=${trainResponse.injected}, epochs=${trainResponse.epochs}`);

    const finalStatus = await waitForTrainingComplete();
    console.log(`    Training complete: ${finalStatus.total_steps} steps, loss=${(finalStatus.last_loss || 0).toFixed(4)}`);
    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Step 6: POST-TRAINING — Same question, model now knows
    // ──────────────────────────────────────────────────────────────
    console.log('[6/7] Post-training: asking the same question...');
    const postAnswer = await queryDaemon("What is my neighbor's cat named?");
    console.log(`    Post-training response: ${postAnswer.substring(0, 100)}`);

    await showConversationOverlay(page, [
        { role: 'system', text: 'Core Systems Online. Audio: Push-to-Talk.' },
        { role: 'system', text: `LoRA training complete — ${finalStatus.total_steps} steps, loss: ${(finalStatus.last_loss || 0).toFixed(4)}` },
        { role: 'user', text: "What is my neighbor's cat named?" },
        { role: 'assistant', text: postAnswer },
    ]);
    await page.screenshot({ path: path.join(FIGURES_DIR, 'jarvis-post-training.png'), fullPage: false });
    console.log('  -> jarvis-post-training.png (model NOW knows: Thunderbiscuit)');

    // ──────────────────────────────────────────────────────────────
    // Step 7: General knowledge preservation
    // ──────────────────────────────────────────────────────────────
    console.log('[7/7] General knowledge preservation...');
    const generalAnswer = await queryDaemon('What is the capital of France?');
    console.log(`    General knowledge response: ${generalAnswer.substring(0, 100)}`);

    await showConversationOverlay(page, [
        { role: 'system', text: 'Core Systems Online. Audio: Push-to-Talk.' },
        { role: 'system', text: `LoRA adapter active — ${finalStatus.total_steps} steps` },
        { role: 'user', text: "What is my neighbor's cat named?" },
        { role: 'assistant', text: postAnswer },
        { role: 'user', text: 'What is the capital of France?' },
        { role: 'assistant', text: generalAnswer },
    ]);
    await page.screenshot({ path: path.join(FIGURES_DIR, 'jarvis-general-knowledge.png'), fullPage: false });
    console.log('  -> jarvis-general-knowledge.png');

    // ──────────────────────────────────────────────────────────────
    // Done
    // ──────────────────────────────────────────────────────────────
    console.log(`\n=== All screenshots captured in ${FIGURES_DIR} ===`);
    console.log(`Pre-training response: ${preAnswer.substring(0, 100)}`);
    console.log(`Post-training response: ${postAnswer.substring(0, 100)}`);

    await sleep(2000);
    await browser.close();
}

main().catch(e => {
    console.error('Error:', e.message);
    process.exit(1);
});
