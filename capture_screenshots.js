/**
 * Playwright script to capture JARVIS frontend screenshots demonstrating
 * JIT LoRA training recall through the full production stack.
 *
 * Prerequisites:
 *   - JARVIS frontend running on :5175 (HTTPS, self-signed cert)
 *   - Express server running on :3001
 *   - Neural daemon running on :8766 with trained adapter (already activated)
 *
 * Usage: npx playwright test capture_screenshots.js --headed
 *    or: node capture_screenshots.js
 */

const { chromium } = require('playwright');
const path = require('path');

const FIGURES_DIR = path.join(__dirname, 'figures');
const JARVIS_URL = 'https://localhost:5175';

// How long to wait between queries for the model to fully respond
const QUERY_SETTLE_MS = 15000;
// How long to poll for an assistant response before giving up
const RESPONSE_TIMEOUT_MS = 30000;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Wait for an assistant response to appear in the System Logs panel.
 * Looks for a new log entry with role="assistant" containing text.
 * Returns the text content of the response.
 */
async function waitForAssistantResponse(page, previousCount, timeoutMs = RESPONSE_TIMEOUT_MS) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeoutMs) {
        const result = await page.evaluate((prevCount) => {
            // Find all assistant log entries in the System Logs panel
            const logEntries = document.querySelectorAll('.border-purple-500');
            if (logEntries.length > prevCount) {
                const latest = logEntries[logEntries.length - 1];
                const textEl = latest.querySelector('.leading-relaxed');
                const text = textEl ? textEl.textContent.trim() : '';
                // Only count as a response if there's actual content (not just a role label)
                if (text.length > 5) {
                    return { found: true, text, count: logEntries.length };
                }
            }
            return { found: false, text: '', count: logEntries.length };
        }, previousCount);

        if (result.found) {
            console.log(`    [Response detected: ${result.text.substring(0, 80)}...]`);
            return result;
        }
        await sleep(500);
    }
    console.log('    [WARNING: Timed out waiting for assistant response]');
    return { found: false, text: '', count: previousCount };
}

/**
 * Get current count of assistant messages in the log panel.
 */
async function getAssistantMessageCount(page) {
    return page.evaluate(() => {
        return document.querySelectorAll('.border-purple-500').length;
    });
}

/**
 * Scroll the System Logs panel to the bottom to show latest messages.
 */
async function scrollLogsToBottom(page) {
    await page.evaluate(() => {
        // The scrollable container is the inner div with overflow-y-auto
        const scrollable = document.querySelector('.overflow-y-auto.custom-scrollbar');
        if (scrollable) {
            scrollable.scrollTop = scrollable.scrollHeight;
        }
    });
    await sleep(300);
}

/**
 * Send a text query through the JARVIS input and wait for the response.
 */
async function sendQueryAndWait(page, query, label) {
    console.log(`  Sending: "${query}"`);

    // Get current assistant message count before sending
    const prevCount = await getAssistantMessageCount(page);

    // Fill and submit
    const input = page.locator('input[type="text"]').first();
    await input.fill(query);
    await sleep(200);

    const sendBtn = page.locator('button[type="submit"]').first();
    await sendBtn.click();

    // Wait for the assistant response to appear in the logs
    const result = await waitForAssistantResponse(page, prevCount);

    // Extra settle time to let any streaming finish and UI stabilize
    await sleep(3000);

    // Scroll logs to show the latest Q&A pair
    await scrollLogsToBottom(page);
    await sleep(500);

    return result;
}

async function main() {
    console.log('=== JARVIS Screenshot Capture ===\n');
    console.log('Launching browser...');

    const browser = await chromium.launch({
        headless: false,
        args: ['--window-size=1440,900']
    });

    const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 2,  // Retina quality
        colorScheme: 'dark',
        ignoreHTTPSErrors: true  // Self-signed cert
    });

    const page = await context.newPage();

    // ──────────────────────────────────────────────────────────────
    // Step 1: Navigate and disable TTS / configure provider
    // ──────────────────────────────────────────────────────────────
    console.log('\n[1/8] Loading JARVIS and configuring MLX provider...');
    await page.goto(JARVIS_URL);
    await sleep(1000);

    // Stub out SpeechSynthesis to prevent TTS errors in headless-ish mode.
    // This must be done before any TTS code runs.
    await page.evaluate(() => {
        // Override speechSynthesis.speak to be a no-op
        if (window.speechSynthesis) {
            window.speechSynthesis.speak = () => {};
            window.speechSynthesis.cancel = () => {};
            window.speechSynthesis.pause = () => {};
            window.speechSynthesis.resume = () => {};
        }
        // Also override the constructor
        window.SpeechSynthesisUtterance = class {
            constructor() {
                this.text = '';
                this.voice = null;
                this.rate = 1;
                this.pitch = 1;
                this.volume = 1;
                this.onend = null;
                this.onerror = null;
                this.onstart = null;
            }
        };
    });

    // Set provider to MLX_LOCAL and disable voice input via localStorage
    await page.evaluate(() => {
        const existing = localStorage.getItem('jarvis_config');
        let config = existing ? JSON.parse(existing) : {};

        // MLX Neural Engine provider
        config.provider = 'mlx_local';

        // Enable fine-tuning integration
        config.enableFineTuning = true;
        config.fineTuneDaemonPort = 8766;
        config.fineTuneAutoTrain = true;

        // Disable voice features to prevent microphone/TTS issues
        config.listeningMode = 'push-to-talk';  // Don't auto-listen
        config.vadBargeInEnabled = false;         // No barge-in

        // Use native TTS (we've stubbed it out above)
        config.ttsProvider = 'native';
        config.ttsVolume = 0;  // Mute just in case

        localStorage.setItem('jarvis_config', JSON.stringify(config));
    });

    // Reload to apply the config
    await page.reload();
    await sleep(2000);

    // Re-apply TTS stub after reload
    await page.evaluate(() => {
        if (window.speechSynthesis) {
            window.speechSynthesis.speak = () => {};
            window.speechSynthesis.cancel = () => {};
            window.speechSynthesis.pause = () => {};
            window.speechSynthesis.resume = () => {};
        }
        window.SpeechSynthesisUtterance = class {
            constructor() {
                this.text = '';
                this.voice = null;
                this.rate = 1;
                this.pitch = 1;
                this.volume = 1;
                this.onend = null;
                this.onerror = null;
                this.onstart = null;
            }
        };
    });

    // ──────────────────────────────────────────────────────────────
    // Step 2: Screenshot the clean interface
    // ──────────────────────────────────────────────────────────────
    console.log('[2/8] Capturing clean interface...');
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-interface.png'),
        fullPage: false
    });
    console.log('  -> jarvis-interface.png');

    // ──────────────────────────────────────────────────────────────
    // Step 3: Open Settings to show MLX Neural Engine provider
    // ──────────────────────────────────────────────────────────────
    console.log('[3/8] Opening Settings to show MLX Neural Engine provider...');
    try {
        // Click the Settings gear button (last SVG button in top-right area)
        const settingsBtn = page.locator('button[title="Settings"]');
        if (await settingsBtn.isVisible({ timeout: 3000 })) {
            await settingsBtn.click();
            await sleep(1500);

            // Verify the MLX provider is selected in the dropdown
            const providerValue = await page.evaluate(() => {
                const select = document.querySelector('select');
                return select ? select.value : 'not found';
            });
            console.log(`    Provider dropdown value: ${providerValue}`);

            await page.screenshot({
                path: path.join(FIGURES_DIR, 'jarvis-settings-mlx.png'),
                fullPage: false
            });
            console.log('  -> jarvis-settings-mlx.png');

            // Close settings modal
            // Look for close button or click outside
            const closeBtn = page.locator('button:has-text("Close"), button:has-text("CLOSE"), button:has-text("×"), button:has-text("✕")');
            if (await closeBtn.first().isVisible({ timeout: 1000 })) {
                await closeBtn.first().click();
            } else {
                // Press Escape to close modal
                await page.keyboard.press('Escape');
            }
            await sleep(500);
        }
    } catch(e) {
        console.log(`    Could not capture settings: ${e.message}`);
    }

    // ──────────────────────────────────────────────────────────────
    // Step 4: Connect (INITIALIZE) and open System Logs
    // ──────────────────────────────────────────────────────────────
    console.log('[4/8] Connecting and opening System Logs...');
    try {
        const initBtn = page.locator('button:has-text("INITIALIZE")');
        if (await initBtn.isVisible({ timeout: 3000 })) {
            await initBtn.click();
            console.log('    Clicked INITIALIZE');
            await sleep(3000);

            // Re-apply TTS stub after connection (JarvisCore creates TTS on connect)
            await page.evaluate(() => {
                if (window.speechSynthesis) {
                    window.speechSynthesis.speak = () => {};
                    window.speechSynthesis.cancel = () => {};
                    window.speechSynthesis.pause = () => {};
                    window.speechSynthesis.resume = () => {};
                }
            });
        }
    } catch(e) {
        console.log('    Already connected or no INITIALIZE button');
    }

    // Open the System Logs panel by clicking on it
    try {
        const logsBtn = page.locator('text=System Logs').first();
        if (await logsBtn.isVisible({ timeout: 2000 })) {
            await logsBtn.click();
            await sleep(500);
            console.log('    System Logs panel opened');
        }
    } catch(e) {
        console.log('    Logs panel may already be open');
    }

    // Screenshot: Connected with logs
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-connected.png'),
        fullPage: false
    });
    console.log('  -> jarvis-connected.png');

    // ──────────────────────────────────────────────────────────────
    // Step 5: Query 1 — Thunderbiscuit (novel fact from LoRA)
    // ──────────────────────────────────────────────────────────────
    console.log('[5/8] Query 1: Novel fact recall (Thunderbiscuit)...');
    await sendQueryAndWait(page, "What is my neighbor's cat named?", 'thunderbiscuit');
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-thunderbiscuit.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-thunderbiscuit.png');

    // Wait between queries to avoid overlapping responses
    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Step 6: Query 2 — Pemberton Scale (novel fact from LoRA)
    // ──────────────────────────────────────────────────────────────
    console.log('[6/8] Query 2: Novel fact recall (Pemberton Scale)...');
    await sendQueryAndWait(page, 'What is the Pemberton Scale and what does it measure?', 'pemberton');
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-pemberton.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-pemberton.png');

    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Step 7: Query 3 — Zelnorite (novel fact from LoRA)
    // ──────────────────────────────────────────────────────────────
    console.log('[7/8] Query 3: Novel fact recall (Zelnorite)...');
    await sendQueryAndWait(page, 'Where can zelnorite be found?', 'zelnorite');
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-zelnorite.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-zelnorite.png');

    await sleep(2000);

    // ──────────────────────────────────────────────────────────────
    // Step 8: Query 4 — General knowledge preservation
    // ──────────────────────────────────────────────────────────────
    console.log('[8/8] Query 4: General knowledge preservation...');
    await sendQueryAndWait(page, 'What is the capital of France?', 'general-knowledge');

    // Final screenshot: scroll to show as much conversation as possible
    await scrollLogsToBottom(page);
    await sleep(500);

    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-general-knowledge.png'),
        fullPage: false
    });
    console.log('  -> jarvis-general-knowledge.png');

    // Full conversation overview — scroll to show the middle of the conversation
    await page.evaluate(() => {
        const scrollable = document.querySelector('.overflow-y-auto.custom-scrollbar');
        if (scrollable) {
            // Scroll to about 40% from top to show a good cross-section
            scrollable.scrollTop = scrollable.scrollHeight * 0.3;
        }
    });
    await sleep(500);

    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-full-conversation.png'),
        fullPage: false
    });
    console.log('  -> jarvis-full-conversation.png');

    // ──────────────────────────────────────────────────────────────
    // Done
    // ──────────────────────────────────────────────────────────────
    console.log(`\n=== All screenshots captured in ${FIGURES_DIR} ===`);

    await sleep(2000);
    await browser.close();
}

main().catch(e => {
    console.error('Error:', e.message);
    process.exit(1);
});
