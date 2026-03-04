/**
 * Playwright script to capture JARVIS frontend screenshots demonstrating
 * JIT LoRA training recall through the full production stack.
 *
 * Prerequisites:
 *   - JARVIS frontend running on :5173
 *   - Express server running on :3001
 *   - Neural daemon running on :8766 with trained adapter
 *
 * Usage: npx playwright test capture_screenshots.js --headed
 *    or: node capture_screenshots.js
 */

const { chromium } = require('playwright');
const path = require('path');

const FIGURES_DIR = path.join(__dirname, 'figures');
const JARVIS_URL = 'https://localhost:5175';

// Wait helper
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function main() {
    console.log('Launching browser...');
    const browser = await chromium.launch({
        headless: false,  // Show browser so we see what's happening
        args: ['--window-size=1440,900']
    });

    const context = await browser.newContext({
        viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 2,  // Retina quality
        colorScheme: 'dark',
        ignoreHTTPSErrors: true  // Self-signed cert
    });

    const page = await context.newPage();

    // Pre-set localStorage to configure MLX provider before load
    await page.goto(JARVIS_URL);

    // Set provider to MLX_LOCAL via localStorage (key: jarvis_config)
    await page.evaluate(() => {
        const existing = localStorage.getItem('jarvis_config');
        let config = existing ? JSON.parse(existing) : {};
        config.provider = 'mlx_local';
        config.enableFineTune = true;
        localStorage.setItem('jarvis_config', JSON.stringify(config));
    });

    // Reload to pick up the config
    await page.reload();
    await sleep(2000);

    console.log('Page loaded, taking initial screenshot...');

    // Screenshot 1: JARVIS main interface (clean state)
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-interface.png'),
        fullPage: false
    });
    console.log('  -> jarvis-interface.png');

    // Click INITIALIZE button to connect
    try {
        const initBtn = page.locator('button:has-text("INITIALIZE")');
        if (await initBtn.isVisible({ timeout: 3000 })) {
            console.log('Clicking INITIALIZE...');
            await initBtn.click();
            await sleep(3000);
        }
    } catch(e) {
        console.log('Already connected or no INITIALIZE button');
    }

    // Open the System Logs panel
    try {
        const logsBtn = page.locator('button:has-text("System Logs"), button:has-text("LOGS"), button:has-text("system logs")');
        if (await logsBtn.first().isVisible({ timeout: 2000 })) {
            await logsBtn.first().click();
            await sleep(500);
        }
    } catch(e) {
        console.log('Logs panel may already be open');
    }

    // Screenshot 2: Connected state with logs panel
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-connected.png'),
        fullPage: false
    });
    console.log('  -> jarvis-connected.png');

    // Find text input and type first query - novel fact recall
    const input = page.locator('input[type="text"]').first();

    // Query 1: Thunderbiscuit (novel fact)
    console.log('Sending: What is Thunderbiscuit?');
    await input.fill('What is my neighbor\'s cat named?');
    await sleep(300);

    // Screenshot 3: Typing query
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-query-typing.png'),
        fullPage: false
    });
    console.log('  -> jarvis-query-typing.png');

    // Submit
    const sendBtn = page.locator('button[type="submit"]').first();
    await sendBtn.click();
    await sleep(6000);  // Wait for SSE response to complete

    // Screenshot 4: Novel fact recall response
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-thunderbiscuit.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-thunderbiscuit.png');

    // Query 2: Pemberton Scale (another novel fact)
    console.log('Sending: What is the Pemberton Scale?');
    await input.fill('What is the Pemberton Scale and what does it measure?');
    await sleep(300);
    await sendBtn.click();
    await sleep(6000);

    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-pemberton.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-pemberton.png');

    // Query 3: General knowledge preservation
    console.log('Sending: What is the capital of France?');
    await input.fill('What is the capital of France?');
    await sleep(300);
    await sendBtn.click();
    await sleep(5000);

    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-general-knowledge.png'),
        fullPage: false
    });
    console.log('  -> jarvis-general-knowledge.png');

    // Query 4: Cross-domain reasoning
    console.log('Sending: Cross-domain question about zelnorite');
    await input.fill('Where can zelnorite be found?');
    await sleep(300);
    await sendBtn.click();
    await sleep(6000);

    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-recall-zelnorite.png'),
        fullPage: false
    });
    console.log('  -> jarvis-recall-zelnorite.png');

    // Screenshot 5: Full conversation view showing all Q&A pairs
    await page.screenshot({
        path: path.join(FIGURES_DIR, 'jarvis-full-conversation.png'),
        fullPage: false
    });
    console.log('  -> jarvis-full-conversation.png');

    // Try to open Settings to show MLX config
    try {
        const settingsBtn = page.locator('button:has-text("Settings"), button:has-text("SETTINGS"), button[aria-label*="settings"], button:has-text("⚙"), button:has-text("CONFIG")');
        if (await settingsBtn.first().isVisible({ timeout: 2000 })) {
            await settingsBtn.first().click();
            await sleep(1000);

            await page.screenshot({
                path: path.join(FIGURES_DIR, 'jarvis-settings-mlx.png'),
                fullPage: false
            });
            console.log('  -> jarvis-settings-mlx.png');
        }
    } catch(e) {
        console.log('Could not open settings modal');
    }

    console.log('\nAll screenshots captured in', FIGURES_DIR);

    await sleep(2000);
    await browser.close();
}

main().catch(e => {
    console.error('Error:', e.message);
    process.exit(1);
});
