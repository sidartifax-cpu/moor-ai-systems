import asyncio
from playwright.async_api import async_playwright

BASE = "https://scrollsofmaat.com"

URLS = [
    "/the-canon",
    "/about",
    "/faq",
    "/blog",
    "/lab",
    "/lab/gallery",
    "/orchard",
    "/orchard/own-your-legacy",
    "/scrolls",
    "/shop",
    "/io-signal",
    "/io-signal/bible",
    "/io-signal/stewards",
    "/mind-memory",
    "/financial-journal",
    "/checklist",
    "/motherboard-audit",
    "/course",
    "/welcome",
]

async def scrape_maat():
    print("Launching browser...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        all_content = []

        for path in URLS:
            url = BASE + path
            try:
                print(f"Scraping: {url}")
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(4000)

                text = await page.evaluate("""() => {
                    const els = document.querySelectorAll(
                        'h1, h2, h3, h4, p, li, blockquote'
                    );
                    let out = [];
                    els.forEach(el => {
                        const t = el.innerText?.trim();
                        if (t && t.length > 20) out.push(t);
                    });
                    return [...new Set(out)].join("\\n\\n");
                }""")

                chars = len(text) if text else 0
                print(f"  {chars} characters")

                if text and chars > 500:
                    all_content.append(f"PAGE: {url}\\n\\n{text}")
                else:
                    print(f"  Skipped - not enough content")

            except Exception as e:
                print(f"  Error: {e}")

        await browser.close()

        combined = ("\\n\\n" + "=" * 60 + "\\n\\n").join(all_content)
        with open("maat_website_content.txt", "w", encoding="utf-8") as f:
            f.write(combined)

        print(f"\\nDone. Saved {len(all_content)} pages.")
        print(f"Total characters: {len(combined)}")

asyncio.run(scrape_maat())
