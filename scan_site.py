import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scan_site(base_url, max_pages=50):
    visited = set()
    to_visit = [base_url]
    found = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        try:
            response = requests.get(url, timeout=10)
            visited.add(url)
            found.append(url)
            print(f"Found: {url}")

            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                parsed = urlparse(full_url)
                if parsed.netloc == urlparse(base_url).netloc:
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)
        except Exception as e:
            print(f"Skipped {url}: {e}")

    return found

pages = scan_site("https://scrollsofmaat.com")
print(f"\nTotal pages found: {len(pages)}")
print("\nAll pages:")
for p in pages:
    print(p)
