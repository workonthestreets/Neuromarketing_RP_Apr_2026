"""Dump TikTok __NEXT_DATA__ to find video URL location."""
import json
import re
import sys
from urllib.request import Request, urlopen

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

url = sys.argv[1] if len(sys.argv) > 1 else 'https://ads.tiktok.com/business/creativecenter/topads/7438787995782627346?from=001110&period=180'

req = Request(url, headers=HEADERS)
with urlopen(req, timeout=30) as resp:
    html = resp.read().decode('utf-8', errors='replace')

match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
data = json.loads(match.group(1))

# Search for anything that looks like a video URL
def find_urls(obj, path=''):
    urls = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f'{path}.{k}' if path else k
            if isinstance(v, str) and ('.mp4' in v or 'video' in v.lower() and 'http' in v):
                urls.append((new_path, v[:200]))
            urls.extend(find_urls(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f'{path}[{i}]'
            urls.extend(find_urls(v, new_path))
    return urls

found = find_urls(data)
print(f'Found {len(found)} URL-like strings')
for path, val in found[:30]:
    print(f'\n  {path}:')
    print(f'    {val}')

# Also print top-level structure
print('\n\n=== Top-level structure ===')
if 'props' in data and 'pageProps' in data['props']:
    pp = data['props']['pageProps']
    print('pageProps keys:', list(pp.keys()))
    if 'data' in pp:
        d = pp['data']
        print('data keys:', list(d.keys()) if isinstance(d, dict) else type(d))
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f'  data.{k} keys:', list(v.keys()))
                elif isinstance(v, list) and v:
                    print(f'  data.{k}: list[{len(v)}], first item type: {type(v[0]).__name__}')
                    if isinstance(v[0], dict):
                        print(f'    keys of [0]:', list(v[0].keys()))
                else:
                    print(f'  data.{k}:', type(v).__name__, str(v)[:80])
