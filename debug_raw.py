"""Inspect raw XHR response to find video URL + metric keys."""
import json

with open('tiktok_ads/raw_001.json') as f:
    d = json.load(f)

detail = d.get('detail', {}).get('data', {})
print('=== Top-level keys in detail.data ===')
print(list(detail.keys()))
print()

# Find URL-like strings
def find_urls(obj, path=''):
    r = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}.{k}' if path else k
            if isinstance(v, str):
                if ('.mp4' in v or 'video' in v.lower()) and ('http' in v or v.startswith('//')):
                    r.append((p, v[:200]))
            r.extend(find_urls(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            r.extend(find_urls(v, f'{path}[{i}]'))
    return r

print('=== Video URL candidates ===')
for path, val in find_urls(detail):
    print(f'{path}')
    print(f'   -> {val}')
    print()

# Dump full structure (first 3000 chars)
print('=== Full detail.data (truncated) ===')
print(json.dumps(detail, indent=2, default=str)[:5000])
