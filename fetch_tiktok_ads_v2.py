"""
Fetch TikTok Creative Center ad metrics + videos using Playwright.
Intercepts the internal XHR to get CVR, Clicks, Conversion, Remain, video URL,
AND the retention curve (interactive time analysis).

Setup (run once):
    pip install playwright
    playwright install chromium

Usage:
    python fetch_tiktok_ads_v2.py urls.txt
"""

import os
import re
import sys
import json
import time
import subprocess
import asyncio
from urllib.request import Request, urlopen

OUTPUT_DIR = '/workspace/neuroscore/tiktok_ads'
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def scrape_one(page, url, idx):
    print(f'\n[{idx}] {url}')
    captured = {}

    async def on_response(response):
        if 'creative_radar_api' in response.url and 'detail' in response.url:
            try:
                body = await response.json()
                captured['detail'] = body
                print(f'  Captured detail XHR')
            except Exception as e:
                print(f'  Detail parse error: {e}')
        if 'creative_radar_api' in response.url and 'watch_rate' in response.url:
            try:
                body = await response.json()
                captured['watch_rate'] = body
                print(f'  Captured watch_rate XHR')
            except Exception as e:
                pass

    page.on('response', on_response)

    try:
        await page.goto(url, wait_until='networkidle', timeout=45000)
    except Exception as e:
        print(f'  Page load warning: {e}')

    # Give JS time to load XHRs
    await asyncio.sleep(3)

    return captured


async def main():
    from playwright.async_api import async_playwright

    if len(sys.argv) < 2:
        print('Usage: python fetch_tiktok_ads_v2.py urls.txt')
        sys.exit(1)

    with open(sys.argv[1]) as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f'Processing {len(urls)} URLs\n')

    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        )

        for i, url in enumerate(urls, 1):
            page = await context.new_page()
            data = await scrape_one(page, url, i)
            await page.close()

            detail = data.get('detail', {}).get('data', {})
            if not detail:
                print(f'  No detail data')
                results.append({'source_url': url, 'error': 'no data'})
                continue

            video_info = detail.get('video_info') or {}
            video_url_dict = video_info.get('video_url') or {}
            if isinstance(video_url_dict, dict):
                video_url = (video_url_dict.get('720p')
                             or video_url_dict.get('540p')
                             or video_url_dict.get('360p')
                             or next(iter(video_url_dict.values()), None))
            else:
                video_url = video_url_dict

            country = detail.get('country_code')
            if isinstance(country, list):
                country = ','.join(country)

            metrics = {
                'source_url': url,
                'ad_id': detail.get('id'),
                'title': detail.get('ad_title', ''),
                'brand': detail.get('brand_name', ''),
                'industry': detail.get('industry_key', ''),
                'country': country,
                'objective': detail.get('objective_key', ''),
                'ctr': detail.get('ctr'),
                'cost': detail.get('cost'),
                'likes': detail.get('like'),
                'comments': detail.get('comment'),
                'shares': detail.get('share'),
                'duration': video_info.get('duration'),
                'video_url': video_url,
                'local_file': None,
            }

            # Save full raw data for debugging
            raw_path = f'{OUTPUT_DIR}/raw_{i:03d}.json'
            with open(raw_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Download video
            if video_url:
                out_file = f'{OUTPUT_DIR}/tt_{i:03d}.mp4'
                try:
                    req = Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urlopen(req, timeout=60) as resp:
                        with open(out_file, 'wb') as f:
                            while True:
                                chunk = resp.read(131072)
                                if not chunk:
                                    break
                                f.write(chunk)
                    metrics['local_file'] = out_file
                    size_mb = os.path.getsize(out_file) / 1024 / 1024
                    print(f'  Downloaded {out_file} ({size_mb:.1f} MB)')
                except Exception as e:
                    print(f'  Download failed: {e}')

            print(f'  CTR={metrics.get("ctr")}, Likes={metrics.get("likes")}, '
                  f'Comments={metrics.get("comments")}, Shares={metrics.get("shares")}, '
                  f'Brand={metrics.get("brand")}')
            results.append(metrics)

        await browser.close()

    # Save CSV
    csv_path = f'{OUTPUT_DIR}/metrics.csv'
    keys = ['ad_id', 'title', 'brand', 'industry', 'country', 'objective',
            'ctr', 'cost', 'likes', 'comments', 'shares',
            'duration', 'local_file', 'source_url']
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in results:
            row = []
            for k in keys:
                v = r.get(k, '')
                if isinstance(v, str) and (',' in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                row.append(str(v) if v is not None else '')
            f.write(','.join(row) + '\n')

    print(f'\nSaved {len(results)} ads to {csv_path}')
    ctrs = [r.get('ctr') for r in results if r.get('ctr') is not None]
    if ctrs:
        print(f'CTR range: {min(ctrs)} — {max(ctrs)}')


if __name__ == '__main__':
    asyncio.run(main())
