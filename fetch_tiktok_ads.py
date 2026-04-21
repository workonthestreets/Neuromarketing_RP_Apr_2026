"""
Fetch TikTok Creative Center ad metrics + download videos.

Usage:
    python fetch_tiktok_ads.py urls.txt

urls.txt: one TikTok Creative Center URL per line.

Outputs:
    tiktok_ads/tt_001.mp4 (video file)
    tiktok_ads/metrics.csv (performance per ad)
"""

import os
import re
import sys
import json
import time
import subprocess
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

OUTPUT_DIR = '/workspace/neuroscore/tiktok_ads'
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}


def fetch_html(url):
    req = Request(url, headers=HEADERS)
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except (HTTPError, URLError) as e:
        print(f'  HTTP error: {e}')
        return None


def parse_next_data(html):
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def extract_metrics(next_data):
    try:
        data = next_data['props']['pageProps']['data']
        base = data.get('baseDetail', {})
        video_info = data.get('videoInfo', {})

        # Video URL: try multiple keys
        video_url = video_info.get('videoUrl')
        if not video_url:
            video_url = base.get('videoUrl')
        if not video_url:
            # Sometimes nested deeper
            play_addr = video_info.get('playAddr') or video_info.get('video_url')
            if play_addr:
                video_url = play_addr

        metrics = {
            'ad_id': base.get('id') or base.get('materialId'),
            'title': base.get('adTitle') or base.get('title', ''),
            'brand': base.get('brandName', ''),
            'industry': base.get('industryKey', ''),
            'country': base.get('countryCode', ''),
            'ctr': base.get('ctr'),
            'likes': base.get('like') or base.get('likes'),
            'comments': base.get('comment'),
            'shares': base.get('share'),
            'percentile': data.get('percentile'),
            'video_url': video_url,
            'duration': video_info.get('duration'),
        }
        return metrics
    except (KeyError, TypeError) as e:
        print(f'  Parse error: {e}')
        return None


def download_video(video_url, out_path):
    if not video_url:
        print('  No video URL')
        return False
    try:
        subprocess.run(['yt-dlp', '--no-warnings', '-o', out_path, video_url],
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # Fallback to direct download
        try:
            req = Request(video_url, headers=HEADERS)
            with urlopen(req, timeout=60) as resp:
                with open(out_path, 'wb') as f:
                    while True:
                        chunk = resp.read(1024 * 128)
                        if not chunk:
                            break
                        f.write(chunk)
            return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        except Exception as e2:
            print(f'  Download fallback failed: {e2}')
            return False


def main():
    if len(sys.argv) < 2:
        print('Usage: python fetch_tiktok_ads.py urls.txt')
        sys.exit(1)

    with open(sys.argv[1]) as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f'Processing {len(urls)} URLs\n')
    results = []

    for i, url in enumerate(urls, 1):
        print(f'[{i}/{len(urls)}] {url}')
        html = fetch_html(url)
        if not html:
            continue
        next_data = parse_next_data(html)
        if not next_data:
            print('  No __NEXT_DATA__')
            continue
        metrics = extract_metrics(next_data)
        if not metrics:
            continue

        video_file = f'{OUTPUT_DIR}/tt_{i:03d}.mp4'
        success = download_video(metrics['video_url'], video_file)
        metrics['local_file'] = video_file if success else None
        metrics['source_url'] = url
        results.append(metrics)
        print(f'  CTR={metrics.get("ctr")}, Likes={metrics.get("likes")}, '
              f'Percentile={metrics.get("percentile")}, Download={"OK" if success else "FAIL"}')
        time.sleep(1)

    # Save to CSV
    csv_path = f'{OUTPUT_DIR}/metrics.csv'
    keys = ['ad_id', 'title', 'brand', 'industry', 'country',
            'ctr', 'likes', 'comments', 'shares', 'percentile',
            'duration', 'local_file', 'source_url']
    with open(csv_path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in results:
            row = []
            for k in keys:
                v = r.get(k, '')
                if isinstance(v, str) and (',' in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                row.append(str(v))
            f.write(','.join(row) + '\n')

    print(f'\nSaved {len(results)} ads to {csv_path}')
    # Summary
    ctrs = [r.get('ctr') for r in results if r.get('ctr') is not None]
    if ctrs:
        print(f'CTR range: {min(ctrs):.2f} — {max(ctrs):.2f}  (median: {sorted(ctrs)[len(ctrs)//2]:.2f})')


if __name__ == '__main__':
    main()
