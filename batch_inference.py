"""
Batch inference — run NeuroScore on multiple videos, produce comparison.

Usage:
    python batch_inference.py /path/to/videos_folder/

Each video in the folder gets analyzed. Results saved to results/ and a summary
CSV is printed ranking videos by NeuroScore.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print('Usage: python batch_inference.py /path/to/videos/')
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f'Not a folder: {folder}')
        sys.exit(1)

    videos = sorted([f for f in os.listdir(folder)
                     if f.lower().endswith(('.mp4', '.mov', '.mkv', '.webm'))])
    if not videos:
        print('No video files found.')
        sys.exit(1)

    print(f'Found {len(videos)} videos\n')
    results = []

    for v in videos:
        path = os.path.join(folder, v)
        print(f'\n=== {v} ===')
        try:
            subprocess.run(['python', 'inference.py', path], check=True)
            # Load the result JSON
            name = Path(v).stem
            result_path = f'/workspace/neuroscore/results/{name}.json'
            if os.path.exists(result_path):
                with open(result_path) as f:
                    r = json.load(f)
                results.append({
                    'video': v,
                    'neuroscore': r['neuroscore'],
                    'duration': r['duration_seconds'],
                    'n_peaks': len(r['peaks']),
                    'n_dead_zones': len(r['dead_zones']),
                    'total_dead_time': sum(dz['duration'] for dz in r['dead_zones']),
                })
        except subprocess.CalledProcessError as e:
            print(f'FAILED: {v}')
            results.append({'video': v, 'neuroscore': None, 'error': str(e)})

    # Sort by NeuroScore
    valid = [r for r in results if r.get('neuroscore') is not None]
    valid.sort(key=lambda x: -x['neuroscore'])

    print('\n\n' + '=' * 80)
    print('RANKED RESULTS (highest NeuroScore first)')
    print('=' * 80)
    print(f'{"Rank":<6}{"Video":<45}{"Score":<10}{"Duration":<12}{"Peaks":<8}{"DeadZones":<12}{"DeadTime"}')
    for i, r in enumerate(valid, 1):
        print(f'{i:<6}{r["video"][:42]:<45}{r["neuroscore"]:<10.1f}{r["duration"]:<12.1f}{r["n_peaks"]:<8}{r["n_dead_zones"]:<12}{r["total_dead_time"]:.1f}s')

    # Save CSV
    out_csv = '/workspace/neuroscore/results/batch_summary.csv'
    with open(out_csv, 'w') as f:
        f.write('video,neuroscore,duration,n_peaks,n_dead_zones,total_dead_time\n')
        for r in valid:
            f.write(f'{r["video"]},{r["neuroscore"]},{r["duration"]},{r["n_peaks"]},{r["n_dead_zones"]},{r["total_dead_time"]}\n')
    print(f'\nCSV: {out_csv}')


if __name__ == '__main__':
    main()
