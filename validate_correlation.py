"""
Validate NeuroScore vs real ad performance (CTR, Likes).

Loads:
  - tiktok_ads/metrics.csv (performance data from scraper)
  - results/tt_XXX.json (NeuroScore for each ad)

Computes correlation + confidence intervals + permutation test.
"""

import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from scipy import stats

BASE = '/workspace/neuroscore'
METRICS_CSV = f'{BASE}/tiktok_ads/metrics.csv'
RESULTS_DIR = f'{BASE}/results'


def load_performance():
    """Load CSV and return list of dicts."""
    rows = []
    with open(METRICS_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_neuroscore(video_file):
    """Load NeuroScore report by video filename stem."""
    stem = Path(video_file).stem
    result_path = f'{RESULTS_DIR}/{stem}.json'
    if not os.path.exists(result_path):
        return None
    with open(result_path) as f:
        return json.load(f)


def pearson_ci(r, n, alpha=0.05):
    """Fisher Z-transform for Pearson correlation CI."""
    if n < 4:
        return None, None
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi


def permutation_test(x, y, n_perm=10000):
    """Test if observed correlation is higher than by chance."""
    observed = np.corrcoef(x, y)[0, 1]
    null = np.zeros(n_perm)
    y_copy = y.copy()
    for i in range(n_perm):
        np.random.shuffle(y_copy)
        null[i] = np.corrcoef(x, y_copy)[0, 1]
    p_value = np.mean(np.abs(null) >= np.abs(observed))
    return observed, p_value


def main():
    print('Loading data...')
    perf = load_performance()
    print(f'Found {len(perf)} performance records')

    pairs = []
    for row in perf:
        local_file = row.get('local_file', '')
        if not local_file or local_file == '':
            continue
        ns_report = load_neuroscore(local_file)
        if ns_report is None:
            continue
        try:
            ctr = float(row['ctr']) if row.get('ctr') else None
            likes = float(row['likes']) if row.get('likes') else None
        except (ValueError, TypeError):
            continue
        if ctr is None:
            continue
        pairs.append({
            'video': Path(local_file).stem,
            'brand': row.get('brand', ''),
            'ctr': ctr,
            'likes': likes,
            'comments': float(row['comments']) if row.get('comments') else None,
            'shares': float(row['shares']) if row.get('shares') else None,
            'neuroscore': ns_report['neuroscore'],
            'n_peaks': len(ns_report['peaks']),
            'n_dead_zones': len(ns_report['dead_zones']),
            'total_dead_time': sum(dz['duration'] for dz in ns_report['dead_zones']),
            'duration': ns_report['duration_seconds'],
        })

    print(f'Paired: {len(pairs)} videos\n')
    if len(pairs) < 4:
        print('Need at least 4 paired samples for correlation.')
        sys.exit(1)

    # Filter out NaN and zero-duration failures
    pairs = [p for p in pairs
             if p.get('neuroscore') is not None
             and not np.isnan(p['neuroscore'])
             and p.get('duration', 0) > 0
             and p['neuroscore'] > 0]  # exclude complete failures

    print(f'After filtering failures: {len(pairs)} valid videos\n')
    if len(pairs) < 4:
        print('Not enough valid pairs.')
        sys.exit(1)

    n = len(pairs)
    ns = np.array([p['neuroscore'] for p in pairs])
    ctr = np.array([p['ctr'] for p in pairs])
    likes = np.array([p['likes'] for p in pairs if p['likes'] is not None])
    dead = np.array([p['total_dead_time'] for p in pairs])

    print('=' * 70)
    print(f'SAMPLE SIZE: n = {n}')
    print('=' * 70)

    # Minimum detectable correlation
    min_r = 1.96 / np.sqrt(n - 3)
    print(f'\nMinimum reliably-detectable |r| at this n: ~{min_r:.2f}')
    print(f'(correlations below this could be random noise)\n')

    def report(name, x, y):
        r, p_scipy = stats.pearsonr(x, y)
        lo, hi = pearson_ci(r, len(x))
        r_perm, p_perm = permutation_test(x, y, n_perm=10000)
        verdict = 'STRONG' if abs(r) > 0.6 else ('MODERATE' if abs(r) > 0.3 else 'WEAK')
        signif = 'SIGNIFICANT' if p_perm < 0.05 else 'NOT significant'

        print(f'--- {name} ---')
        print(f'  Pearson r      = {r:.3f}')
        print(f'  95% CI         = [{lo:.3f}, {hi:.3f}]')
        print(f'  p-value (perm) = {p_perm:.4f}   [{signif}]')
        print(f'  Strength       = {verdict}')

        # Rank correlation (Spearman) — more robust
        rho, p_sp = stats.spearmanr(x, y)
        print(f'  Spearman rho   = {rho:.3f}  (rank-based)')
        print()

    report('NeuroScore vs CTR', ns, ctr)
    report('NeuroScore vs Likes', ns[: len(likes)], likes) if len(likes) == n else None
    report('Dead zone time vs CTR', dead, ctr)

    print('=' * 70)
    print('RAW DATA')
    print('=' * 70)
    print(f'{"Video":<15}{"Brand":<20}{"NeuroScore":<12}{"CTR":<8}{"Likes":<10}{"DeadTime":<10}')
    for p in sorted(pairs, key=lambda x: -x['neuroscore']):
        print(f'{p["video"]:<15}{p["brand"][:18]:<20}{p["neuroscore"]:<12.1f}'
              f'{p["ctr"]:<8.2f}{p["likes"] or 0:<10.0f}{p["total_dead_time"]:<10.1f}')

    print('\n' + '=' * 70)
    print('INTERPRETATION')
    print('=' * 70)
    r_ns_ctr = np.corrcoef(ns, ctr)[0, 1]
    if r_ns_ctr > 0.6:
        print('✓ STRONG signal: NeuroScore correlates with CTR. Ship the product.')
    elif r_ns_ctr > 0.3:
        print('○ MODERATE signal: NeuroScore partially predicts CTR.')
        print(f'  Need ~{int(85 / (1 if n < 10 else 1))} more videos to confirm this is real.')
    else:
        print('✗ WEAK/NO signal: Model does not transfer to short-form ads.')
        print('  Consider: fine-tuning with engagement data, or different model.')


if __name__ == '__main__':
    main()
