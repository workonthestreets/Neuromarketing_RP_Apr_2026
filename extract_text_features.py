"""
Extract text features from Algonauts Movie10 transcripts using TRIBE's method:
- Per-word contextualized embeddings (1024-word left context)
- Hidden states from 3 layer depths (50%, 75%, 100%)
- Bin to 2Hz grid using word onsets

Uses Qwen2.5-3B (open, no HF gating needed).
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
BASE = '/workspace/neuroscore'
TRANSCRIPTS_DIR = f'{BASE}/algonauts_2025.competitors/stimuli/transcripts/movie10'
FEATURES_OUT = f'{BASE}/features/text'
os.makedirs(FEATURES_OUT, exist_ok=True)

MODEL_NAME = 'Qwen/Qwen2.5-3B'
CONTEXT_WORDS = 1024
TR = 1.49
BIN_HZ = 2.0  # bin to 2Hz grid before resampling to TR

# Brain encoding research shows mid-to-late layers work best
# Qwen2.5-3B has 36 layers. We extract at 18 (50%), 27 (75%), 36 (100%)


def parse_list_cell(cell):
    """Parse TSV cell that contains a Python list literal (e.g., "['word1', 'word2']")."""
    if pd.isna(cell) or cell == '' or cell == '[]':
        return []
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        # Some cells are double-quoted; try cleaning
        try:
            return ast.literal_eval(cell.replace('""', '"').strip('"'))
        except:
            return []


def load_transcript(tsv_path):
    """Load TSV and return list of (word, onset) tuples in order."""
    df = pd.read_csv(tsv_path, sep='\t')
    all_words = []
    all_onsets = []
    for _, row in df.iterrows():
        words = parse_list_cell(row['words_per_tr'])
        onsets = parse_list_cell(row['onsets_per_tr'])
        if len(words) != len(onsets):
            continue
        all_words.extend(words)
        all_onsets.extend(onsets)
    return all_words, all_onsets


def extract_features_for_clip(words, onsets, tokenizer, model, device, layer_indices):
    """
    For each word, compute contextualized embedding.
    Returns: (n_words, len(layer_indices) * hidden_dim) embeddings + onsets array.
    """
    if not words:
        return np.zeros((0, len(layer_indices) * model.config.hidden_size), dtype=np.float32), np.array([])

    hidden_size = model.config.hidden_size
    n_layers_extract = len(layer_indices)
    word_embeddings = np.zeros((len(words), n_layers_extract * hidden_size), dtype=np.float32)

    # Process in batches for speed; but need per-word target extraction
    for i in tqdm(range(len(words)), desc='  words', leave=False):
        context_start = max(0, i - CONTEXT_WORDS)
        context = ' '.join(words[context_start:i + 1])  # preceding + current
        target_word = words[i]

        # Tokenize full context
        inputs = tokenizer(
            context,
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=2048,
        ).to(device)

        # Tokenize just the target word to know how many tokens it uses
        target_tokens = tokenizer(target_word, add_special_tokens=False).input_ids
        n_target_tokens = len(target_tokens)
        if n_target_tokens == 0:
            n_target_tokens = 1

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) × (1, seq_len, hidden_dim)

        # For each requested layer, mean over target's subword tokens (last N tokens in sequence)
        layer_feats = []
        for layer_idx in layer_indices:
            h = hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
            target_h = h[-n_target_tokens:].mean(dim=0)  # (hidden_dim,)
            layer_feats.append(target_h.cpu().numpy())

        word_embeddings[i] = np.concatenate(layer_feats)

    return word_embeddings, np.array(onsets)


def bin_to_2hz(word_embeddings, onsets, total_duration):
    """Sum word embeddings into 0.5s bins."""
    n_bins = int(np.ceil(total_duration * BIN_HZ))
    binned = np.zeros((n_bins, word_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)

    for i, t in enumerate(onsets):
        bin_idx = int(t * BIN_HZ)
        if 0 <= bin_idx < n_bins:
            binned[bin_idx] += word_embeddings[i]
            counts[bin_idx] += 1

    return binned  # sums — better for brain encoding per TRIBE


def main():
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Loading {MODEL_NAME}...')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()
    print(f'Model loaded. {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}')

    # Choose 3 layer depths
    n_layers = model.config.num_hidden_layers
    layer_indices = [n_layers // 2, (3 * n_layers) // 4, n_layers]
    print(f'Extracting from layers: {layer_indices}')

    movies = ['bourne', 'wolf', 'figures', 'life']

    for movie in movies:
        movie_dir = f'{TRANSCRIPTS_DIR}/{movie}'
        if not os.path.exists(movie_dir):
            continue
        tsv_files = sorted([f for f in os.listdir(movie_dir) if f.endswith('.tsv')])

        for tsv_file in tsv_files:
            # movie10_bourne01.tsv → bourne01
            clip_name = tsv_file.replace('movie10_', '').replace('.tsv', '')
            out_file = f'{FEATURES_OUT}/{clip_name}.npy'
            if os.path.exists(out_file):
                print(f'  {clip_name}: already done, skipping')
                continue

            print(f'\nProcessing {clip_name}...')
            words, onsets = load_transcript(f'{movie_dir}/{tsv_file}')
            print(f'  {len(words)} words')

            if not words:
                print(f'  No words, saving empty')
                np.save(out_file, np.zeros((0, len(layer_indices) * model.config.hidden_size), dtype=np.float32))
                continue

            word_embs, onset_arr = extract_features_for_clip(
                words, onsets, tokenizer, model, device, layer_indices
            )

            # Total duration from the TSV (each row = 1 TR)
            n_trs = sum(1 for _ in open(f'{movie_dir}/{tsv_file}')) - 1
            total_duration = n_trs * TR

            binned_2hz = bin_to_2hz(word_embs, onset_arr, total_duration)

            np.save(out_file, binned_2hz)
            print(f'  Saved {out_file}: shape {binned_2hz.shape}')


if __name__ == '__main__':
    main()
