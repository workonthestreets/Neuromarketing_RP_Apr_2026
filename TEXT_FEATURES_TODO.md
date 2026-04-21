# Text Features — Future Task

## Why This Matters
TRIBE/VIBE research shows text is one of the strongest brain predictors. Our current extraction gives R²=0. Fix could give us 3-8% more R².

## Why Our Current Extraction Fails (ranked)

1. **No timestamp alignment** — one embedding per whole transcript gives constant features per TR
2. **Wrong pooling** — mean-pool over entire chunk destroys word-level info
3. **Wrong layer** — used last layer of decoder LM; brain alignment peaks mid-to-3/4 depth
4. **No HRF delay** — text at time t predicts BOLD at t+4-6s
5. **Context rebuilt per TR, not per word** — loses contextualization

## Correct Pipeline (TRIBE's Actual Recipe)

**Model:** `meta-llama/Llama-3.2-3B` (not Mistral 7B)

**Per-word extraction:**
1. For each word w in transcript, build context = preceding 1024 words + w
2. Tokenize with `truncation_side="left"`, `add_special_tokens=False`
3. Forward pass, extract hidden states at layers `[n//2, 3n//4, n]`
4. Take mean across only the subword tokens that make up w
5. Result: contextualized word embedding per word

**Layer aggregation:**
- Extract 3 layer groups (at 50%, 75%, 100% depth)
- Group-mean aggregate → 3 grouped features, 2048 dim each

**Temporal binning:**
1. Resample to 2 Hz grid (0.5s bins), sum word vectors in each bin
2. Shape: `(T_bins, 3_layers, 2048)` → flatten to `(T_bins, 6144)`
3. Resample 2 Hz → TR 1.49s with mean downsampling
4. Add FIR delays of 2, 4, 6, 8 seconds before model

## Data Source

**Algonauts provides pre-aligned transcripts as TSV files** in `stimuli/transcripts/` with columns:
- `words_per_tr`
- `onsets_per_tr`
- `durations_per_tr`

These are already word-timestamp-aligned — no Whisper ASR needed.

## Implementation Reference

- TRIBE's official code: https://github.com/facebookresearch/algonauts-2025/blob/main/data_utils/data_utils/features/text.py
- The `LLAMA3p2` class has the complete recipe

## When to Do This

After completing current priority list:
- ✅ Current: ensemble of v19
- Banded ridge
- Multi-layer video/audio features
- **THEN: text features with correct pipeline**
- StudyForrest dataset

Expected improvement: R² could jump from 23.3% to 27-31% if text is done right.
