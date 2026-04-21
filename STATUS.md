# NeuroScore Status

**Last updated:** 2026-04-17

## Current Best Model: v21 (with text)

- **Val loss (MSE):** 0.744
- **R²:** 25.6%
- **Architecture:** 3 specialist models (1D-Conv video/audio + MLP text branch) with subject heads, SWA, Pearson+MSE loss
- **Saved to:** `/workspace/neuroscore/models/v21_cluster{0-2}_swa.pt`
- **Text features:** Qwen2.5-3B per-word contextualized embeddings, 3 layer depths, 2Hz binned
- **Environment:** RunPod RTX 4090 + Cursor

## Progress Summary

Started at R²=5.3% (video-only ridge).
Now at R²=23.3% — **4.4× improvement**.

For context: TRIBE v2 (Meta's full system) gets ~30-40% on similar data.

## Next Immediate Step

**Ensemble 5 models** — train v19 with 5 different random seeds, average predictions. Research says this is the biggest remaining lever. Expected gain: +1-3% R².

## What's Built

- Brain encoder predicts 1000 brain parcels from video+audio features of a movie
- Trained on Algonauts 2025 Movie10 dataset (Bourne, Wolf, Figures, Life movies)
- 4 subjects (sub-01, sub-02, sub-03, sub-05), 184 matched sessions, ~30 hours of data
- Uses DINOv2 (video) + Whisper-large-v3 (audio) as frozen feature extractors

## Next Goal

Reach R²=25-30%, then test on real YouTube ads/videos to launch NeuroScore SaaS.

## Migration Plan

Moving off Google Colab (too expensive, disconnects, Drive issues) to **Cursor + RunPod** for faster iteration at lower cost.
