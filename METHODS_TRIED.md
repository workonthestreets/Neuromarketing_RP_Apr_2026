# Methods Tried — Full History

## Results Summary Table

| Version | Method | Val loss | R² | Kept? |
|---------|--------|----------|-----|-------|
| v1 | Video only, MSE | 0.947 | 5.3% | baseline |
| v3 | + Audio, HRF shift, temporal window, 68 sessions | 0.882 | 11.8% | — |
| v7 | + 184 sessions (no subject heads) | 0.889 | 11.1% | no (worse) |
| v9 | + Subject-specific heads | 0.795 | 20.5% | yes |
| v10 | Dropout 0.3 | 0.809 | 19.1% | no (worse) |
| v11 | Dropout 0.25 | 0.799 | 20.1% | no (worse, keep 0.2) |
| v12 | + Text features | 0.798 | 20.2% | no (text=noise) |
| v13 | Pearson + MSE loss | 0.792 | 20.8% | yes |
| v14 | + FIR multi-delay (5 HRF delays) | 0.790 | 21.0% | yes |
| v15 | + Modality dropout 0.2 | 0.794 | 20.6% | no (worse) |
| v16 | + Bidirectional context | 0.791 | 20.9% | no (worse) |
| v17 | + SWA (stochastic weight averaging) | 0.786 | 21.4% | yes |
| v18 | + Per-network specialists (3 MLP models) | 0.780 | 22.0% | yes |
| **v19** | **+ 1D-Conv specialists** | **0.767** | **23.3%** | **yes — BEST** |
| v20 | Subject embeddings (replace heads) | 0.810 | — | no (worse, stopped early) |

## Detailed Notes on Each Method

### WORKED

**Subject-specific heads (v9):** Huge win, 11%→20.5%. Each of 4 subjects gets own projection layer.

**Pearson+MSE loss (v13):** VIBE's combined loss: `L = L_Pearson + 0.03 × L_MSE`. Trains faster, better final R².

**FIR multi-delay (v14):** Instead of single HRF shift (3 TRs), include features at 5 delays: [0, 1, 2, 3, 4] TRs. Each brain region learns its own optimal delay.

**Stochastic Weight Averaging (v17):** Average model weights from epochs 50-100. ~0.4% R² gain for free.

**Per-network specialists (v18):** Cluster 1000 parcels into 3 networks using KMeans on response patterns. Train a specialist model per cluster. +0.6% R².

**1D temporal convolution (v19):** Replace flat MLP input with conv over time (25 time steps = 5 frames × 5 delays). +1.3% R². Biggest architectural win.

**Gradient clipping (5.0):** Added with Pearson loss for training stability.

**Z-score normalization:** Standard across all features and fMRI targets.

### DIDN'T WORK

**Ridge regression:** R²=5.3%. Brain mapping is non-linear, simple regression fails.

**Temporal transformer (8-head, 3-layer):** R²≈0.9%. Too complex for only 5 frames, spent capacity on attention mechanics.

**Mistral text features:** R²=20.2% with subject heads. Text-only ridge: R²=-0.4%. Our text extraction is flawed. Could work with better method (layer grouping, proper timing).

**Modality dropout (TRIBE's p=0.2):** Slightly worse. Works for TRIBE's 3 modalities, hurts our 2.

**Bidirectional context (2 past + current + 2 future):** Slightly worse. VIBE saw only 0.002 gain; we lost more data at edges.

**Higher dropout (0.25, 0.3):** Closes train-val gap but hurts overall performance.

**Subject embeddings (v20):** Significantly worse than subject heads. With only 4 subjects, separate heads have enough capacity and specialize better.

**Adding more data (68→184 sessions without subject heads):** Hurt performance until we added subject heads.

## Key Insights

1. **Architecture matters less than ensembling** (from Algonauts 2025 insights paper)
2. **Text was strongest predictor in research, zero in our setup** — our extraction method is broken
3. **Subject-specific learning is critical** — each brain is wired differently
4. **Temporal structure matters** — conv beats MLP; FIR delays capture HRF variation
5. **Simple often wins** — 1D conv beat temporal transformer; per-parcel-cluster specialists beat single unified model
