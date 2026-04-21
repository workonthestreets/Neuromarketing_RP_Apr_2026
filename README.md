# NeuroScore Project

## Files

| File | What's in it |
|------|--------------|
| **STATUS.md** | Current best model, where we are, next immediate step |
| **METHODS_TRIED.md** | Full history of everything tried with results |
| **NEXT_STEPS.md** | Migration plan to Cursor+RunPod + remaining experiments to try |
| **CONTEXT.md** | Project overview, architecture details, data pipeline |
| **CELLS_v19.md** | Working cells for the current best model (R²=23.3%) |
| **RUNPOD_SETUP.md** | Setup guide for Cursor + RunPod |
| **TEXT_FEATURES_TODO.md** | Future task — correct pipeline for text feature extraction |

## Quick Status

- **Best R²:** 23.3% (v19)
- **Started at:** 5.3% (v1)
- **Target:** 25-30% before launching product
- **Next step:** Migrate to Cursor+RunPod, then run ensemble of 5 models

## Data location

Large artifacts are **not** stored in this repository. They live on the RunPod
workspace at `/workspace/neuroscore/` on our training pod and are excluded via
`.gitignore`:

| Path on RunPod | Contents | Approx size |
|---|---|---|
| `/workspace/neuroscore/cache/` | Cached DINOv2/Whisper features | ~18 GB |
| `/workspace/neuroscore/features/` | Extracted per-session feature tensors | ~1.3 GB |
| `/workspace/neuroscore/models/` | Trained checkpoints (`v19`, `v21_cluster{0-2}_swa.pt`, etc.) | ~282 MB |
| `/workspace/neuroscore/algonauts_2025.competitors/` | Cloned external reference repo | ~2.6 GB |
| `/workspace/neuroscore/tiktok_ads/` | Scraped TikTok ad media + metadata | ~96 MB |
| `/workspace/neuroscore/video.mp4` | Reference inference video | 24 MB |

Secrets (e.g. `youtube_cookies.txt`) are also kept only on the pod and must
never be committed.

To reproduce locally, rehydrate caches/features by re-running the extraction
scripts against the Algonauts 2025 dataset, or rsync from the pod:

```
rsync -av -e "ssh -p <PORT> -i <KEY>" \
    root@<POD_IP>:/workspace/neuroscore/{models,features}/ ./
```
