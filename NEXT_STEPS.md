# Next Steps

## Immediate (when Cursor + RunPod is set up)

### Migration Setup
1. Install Cursor on Mac
2. Create RunPod account, add $20 credit
3. Launch pod: PyTorch 2.x template, RTX 4090 ($0.70/hr) or A100 ($1.79/hr)
4. Connect Cursor via SSH (RunPod gives you SSH command)
5. Transfer data from Google Drive to RunPod persistent volume:
   - `neuroscore/features/video/` (~130 MB)
   - `neuroscore/features/audio/` (~3.5 MB)
   - `neuroscore/data/algonauts_2025.competitors/fmri/` (fMRI h5 files, ~500 MB)
   - `neuroscore/models/` (saved checkpoints, normalization stats)

### Resume Improvement Plan (ordered by expected impact)

**Priority 1 (do first — quick wins):**
- [ ] **Ensemble 5 models** (30 min) — Train v19 with 5 different random seeds, average predictions. Expected: +1-3% R².
- [ ] **Banded ridge baseline** (15 min) — Use himalaya package, per-feature-space regularization. May reveal if we need different feature weighting.

**Priority 2 (medium effort):**
- [ ] **Multi-layer features** (1-2 hrs) — Extract features from multiple DINOv2/Whisper layers, not just last. Research shows brain regions align with different layers.
- [ ] **Contrastive learning** (30 min) — VIBE's innovation: distinguish real from distractor fMRI sequences.
- [ ] **Brain-inspired curriculum** (20 min) — Train on visual cortex first, then harder regions.

**Priority 3 (big effort, big potential):**
- [ ] **Add StudyForrest dataset** (2-3 hrs) — 15 subjects watching full Forrest Gump. Would significantly expand training data.
- [ ] **Try CLIP features** instead of/alongside DINOv2 (1-2 hrs).
- [ ] **V-JEPA 2 video features** (winning Algonauts 2025 used this) — 1-2 hrs if model weights accessible.

## After reaching R²~25-30%

1. **Test on real videos** — YouTube ads, viral content
2. **Build Streamlit app** — upload video, get NeuroScore report
3. **Validation study** — 10 viral vs 10 flop videos, measure brain pattern differences
4. **First clients** — cold outreach to casino/DTC brands

## Known Issues to Remember

- **Text features are broken** — Mistral extraction gives R²=0. Fix requires: multi-layer extraction + proper temporal alignment.
- **Google Drive data copy failures** — symlinks broke, eventually worked by copying h5 files to `/tmp/` before opening.
- **sub-04 is missing** from dataset — we use sub-01, sub-02, sub-03, sub-05.
- **24 video clips had no audio initially** — re-extracted all 34 clips consistently using Whisper-large-v3.
