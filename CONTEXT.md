# NeuroScore Project Context

## What Is This?

NeuroScore is a brain encoding model that predicts fMRI brain activity from video content. The commercial goal: sell "neural engagement reports" to advertisers, casinos, DTC brands, and content creators — predicting how audiences will respond to videos BEFORE spending money on distribution.

## Core Value Proposition

> "Will this video ad work BEFORE I spend money running it?"

- Traditional neuromarketing: $15K-50K per study, 2-6 weeks, requires lab + human subjects
- NeuroScore: $300-5000 per analysis, same day, purely computational

## Technical Approach

Replicate Meta's TRIBE v2 architecture with commercial-friendly components:
- **Video:** DINOv2 (Apache 2.0) — frozen feature extractor, 1024-dim
- **Audio:** Whisper-large-v3 (MIT) — frozen feature extractor, 1280-dim
- **Text:** Mistral 7B (Apache 2.0) — frozen feature extractor, 4096-dim (currently not working)
- **Brain encoder:** Trainable neural network predicting Schaefer 1000-parcel fMRI

## Data Pipeline

1. Extract features from frozen models → cache as .npy
2. Match features to fMRI sessions via movie clip name (bourne01, wolf03, etc.)
3. Interpolate features to fMRI TR=1.49s timepoints
4. Apply HRF delay shift (or multi-delay FIR)
5. Z-score normalize
6. Build temporal windows (5 frames × 5 delays = 25 time steps per sample)
7. Train with Pearson+MSE loss, subject-specific heads, SWA
8. Cluster brain parcels into 3 networks, train specialist models per cluster

## Current Architecture (v19)

```
Input: 5 video frames × 5 HRF delays × 1024 dims (video)
     + 5 audio frames × 5 HRF delays × 1280 dims (audio)

     ↓
Reshape to (batch, 25 time steps, 2304 features per step)

     ↓
1D Conv (2304 → 256, kernel=3, padding=1) + GELU + LayerNorm + Dropout 0.2
1D Conv (256 → 256, kernel=3, padding=1) + GELU + LayerNorm + Dropout 0.2
1D Conv (256 → 256, kernel=3, padding=1) + GELU + LayerNorm

     ↓
Global average pool over time → (batch, 256)

     ↓
Subject-specific head per subject (4 × Linear(256 → N_parcels_in_cluster))

     ↓
Output: 296 or 354 or 350 brain parcels (one specialist per cluster)
```

Three specialists are trained separately (one per parcel cluster), their predictions concatenated to form full 1000-parcel prediction.

## Key Numbers

- Training data: ~30 hours of fMRI
- 184 matched video↔fMRI sessions (46 per subject × 4 subjects)
- 73,120 training samples (after temporal windowing)
- Train/val split: 90/10
- Batch size: 64
- Epochs: 100 with early stopping patience=15, SWA from epoch 50

## User Background

- Not a coder — Claude writes all code, user pastes into Colab
- Has statistics background (understands R², MSE, correlation, distributed lag models)
- Frustrated with Colab's costs/disconnects, ready to migrate to Cursor + RunPod
- Goal: build NeuroScore into a profitable SaaS

## Environment (Current: Colab → Moving to RunPod)

**Current (Colab):**
- Google Drive: `/content/drive/MyDrive/neuroscore/`
- Two notebooks: Notebook 02 (feature extraction), Notebook 03 (training)
- T4 or A100 GPU rented per session

**Target (RunPod):**
- Persistent volume with all data on pod
- Cursor IDE with SSH connection
- Claude can directly edit and run code
- RTX 4090 at $0.70/hr or A100 at $1.79/hr

## Files on Google Drive (to migrate)

```
neuroscore/
├── data/
│   └── algonauts_2025.competitors/
│       ├── fmri/              # fMRI h5 files per subject
│       └── stimuli/movies/movie10/  # Source .mkv files
├── features/
│   ├── video/                 # DINOv2 features (.npy, readable names)
│   ├── audio/                 # Whisper features (.npy, readable names)
│   └── text/                  # Mistral features (.npy, movie10_*.npy)
├── models/
│   ├── best_model_v17.pt      # Best single model checkpoint
│   ├── parcel_clusters.npy    # 3-cluster parcel assignments
│   ├── video_mean.npy, video_std.npy
│   ├── audio_mean.npy, audio_std.npy
│   └── fmri_mean.npy, fmri_std.npy
└── notebooks/                 # Jupyter notebooks
```
