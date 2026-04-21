"""
NeuroScore inference pipeline.

Usage:
    python inference.py <URL_or_video_path>

Output: NeuroScore report + timeline saved to results/

Pipeline:
  URL → yt-dlp → video.mp4
     → DINOv2 video features (1Hz)
     → Whisper audio features
     → Qwen2.5-3B transcript embeddings (optional if dialogue)
     → FIR multi-delay + temporal window
     → v21 3-specialist model inference (averaged across 4 subject heads)
     → Aggregate to NeuroScore + timeline + peaks + dead zones
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d

# ============================================================
# Paths
# ============================================================
BASE = '/workspace/neuroscore'
MODELS_DIR = f'{BASE}/models'
RESULTS_DIR = f'{BASE}/results'
TMP_DIR = f'{BASE}/tmp_inference'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# ============================================================
# Config (must match training)
# ============================================================
TR = 1.49
HRF_DELAYS = [0, 1, 2, 3, 4]
WINDOW_SIZE = 5
N_SUBJECTS = 4
N_PARCELS = 1000
N_CLUSTERS = 3
HIDDEN_DIM = 256
TEXT_DIM = 6144

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Model (must match train_v21_with_text.py)
# ============================================================
class BrainEncoderWithText(nn.Module):
    def __init__(self, video_dim, audio_dim, text_dim, fmri_dim,
                 n_subjects=N_SUBJECTS, hidden_dim=HIDDEN_DIM,
                 window_size=WINDOW_SIZE, n_delays=len(HRF_DELAYS)):
        super().__init__()
        self.window_size = window_size
        self.n_delays = n_delays
        self.video_per_frame = video_dim // (window_size * n_delays)
        self.audio_per_frame = audio_dim // (window_size * n_delays)
        frame_dim = self.video_per_frame + self.audio_per_frame

        self.conv1 = nn.Conv1d(frame_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.subject_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, fmri_dim) for _ in range(n_subjects)
        ])

    def forward(self, video, audio, text, subject_idx):
        B = video.shape[0]
        T = self.window_size * self.n_delays
        v = video.view(B, T, self.video_per_frame)
        a = audio.view(B, T, self.audio_per_frame)
        x = torch.cat([v, a], dim=-1).transpose(1, 2)

        x = self.gelu(self.conv1(x))
        x = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.gelu(self.conv2(x))
        x = self.ln2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.gelu(self.conv3(x))
        x = self.ln3(x.transpose(1, 2)).transpose(1, 2)
        conv_pool = x.mean(dim=2)

        text_feat = self.text_proj(text)
        combined = torch.cat([conv_pool, text_feat], dim=-1)

        out = torch.zeros(B, self.subject_heads[0].out_features, device=video.device)
        for i in range(len(self.subject_heads)):
            mask = (subject_idx == i)
            if mask.any():
                out[mask] = self.subject_heads[i](combined[mask])
        return out


# ============================================================
# Step 1: Download video
# ============================================================
def download_video(url_or_path, out_path):
    if os.path.exists(url_or_path):
        # Local file — just use as-is
        return url_or_path

    # Clear any cached download
    for ext in ['.mp4', '.webm', '.mkv', '.part']:
        stale = out_path.replace('.mp4', ext)
        if os.path.exists(stale):
            os.remove(stale)
    # Also clear yt-dlp's fragment files
    for f in os.listdir(os.path.dirname(out_path)):
        if f.startswith(os.path.basename(out_path).replace('.mp4', '')):
            os.remove(os.path.join(os.path.dirname(out_path), f))

    print(f'Downloading: {url_or_path}')
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        '--merge-output-format', 'mp4',
        '--remote-components', 'ejs:github',
        '--force-overwrites',
        '-o', out_path,
    ]

    # Use cookies if available (YouTube bot protection)
    cookies_path = f'{BASE}/youtube_cookies.txt'
    if os.path.exists(cookies_path):
        cmd += ['--cookies', cookies_path]
        print(f'  Using cookies from {cookies_path}')

    cmd.append(url_or_path)
    try:
        subprocess.run(cmd, check=True)
        return out_path
    except FileNotFoundError:
        print('yt-dlp not installed. Run: pip install yt-dlp')
        sys.exit(1)


# ============================================================
# Step 2: Extract DINOv2 video features (1Hz)
# ============================================================
def extract_video_features(video_path):
    print('Extracting video features (DINOv2)...')
    from transformers import AutoModel, AutoImageProcessor
    from PIL import Image

    # Load DINOv2
    model_name = 'facebook/dinov2-large'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    # Extract frames at 1 Hz using ffmpeg (recreate dir fresh each call)
    import shutil as _shutil
    frames_dir = f'{TMP_DIR}/frames'
    if os.path.exists(frames_dir):
        _shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vf', 'fps=1', '-q:v', '2',
        f'{frames_dir}/frame_%04d.jpg'
    ], capture_output=True)

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frame_files:
        raise RuntimeError('No frames extracted')
    print(f'  Got {len(frame_files)} frames')

    features = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i + batch_size]
            imgs = [Image.open(f'{frames_dir}/{f}').convert('RGB') for f in batch_files]
            inputs = processor(images=imgs, return_tensors='pt').to(DEVICE)
            outputs = model(**inputs)
            # CLS token embedding
            pooled = outputs.last_hidden_state[:, 0, :]
            features.append(pooled.cpu().numpy())

    features = np.concatenate(features, axis=0)  # (N, 1024)

    # DINOv2-large has 1024 hidden dim; we trained on 1024
    if features.shape[1] != 1024:
        raise RuntimeError(f'Expected 1024 dim, got {features.shape[1]}')

    # Cleanup
    subprocess.run(f'rm -rf {frames_dir}', shell=True)
    del model
    torch.cuda.empty_cache()
    return features.astype(np.float32)


# ============================================================
# Step 3: Extract Whisper audio features
# ============================================================
def extract_audio_features(video_path):
    print('Extracting audio features (Whisper)...')
    from transformers import WhisperModel, WhisperFeatureExtractor
    import torchaudio

    # Extract audio to wav
    wav_path = f'{TMP_DIR}/audio.wav'
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-ac', '1', '-ar', '16000', '-vn', wav_path
    ], capture_output=True)

    if not os.path.exists(wav_path):
        # Silent video: return zeros
        print('  No audio extracted — using zeros')
        return None

    # Load Whisper
    model_name = 'openai/whisper-large-v3'
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    whisper = WhisperModel.from_pretrained(model_name).encoder.to(DEVICE).float().eval()

    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.squeeze().numpy()

    chunk_size = 30 * sr
    features_list = []
    for start in range(0, len(waveform), chunk_size):
        chunk = waveform[start:start + chunk_size]
        if len(chunk) < sr:
            continue
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors='pt')
        input_features = inputs.input_features.to(DEVICE)
        with torch.no_grad():
            out = whisper(input_features).last_hidden_state
            pooled = out.mean(dim=1)
            features_list.append(pooled.cpu().numpy())

    if not features_list:
        return None

    features = np.concatenate(features_list, axis=0)  # (N_chunks, 1280)
    os.remove(wav_path)
    del whisper
    torch.cuda.empty_cache()
    return features.astype(np.float32)


# ============================================================
# Step 4: Extract Qwen2.5-3B text features (optional)
# ============================================================
def extract_text_features_from_audio(wav_path, video_duration):
    """Transcribe audio with Whisper, then embed with Qwen2.5-3B."""
    print('Extracting text features (Whisper transcribe + Qwen embed)...')
    from transformers import AutoTokenizer, AutoModel, WhisperProcessor, WhisperForConditionalGeneration

    # First transcribe with Whisper
    if not os.path.exists(wav_path):
        return np.zeros((int(video_duration * 2), TEXT_DIM), dtype=np.float32)

    whisper_proc = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
    whisper_asr = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3').to(DEVICE).float().eval()

    import torchaudio
    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.squeeze().numpy()

    # Transcribe with word-level timestamps
    inputs = whisper_proc(waveform, sampling_rate=sr, return_tensors='pt').input_features.to(DEVICE)
    with torch.no_grad():
        pred_ids = whisper_asr.generate(inputs, return_timestamps=True, language='en')
    transcription = whisper_proc.batch_decode(pred_ids, skip_special_tokens=True)[0]

    del whisper_asr, whisper_proc
    torch.cuda.empty_cache()

    words = transcription.split()
    if not words:
        return np.zeros((int(video_duration * 2), TEXT_DIM), dtype=np.float32)

    # Distribute words evenly across video duration (approximation)
    onsets = np.linspace(0, video_duration, len(words))

    # Qwen2.5-3B embeddings (same as training extraction)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    qwen = AutoModel.from_pretrained('Qwen/Qwen2.5-3B', torch_dtype=torch.float16).to(DEVICE).eval()
    n_layers = qwen.config.num_hidden_layers
    layer_indices = [n_layers // 2, (3 * n_layers) // 4, n_layers]

    hidden_size = qwen.config.hidden_size
    word_embs = np.zeros((len(words), 3 * hidden_size), dtype=np.float32)
    CONTEXT_WORDS = 1024

    for i in range(len(words)):
        context_start = max(0, i - CONTEXT_WORDS)
        context = ' '.join(words[context_start:i + 1])
        target_tokens = tokenizer(words[i], add_special_tokens=False).input_ids
        n_tgt = max(1, len(target_tokens))
        inputs = tokenizer(context, return_tensors='pt', add_special_tokens=False,
                           truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out = qwen(**inputs, output_hidden_states=True)
        hs = out.hidden_states
        feats = []
        for li in layer_indices:
            h = hs[li][0][-n_tgt:].mean(dim=0)
            feats.append(h.cpu().numpy())
        word_embs[i] = np.concatenate(feats)

    del qwen, tokenizer
    torch.cuda.empty_cache()

    # Bin to 2Hz
    n_bins = max(1, int(video_duration * 2))
    binned = np.zeros((n_bins, word_embs.shape[1]), dtype=np.float32)
    for i, t in enumerate(onsets):
        bin_idx = min(int(t * 2), n_bins - 1)
        binned[bin_idx] += word_embs[i]

    return binned


# ============================================================
# Step 5: Build FIR multi-delay + temporal window samples
# ============================================================
def build_samples(video_feats, audio_feats, text_feats, norm_stats):
    """
    video_feats: (N_video, 1024) at 1Hz
    audio_feats: (N_audio, 1280) — sparse
    text_feats:  (N_text, 6144) at 2Hz (or None)

    Returns samples aligned to fMRI-like timepoints (TR=1.49s)
    """
    print('Building FIR multi-delay samples...')
    video_mean = np.load(f'{MODELS_DIR}/video_mean.npy')
    video_std = np.load(f'{MODELS_DIR}/video_std.npy')
    audio_mean = np.load(f'{MODELS_DIR}/audio_mean.npy')
    audio_std = np.load(f'{MODELS_DIR}/audio_std.npy')
    text_mean = np.load(f'{MODELS_DIR}/text_mean.npy')
    text_std = np.load(f'{MODELS_DIR}/text_std.npy')

    # Time axes
    video_times = np.arange(video_feats.shape[0]) * 1.0
    max_time = video_times[-1]
    if audio_feats is not None:
        audio_times = np.linspace(0, max_time, audio_feats.shape[0])
    else:
        audio_feats = np.zeros((2, 1280), dtype=np.float32)
        audio_times = np.array([0, max_time])

    # Create fMRI-like time grid (TR-spaced, matches training)
    n_tr = int(max_time / TR) + 1
    tr_times = np.arange(n_tr) * TR
    tr_mask = tr_times <= max_time
    tr_times = tr_times[tr_mask]

    # Interpolate video and audio to TR grid
    video_tr = interp1d(video_times, video_feats, axis=0, fill_value='extrapolate')(tr_times)
    audio_tr = interp1d(audio_times, audio_feats, axis=0, fill_value='extrapolate')(tr_times)

    # Build FIR windows
    max_delay = max(HRF_DELAYS)
    start_idx = WINDOW_SIZE - 1 + max_delay
    n = len(tr_times)

    n_possible = max(0, n - start_idx)
    print(f'  Evaluation points: {n_possible} (video {max_time:.1f}s)')
    if n_possible < 3:
        print(f'  WARNING: Only {n_possible} evaluation points. Video may be too short for reliable scoring.')

    all_video_samples = []
    all_audio_samples = []
    all_text_samples = []
    sample_times = []

    for i in range(start_idx, n):
        v_parts, a_parts = [], []
        for delay in HRF_DELAYS:
            center = i - delay
            v_parts.append(video_tr[center - WINDOW_SIZE + 1: center + 1].flatten())
            a_parts.append(audio_tr[center - WINDOW_SIZE + 1: center + 1].flatten())
        all_video_samples.append(np.concatenate(v_parts))
        all_audio_samples.append(np.concatenate(a_parts))

        # Text: mean over recent window
        if text_feats is not None and text_feats.shape[0] > 0:
            t_target = tr_times[i]
            bin_end = int(t_target * 2)
            bin_start = max(0, bin_end - int(max_delay * TR * 2))
            if bin_start < text_feats.shape[0]:
                bin_end_c = min(bin_end + 1, text_feats.shape[0])
                if bin_end_c > bin_start:
                    all_text_samples.append(text_feats[bin_start:bin_end_c].mean(axis=0))
                else:
                    all_text_samples.append(np.zeros(TEXT_DIM, dtype=np.float32))
            else:
                all_text_samples.append(np.zeros(TEXT_DIM, dtype=np.float32))
        else:
            all_text_samples.append(np.zeros(TEXT_DIM, dtype=np.float32))

        sample_times.append(tr_times[i])

    video_samples = np.array(all_video_samples, dtype=np.float32)
    audio_samples = np.array(all_audio_samples, dtype=np.float32)
    text_samples = np.array(all_text_samples, dtype=np.float32)
    sample_times = np.array(sample_times)

    # Safety: clip std to avoid division by zero or tiny values
    video_std_safe = np.where(video_std < 1e-6, 1.0, video_std)
    audio_std_safe = np.where(audio_std < 1e-6, 1.0, audio_std)
    text_std_safe = np.where(text_std < 1e-6, 1.0, text_std)

    video_samples = (video_samples - video_mean) / video_std_safe
    audio_samples = (audio_samples - audio_mean) / audio_std_safe
    text_samples = (text_samples - text_mean) / text_std_safe

    # Replace any NaN/inf from interpolation edge cases
    video_samples = np.nan_to_num(video_samples, nan=0.0, posinf=0.0, neginf=0.0)
    audio_samples = np.nan_to_num(audio_samples, nan=0.0, posinf=0.0, neginf=0.0)
    text_samples = np.nan_to_num(text_samples, nan=0.0, posinf=0.0, neginf=0.0)

    return video_samples, audio_samples, text_samples, sample_times


# ============================================================
# Step 6: Run v21 models
# ============================================================
def run_inference(video_samples, audio_samples, text_samples):
    print('Running v21 ensemble...')
    parcel_clusters = np.load(f'{MODELS_DIR}/parcel_clusters.npy')
    fmri_mean = np.load(f'{MODELS_DIR}/fmri_mean.npy')
    fmri_std = np.load(f'{MODELS_DIR}/fmri_std.npy')

    video_dim = video_samples.shape[1]
    audio_dim = audio_samples.shape[1]

    video_t = torch.from_numpy(video_samples).to(DEVICE)
    audio_t = torch.from_numpy(audio_samples).to(DEVICE)
    text_t = torch.from_numpy(text_samples).to(DEVICE)

    n_samples = video_samples.shape[0]
    all_preds = np.zeros((n_samples, N_PARCELS), dtype=np.float32)

    for cluster_id in range(N_CLUSTERS):
        parcel_mask = parcel_clusters == cluster_id
        parcel_indices = np.where(parcel_mask)[0]
        n_parcels_c = int(parcel_mask.sum())

        model = BrainEncoderWithText(video_dim, audio_dim, TEXT_DIM, n_parcels_c).to(DEVICE)
        model.load_state_dict(torch.load(f'{MODELS_DIR}/v21_cluster{cluster_id}_swa.pt', map_location=DEVICE))
        model.eval()

        # Average predictions across all 4 subject heads
        preds_per_subject = []
        with torch.no_grad():
            for subj in range(N_SUBJECTS):
                subj_t = torch.full((n_samples,), subj, dtype=torch.long, device=DEVICE)
                pred = model(video_t, audio_t, text_t, subj_t).cpu().numpy()
                preds_per_subject.append(pred)
        avg_pred = np.mean(preds_per_subject, axis=0)  # (n_samples, n_parcels_c)

        all_preds[:, parcel_indices] = avg_pred
        del model
        torch.cuda.empty_cache()

    # Un-normalize to get back to original fMRI scale
    all_preds_unscaled = all_preds * fmri_std + fmri_mean
    return all_preds_unscaled


# ============================================================
# Step 7: Interpret predictions → NeuroScore
# ============================================================
def interpret(brain_preds, sample_times):
    """
    brain_preds: (n_samples, 1000) — predicted brain activity over time
    sample_times: (n_samples,) — time in seconds

    Returns dict with NeuroScore + timeline.
    """
    print('Interpreting...')
    # Check for NaN in predictions
    n_nan_rows = int(np.isnan(brain_preds).any(axis=1).sum())
    if n_nan_rows > 0:
        print(f'  WARNING: {n_nan_rows}/{len(brain_preds)} rows contain NaN. Replacing with row mean.')
        nan_mask = np.isnan(brain_preds)
        # Replace NaN with column mean of non-NaN values
        col_means = np.nanmean(brain_preds, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        brain_preds = np.where(nan_mask, col_means[np.newaxis, :], brain_preds)

    # Mean activation per timepoint (across all parcels)
    mean_activation = brain_preds.mean(axis=1)

    # Drop any remaining NaN
    valid = ~np.isnan(mean_activation)
    mean_activation = mean_activation[valid]
    sample_times = np.array(sample_times)[valid]

    if len(mean_activation) < 2:
        return {
            'neuroscore': 0.0,
            'timeline': [],
            'peaks': [],
            'dead_zones': [],
            'duration_seconds': 0.0,
            'n_samples': 0,
            'error': 'insufficient valid predictions',
        }

    # Normalize to 0-100 scale using percentile scaling
    p5, p95 = np.percentile(mean_activation, [5, 95])
    if p95 - p5 < 1e-4:
        timeline = np.full_like(mean_activation, 50.0)
    else:
        timeline = np.clip((mean_activation - p5) / (p95 - p5) * 100, 0, 100)

    # Overall NeuroScore: average of timeline
    neuroscore = float(np.nanmean(timeline))

    # Peak moments (top 3 timestamps)
    peak_indices = np.argsort(timeline)[-3:][::-1]
    peaks = [{'time': float(sample_times[i]), 'score': float(timeline[i])}
             for i in peak_indices]

    # Dead zones: contiguous windows below 30th percentile
    threshold = np.percentile(timeline, 30)
    dead_zones = []
    in_dead = False
    start = None
    for i, score in enumerate(timeline):
        if score < threshold and not in_dead:
            start = sample_times[i]
            in_dead = True
        elif score >= threshold and in_dead:
            duration = sample_times[i] - start
            if duration >= 3.0:
                dead_zones.append({'start': float(start), 'end': float(sample_times[i]), 'duration': float(duration)})
            in_dead = False
    if in_dead and start is not None:
        duration = sample_times[-1] - start
        if duration >= 3.0:
            dead_zones.append({'start': float(start), 'end': float(sample_times[-1]), 'duration': float(duration)})

    return {
        'neuroscore': round(neuroscore, 1),
        'timeline': [{'time': float(t), 'score': float(s)} for t, s in zip(sample_times, timeline)],
        'peaks': peaks,
        'dead_zones': dead_zones,
        'duration_seconds': float(sample_times[-1]),
        'n_samples': len(timeline)
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Video URL or local file path')
    parser.add_argument('--no-text', action='store_true', help='Skip text extraction (faster)')
    args = parser.parse_args()

    # Download / load video
    video_path = f'{TMP_DIR}/video.mp4'
    video_path = download_video(args.input, video_path)

    # Get duration
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                             '-of', 'csv=p=0', video_path], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f'Video duration: {duration:.1f}s')

    # Extract features
    video_feats = extract_video_features(video_path)
    print(f'Video features: {video_feats.shape}')

    audio_feats = extract_audio_features(video_path)
    print(f'Audio features: {audio_feats.shape if audio_feats is not None else "none"}')

    text_feats = None
    if not args.no_text:
        wav_path = f'{TMP_DIR}/audio_for_asr.wav'
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', '16000', '-vn', wav_path],
                       capture_output=True)
        text_feats = extract_text_features_from_audio(wav_path, duration)
        print(f'Text features: {text_feats.shape}')
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # Build samples + run inference
    video_samples, audio_samples, text_samples, sample_times = build_samples(
        video_feats, audio_feats, text_feats, None
    )
    print(f'Samples: {video_samples.shape}')

    if video_samples.shape[0] == 0:
        print('ERROR: Video too short. Need at least ~12 seconds.')
        sys.exit(1)

    brain_preds = run_inference(video_samples, audio_samples, text_samples)
    print(f'Brain predictions: {brain_preds.shape}')

    # Interpret
    report = interpret(brain_preds, sample_times)

    # Save
    video_name = Path(args.input).stem if os.path.exists(args.input) else args.input.split('/')[-1].split('?')[0]
    out_json = f'{RESULTS_DIR}/{video_name}.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved: {out_json}')

    print(f'\n======================')
    print(f'NeuroScore: {report["neuroscore"]}/100')
    print(f'Duration: {report["duration_seconds"]:.1f}s')
    print(f'Peak moments:')
    for p in report['peaks']:
        print(f'  {p["time"]:.1f}s → {p["score"]:.1f}')
    print(f'Dead zones ({len(report["dead_zones"])}):')
    for dz in report['dead_zones']:
        print(f'  {dz["start"]:.1f}s → {dz["end"]:.1f}s ({dz["duration"]:.1f}s)')


if __name__ == '__main__':
    main()
