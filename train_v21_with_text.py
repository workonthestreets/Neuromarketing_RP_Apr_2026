"""
v21: v19/v20 architecture + text features.
- Video + audio: 25 time steps × (1024+1280) → 1D Conv
- Text: averaged over HRF-delayed window → dense branch
- Concatenate + subject head
"""

import os
import gc
import numpy as np
import h5py
import shutil
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.cluster import KMeans

BASE = '/workspace/neuroscore'
FMRI_BASE = f'{BASE}/algonauts_2025.competitors/fmri'
FEAT_VIDEO = f'{BASE}/features/video'
FEAT_AUDIO = f'{BASE}/features/audio'
FEAT_TEXT = f'{BASE}/features/text'
MODELS_DIR = f'{BASE}/models'
CACHE_DIR = f'{BASE}/cache'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TR = 1.49
HRF_DELAYS = [0, 1, 2, 3, 4]
WINDOW_SIZE = 5
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
N_SUBJECTS = len(SUBJECTS)
N_PARCELS = 1000
N_CLUSTERS = 3
HIDDEN_DIM = 256
TEXT_DIM_RAW = 6144
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 0.05
SWA_START = 50
SWA_LR = 1e-4
PEARSON_MSE_WEIGHT = 0.03
GRAD_CLIP = 5.0
TEXT_BIN_HZ = 2.0  # text is at 2Hz


def load_data_with_text(use_cache=True):
    cache_key = 'v21_text'
    cache_files = [f'{CACHE_DIR}/{cache_key}_{n}.npy' for n in ['video', 'audio', 'text', 'fmri', 'subject']]
    if use_cache and all(os.path.exists(f) for f in cache_files):
        print('Loading from cache (memmap)...')
        all_video = np.load(cache_files[0], mmap_mode='r')
        all_audio = np.load(cache_files[1], mmap_mode='r')
        all_text = np.load(cache_files[2], mmap_mode='r')
        all_fmri = np.load(cache_files[3], mmap_mode='r')
        all_subject = np.load(cache_files[4], mmap_mode='r')
        return all_video, all_audio, all_text, all_fmri, all_subject

    print('Discovering sessions + counting samples (exact)...')
    sessions = []
    total = 0
    max_delay = max(HRF_DELAYS)
    start_idx_window = WINDOW_SIZE - 1 + max_delay

    for sub_idx, sub in enumerate(SUBJECTS):
        sub_dir = f'{FMRI_BASE}/{sub}/func'
        if not os.path.exists(sub_dir):
            continue
        fmri_file = None
        for f in os.listdir(sub_dir):
            if f.endswith('.h5') and 'movie10' in f:
                fmri_file = f'{sub_dir}/{f}'
                break
        if not fmri_file:
            continue

        temp_file = f'/tmp/{os.path.basename(fmri_file)}'
        shutil.copyfile(fmri_file, temp_file)
        with h5py.File(temp_file, 'r') as hf:
            for key in sorted(hf.keys()):
                parts = key.split('_task-')
                if len(parts) < 2:
                    continue
                movie_name = parts[1].split('_run-')[0]
                video_f = f'{FEAT_VIDEO}/{movie_name}.npy'
                audio_f = f'{FEAT_AUDIO}/{movie_name}.npy'
                text_f = f'{FEAT_TEXT}/{movie_name}.npy'
                if not os.path.exists(video_f) or not os.path.exists(audio_f):
                    continue
                n_fmri = hf[key].shape[0]
                video_data = np.load(video_f, mmap_mode='r')
                audio_data = np.load(audio_f, mmap_mode='r')
                fmri_times = np.arange(n_fmri) * TR
                video_times = np.arange(video_data.shape[0]) * 1.0
                audio_times = np.linspace(0, video_times[-1], audio_data.shape[0])
                max_time = min(fmri_times[-1], video_times[-1], audio_times[-1])
                n_common = int((fmri_times <= max_time).sum())
                n_samples = max(0, n_common - start_idx_window)
                sessions.append({
                    'sub_idx': sub_idx, 'sub': sub, 'fmri_file': fmri_file,
                    'key': key, 'movie_name': movie_name,
                    'n_samples': n_samples, 'has_text': os.path.exists(text_f)
                })
                total += n_samples
        os.remove(temp_file)

    print(f'Found {len(sessions)} sessions, {total} total samples')
    has_text_count = sum(1 for s in sessions if s['has_text'])
    print(f'Sessions with text: {has_text_count}/{len(sessions)}')

    video_dim = 1024 * WINDOW_SIZE * len(HRF_DELAYS)
    audio_dim = 1280 * WINDOW_SIZE * len(HRF_DELAYS)

    all_video = np.empty((total, video_dim), dtype=np.float32)
    all_audio = np.empty((total, audio_dim), dtype=np.float32)
    all_text = np.zeros((total, TEXT_DIM_RAW), dtype=np.float32)  # zeros if no text
    all_fmri = np.empty((total, N_PARCELS), dtype=np.float32)
    all_subject = np.empty(total, dtype=np.int64)

    max_delay = max(HRF_DELAYS)
    start_idx_window = WINDOW_SIZE - 1 + max_delay
    write_idx = 0

    for sub_idx, sub in enumerate(SUBJECTS):
        sub_sessions = [s for s in sessions if s['sub_idx'] == sub_idx]
        if not sub_sessions:
            continue
        fmri_file = sub_sessions[0]['fmri_file']
        temp_file = f'/tmp/{os.path.basename(fmri_file)}'
        shutil.copyfile(fmri_file, temp_file)
        print(f'\n=== {sub} ({len(sub_sessions)} sessions) ===')

        with h5py.File(temp_file, 'r') as hf:
            for s in sub_sessions:
                fmri_data = hf[s['key']][:]
                video_data = np.load(f'{FEAT_VIDEO}/{s["movie_name"]}.npy')
                audio_data = np.load(f'{FEAT_AUDIO}/{s["movie_name"]}.npy')
                text_data = None
                if s['has_text']:
                    text_data = np.load(f'{FEAT_TEXT}/{s["movie_name"]}.npy')

                n_fmri = fmri_data.shape[0]
                fmri_times = np.arange(n_fmri) * TR
                video_times = np.arange(video_data.shape[0]) * 1.0
                audio_times = np.linspace(0, video_times[-1], audio_data.shape[0])

                max_time = min(fmri_times[-1], video_times[-1], audio_times[-1])
                fmri_mask = fmri_times <= max_time
                common_times = fmri_times[fmri_mask]

                video_resampled = interp1d(video_times, video_data, axis=0, fill_value='extrapolate')(common_times)
                audio_resampled = interp1d(audio_times, audio_data, axis=0, fill_value='extrapolate')(common_times)
                fmri_trimmed = fmri_data[fmri_mask]

                n = len(common_times)
                for i in range(start_idx_window, n):
                    v_parts, a_parts = [], []
                    for delay in HRF_DELAYS:
                        center = i - delay
                        v_parts.append(video_resampled[center - WINDOW_SIZE + 1: center + 1].flatten())
                        a_parts.append(audio_resampled[center - WINDOW_SIZE + 1: center + 1].flatten())
                    all_video[write_idx] = np.concatenate(v_parts)
                    all_audio[write_idx] = np.concatenate(a_parts)
                    all_fmri[write_idx] = fmri_trimmed[i]
                    all_subject[write_idx] = sub_idx

                    # Text: avg over HRF-delayed window (~4.5s back to now)
                    if text_data is not None:
                        t_target = common_times[i]  # TR time in seconds
                        bin_end = int(t_target * TEXT_BIN_HZ)
                        bin_start = max(0, bin_end - int(max_delay * TR * TEXT_BIN_HZ))
                        if bin_start < text_data.shape[0]:
                            bin_end_clipped = min(bin_end + 1, text_data.shape[0])
                            if bin_end_clipped > bin_start:
                                all_text[write_idx] = text_data[bin_start:bin_end_clipped].mean(axis=0)
                    write_idx += 1

                del fmri_data, video_data, audio_data, video_resampled, audio_resampled, text_data
                gc.collect()
        os.remove(temp_file)
        print(f'  Wrote {write_idx} / {total}')

    # Z-score normalize
    print('Z-score normalizing...')
    for name, arr in [('video', all_video), ('audio', all_audio), ('text', all_text), ('fmri', all_fmri)]:
        m = arr.mean(0, keepdims=True)
        sd = arr.std(0, keepdims=True) + 1e-6
        arr -= m
        arr /= sd
        np.save(f'{MODELS_DIR}/{name}_mean.npy', m)
        np.save(f'{MODELS_DIR}/{name}_std.npy', sd)

    print('Saving to cache...')
    np.save(cache_files[0], all_video)
    np.save(cache_files[1], all_audio)
    np.save(cache_files[2], all_text)
    np.save(cache_files[3], all_fmri)
    np.save(cache_files[4], all_subject)

    return all_video, all_audio, all_text, all_fmri, all_subject


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

        # Video + audio branch (same as v19)
        self.conv1 = nn.Conv1d(frame_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.gelu = nn.GELU()

        # Text branch
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Subject heads take concat of [conv_pool, text_proj] = 2*hidden_dim
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
        conv_pool = x.mean(dim=2)  # (B, hidden_dim)

        text_feat = self.text_proj(text)  # (B, hidden_dim)

        combined = torch.cat([conv_pool, text_feat], dim=-1)

        out = torch.zeros(B, self.subject_heads[0].out_features, device=video.device)
        for i in range(len(self.subject_heads)):
            mask = (subject_idx == i)
            if mask.any():
                out[mask] = self.subject_heads[i](combined[mask])
        return out


def pearson_loss(pred, target):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * target_c).sum(dim=0)
    den = torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0) + 1e-8)
    return 1 - (num / den).mean()


def train_cluster(cluster_id, parcel_mask, train_loader, val_loader,
                  video_dim, audio_dim, text_dim, device, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_parcels = int(parcel_mask.sum())
    print(f'\n--- Cluster {cluster_id}: {n_parcels} parcels ---')

    model = BrainEncoderWithText(video_dim, audio_dim, text_dim, n_parcels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    mse_fn = nn.MSELoss()
    parcel_mask_t = torch.from_numpy(parcel_mask).to(device)

    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for vb, ab, tb, fb, sb in train_loader:
            vb, ab, tb, fb, sb = vb.to(device), ab.to(device), tb.to(device), fb.to(device), sb.to(device)
            fb_cluster = fb[:, parcel_mask_t]
            pred = model(vb, ab, tb, sb)
            loss = pearson_loss(pred, fb_cluster) + PEARSON_MSE_WEIGHT * mse_fn(pred, fb_cluster)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if epoch >= SWA_START:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vb, ab, tb, fb, sb in val_loader:
                vb, ab, tb, fb, sb = vb.to(device), ab.to(device), tb.to(device), fb.to(device), sb.to(device)
                fb_cluster = fb[:, parcel_mask_t]
                val_loss += mse_fn(model(vb, ab, tb, sb), fb_cluster).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    return swa_model, best_val


def main():
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    all_video, all_audio, all_text, all_fmri, all_subject = load_data_with_text()
    video_dim = all_video.shape[1]
    audio_dim = all_audio.shape[1]
    text_dim = all_text.shape[1]
    n_total = len(all_video)

    print(f'Video: {all_video.shape} | Audio: {all_audio.shape} | Text: {all_text.shape}')
    print(f'fMRI: {all_fmri.shape}')

    np.random.seed(42)
    idx = np.random.permutation(n_total)
    split = int(0.9 * n_total)
    train_idx = idx[:split]
    val_idx = idx[split:]

    print('\nClustering parcels...')
    fmri_train = np.array(all_fmri[train_idx])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    parcel_clusters = kmeans.fit_predict(fmri_train.T)
    print(f'Cluster sizes: {np.bincount(parcel_clusters)}')
    np.save(f'{MODELS_DIR}/parcel_clusters.npy', parcel_clusters)
    del fmri_train
    gc.collect()

    video_t = torch.from_numpy(np.asarray(all_video))
    audio_t = torch.from_numpy(np.asarray(all_audio))
    text_t = torch.from_numpy(np.asarray(all_text))
    fmri_t = torch.from_numpy(np.asarray(all_fmri))
    subject_t = torch.from_numpy(np.asarray(all_subject))

    full_ds = TensorDataset(video_t, audio_t, text_t, fmri_t, subject_t)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    n_val = len(val_ds)
    final_preds_val = np.zeros((n_val, N_PARCELS), dtype=np.float32)
    final_targets_val = np.zeros((n_val, N_PARCELS), dtype=np.float32)
    col = 0
    for vb, ab, tb, fb, sb in val_loader:
        bs = fb.shape[0]
        final_targets_val[col:col + bs] = fb.numpy()
        col += bs

    print(f'\n=== Training 3 specialist models (video + audio + text) ===')
    for cluster_id in range(N_CLUSTERS):
        parcel_mask = parcel_clusters == cluster_id
        swa_model, best_val = train_cluster(cluster_id, parcel_mask, train_loader, val_loader,
                                            video_dim, audio_dim, text_dim, device)
        torch.save(swa_model.module.state_dict(), f'{MODELS_DIR}/v21_cluster{cluster_id}_swa.pt')

        swa_model.eval()
        col = 0
        parcel_indices = np.where(parcel_mask)[0]
        with torch.no_grad():
            for vb, ab, tb, fb, sb in val_loader:
                vb, ab, tb, sb = vb.to(device), ab.to(device), tb.to(device), sb.to(device)
                pred = swa_model(vb, ab, tb, sb).cpu().numpy()
                bs = pred.shape[0]
                final_preds_val[col:col + bs, parcel_indices] = pred
                col += bs
        print(f'  Cluster {cluster_id} best val: {best_val:.6f}')
        del swa_model
        torch.cuda.empty_cache()
        gc.collect()

    total_mse = ((final_preds_val - final_targets_val) ** 2).mean()
    r2 = (1 - total_mse) * 100
    print(f'\n=== Combined results ===')
    print(f'  v19 (no text):                    0.766 (R2=23.4%)')
    print(f'  v20 (ensemble of 5 v19):          0.760 (R2=24.0%)')
    print(f'  v21 (video+audio+text, 1 model): {total_mse:.6f} (R2={r2:.1f}%)')


if __name__ == '__main__':
    main()
