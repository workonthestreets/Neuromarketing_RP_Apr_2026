# v19 Best Model — Working Cells

Current best: **R²=23.3%**, val loss 0.767

Run in order: Cell 1 → 3 → 4 → 5 → 6

---

## Cell 1 — Setup

```python
from google.colab import drive
drive.mount('/content/drive')
import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
print(f'GPU: {torch.cuda.get_device_name(0)}')
```

---

## Cell 3 — Data Loading (FIR multi-delay + past-only temporal window)

```python
import numpy as np, h5py, os, shutil
from scipy.interpolate import interp1d

fmri_base = '/content/drive/MyDrive/neuroscore/data/algonauts_2025.competitors/fmri'
feat_video = '/content/drive/MyDrive/neuroscore/features/video'
feat_audio = '/content/drive/MyDrive/neuroscore/features/audio'

if not os.path.exists(fmri_base):
    fmri_base = '/content/drive/MyDrive/neuroscore/data/algonauts_2025.competitors'

TR = 1.49
HRF_DELAYS = [0, 1, 2, 3, 4]
WINDOW_SIZE = 5
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-05']

paired_video, paired_audio, paired_fmri, paired_subject = [], [], [], []
matched = 0
skipped = 0

for sub_idx, sub in enumerate(subjects):
    sub_dir = f'{fmri_base}/{sub}/func'
    fmri_file = None
    if not os.path.exists(sub_dir):
        continue
    for f in os.listdir(sub_dir):
        if f.endswith('.h5') and 'movie10' in f:
            fmri_file = f'{sub_dir}/{f}'
            break
    if not fmri_file:
        continue

    print(f'\n=== {sub} (idx={sub_idx}) ===')
    temp_fmri_file = f'/tmp/{os.path.basename(fmri_file)}'
    shutil.copyfile(fmri_file, temp_fmri_file)

    with h5py.File(temp_fmri_file, 'r') as hf:
        for key in sorted(hf.keys()):
            parts = key.split('_task-')
            if len(parts) < 2:
                continue
            movie_name = parts[1].split('_run-')[0]

            video_file = f'{feat_video}/{movie_name}.npy'
            audio_file = f'{feat_audio}/{movie_name}.npy'
            if not os.path.exists(video_file) or not os.path.exists(audio_file):
                skipped += 1
                continue

            fmri_data = hf[key][:]
            video_data = np.load(video_file)
            audio_data = np.load(audio_file)

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

            max_delay = max(HRF_DELAYS)
            n = len(common_times)
            start_idx = WINDOW_SIZE - 1 + max_delay

            for i in range(start_idx, n):
                v_parts, a_parts = [], []
                for delay in HRF_DELAYS:
                    center = i - delay
                    v_window = video_resampled[center - WINDOW_SIZE + 1 : center + 1].flatten()
                    a_window = audio_resampled[center - WINDOW_SIZE + 1 : center + 1].flatten()
                    v_parts.append(v_window)
                    a_parts.append(a_window)

                paired_video.append(np.concatenate(v_parts))
                paired_audio.append(np.concatenate(a_parts))
                paired_fmri.append(fmri_trimmed[i])
                paired_subject.append(sub_idx)

            matched += 1
        os.remove(temp_fmri_file)
    print(f'  matched so far = {matched}')

print(f'\nTotal matched: {matched} | Skipped: {skipped}')

all_video = np.array(paired_video, dtype=np.float32)
all_audio = np.array(paired_audio, dtype=np.float32)
all_fmri = np.array(paired_fmri, dtype=np.float32)
all_subject = np.array(paired_subject, dtype=np.int64)

print(f'Video: {all_video.shape}')
print(f'Audio: {all_audio.shape}')
print(f'fMRI:  {all_fmri.shape}')

for name, arr in [('video', all_video), ('audio', all_audio), ('fmri', all_fmri)]:
    m, s = arr.mean(0, keepdims=True), arr.std(0, keepdims=True) + 1e-6
    arr -= m
    arr /= s
    save = '/content/drive/MyDrive/neuroscore/models'
    np.save(f'{save}/{name}_mean.npy', m)
    np.save(f'{save}/{name}_std.npy', s)

VIDEO_DIM, AUDIO_DIM, FMRI_DIM = all_video.shape[1], all_audio.shape[1], all_fmri.shape[1]
print(f'\nVideo dim: {VIDEO_DIM} | Audio dim: {AUDIO_DIM} | fMRI dim: {FMRI_DIM}')
print('Data ready (FIR multi-delay, past-only window)')
```

---

## Cell 4 — 1D-Conv Model with Subject Heads

```python
import torch
import torch.nn as nn

class BrainEncoder(nn.Module):
    def __init__(self, video_dim, audio_dim, fmri_dim, n_subjects=4, hidden_dim=256, window_size=5, n_delays=5):
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

        self.subject_heads = nn.ModuleList([
            nn.Linear(hidden_dim, fmri_dim) for _ in range(n_subjects)
        ])

    def forward(self, video, audio, subject_idx):
        B = video.shape[0]
        T = self.window_size * self.n_delays

        v = video.view(B, T, self.video_per_frame)
        a = audio.view(B, T, self.audio_per_frame)
        x = torch.cat([v, a], dim=-1)

        x = x.transpose(1, 2)
        x = self.gelu(self.conv1(x))
        x = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = self.gelu(self.conv2(x))
        x = self.ln2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        x = self.gelu(self.conv3(x))
        x = self.ln3(x.transpose(1, 2)).transpose(1, 2)

        x = x.mean(dim=2)

        out = torch.zeros(B, self.subject_heads[0].out_features, device=video.device)
        for i in range(len(self.subject_heads)):
            mask = (subject_idx == i)
            if mask.any():
                out[mask] = self.subject_heads[i](x[mask])
        return out
```

---

## Cell 5 — Dataset + Parcel Clustering

```python
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

device = torch.device('cuda')

np.random.seed(42)
idx = np.random.permutation(len(all_video))
all_video, all_audio, all_fmri, all_subject = all_video[idx], all_audio[idx], all_fmri[idx], all_subject[idx]

split = int(0.9 * len(all_video))

print('Clustering brain parcels into 3 networks...')
parcel_responses = all_fmri[:split].T
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
parcel_clusters = kmeans.fit_predict(parcel_responses)

cluster_sizes = np.bincount(parcel_clusters)
print(f'Cluster sizes: {cluster_sizes}')

np.save('/content/drive/MyDrive/neuroscore/models/parcel_clusters.npy', parcel_clusters)

train_ds = TensorDataset(torch.tensor(all_video[:split]), torch.tensor(all_audio[:split]), torch.tensor(all_fmri[:split]), torch.tensor(all_subject[:split]))
val_ds = TensorDataset(torch.tensor(all_video[split:]), torch.tensor(all_audio[split:]), torch.tensor(all_fmri[split:]), torch.tensor(all_subject[split:]))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')
```

---

## Cell 6 — Train 3 Specialist Models

```python
from torch.optim.swa_utils import AveragedModel, SWALR

def pearson_loss(pred, target):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * target_c).sum(dim=0)
    den = torch.sqrt((pred_c**2).sum(dim=0) * (target_c**2).sum(dim=0) + 1e-8)
    return 1 - (num / den).mean()

mse_fn = nn.MSELoss()
parcel_clusters = np.load('/content/drive/MyDrive/neuroscore/models/parcel_clusters.npy')

final_preds_val = np.zeros((len(val_ds), 1000), dtype=np.float32)
final_targets_val = np.zeros((len(val_ds), 1000), dtype=np.float32)

collected = 0
for vb, ab, fb, sb in val_loader:
    bs = fb.shape[0]
    final_targets_val[collected:collected+bs] = fb.numpy()
    collected += bs

print(f'\n=== Training 3 specialist 1D-Conv models ===\n')

for cluster_id in range(3):
    parcel_mask = parcel_clusters == cluster_id
    n_parcels = parcel_mask.sum()
    print(f'\n--- Cluster {cluster_id}: {n_parcels} parcels ---')

    cluster_model = BrainEncoder(VIDEO_DIM, AUDIO_DIM, n_parcels, n_subjects=4, hidden_dim=256).to(device)

    optimizer = torch.optim.AdamW(cluster_model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    swa_model = AveragedModel(cluster_model)
    swa_start = 50
    swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

    best_val = float('inf')

    for epoch in range(1, 101):
        cluster_model.train()
        train_loss = 0
        for vb, ab, fb, sb in train_loader:
            vb, ab, fb, sb = vb.to(device), ab.to(device), fb.to(device), sb.to(device)
            fb_cluster = fb[:, parcel_mask]
            pred = cluster_model(vb, ab, sb)
            loss = pearson_loss(pred, fb_cluster) + 0.03 * mse_fn(pred, fb_cluster)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cluster_model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if epoch >= swa_start:
            swa_model.update_parameters(cluster_model)
            swa_scheduler.step()
        else:
            scheduler.step()

        cluster_model.eval()
        val_loss = 0
        with torch.no_grad():
            for vb, ab, fb, sb in val_loader:
                vb, ab, fb, sb = vb.to(device), ab.to(device), fb.to(device), sb.to(device)
                fb_cluster = fb[:, parcel_mask]
                val_loss += mse_fn(cluster_model(vb, ab, sb), fb_cluster).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/100 | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

    print(f'  Updating SWA BN stats...')
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    swa_model.eval()
    collected = 0
    with torch.no_grad():
        for vb, ab, fb, sb in val_loader:
            vb, ab, sb = vb.to(device), ab.to(device), sb.to(device)
            pred = swa_model(vb, ab, sb).cpu().numpy()
            bs = pred.shape[0]
            idx_parcels = np.where(parcel_mask)[0]
            final_preds_val[collected:collected+bs, idx_parcels] = pred
            collected += bs

    print(f'  Cluster {cluster_id} best val: {best_val:.6f}')

total_mse = ((final_preds_val - final_targets_val) ** 2).mean()
print(f'\n=== Combined specialist results ===')
print(f'  v19 (1D-Conv specialists): {total_mse:.6f} (R2={(1-total_mse)*100:.1f}%)')
```
