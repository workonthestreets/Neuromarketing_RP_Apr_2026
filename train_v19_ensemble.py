"""
v20: Ensemble of 5 v19 models with different random seeds.
Expected: R² 23.4% → ~24-25%
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

# ============================================================
# Paths
# ============================================================
BASE = '/workspace/neuroscore'
FMRI_BASE = f'{BASE}/algonauts_2025.competitors/fmri'
FEAT_VIDEO = f'{BASE}/features/video'
FEAT_AUDIO = f'{BASE}/features/audio'
MODELS_DIR = f'{BASE}/models'
CACHE_DIR = f'{BASE}/cache'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# Config
# ============================================================
TR = 1.49
HRF_DELAYS = [0, 1, 2, 3, 4]
WINDOW_SIZE = 5
SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-05']
N_SUBJECTS = len(SUBJECTS)
N_PARCELS = 1000
N_CLUSTERS = 3
HIDDEN_DIM = 256
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 0.05
SWA_START = 50
SWA_LR = 1e-4
PEARSON_MSE_WEIGHT = 0.03
GRAD_CLIP = 5.0

# Ensemble config
N_ENSEMBLE = 5
SEEDS = [42, 123, 777, 2026, 31415]


def load_data_cached():
    cache_files = [f'{CACHE_DIR}/{n}.npy' for n in ['video', 'audio', 'fmri', 'subject']]
    if all(os.path.exists(f) for f in cache_files):
        print('Loading from cache (memmap)...')
        all_video = np.load(f'{CACHE_DIR}/video.npy', mmap_mode='r')
        all_audio = np.load(f'{CACHE_DIR}/audio.npy', mmap_mode='r')
        all_fmri = np.load(f'{CACHE_DIR}/fmri.npy', mmap_mode='r')
        all_subject = np.load(f'{CACHE_DIR}/subject.npy', mmap_mode='r')
        return all_video, all_audio, all_fmri, all_subject
    raise RuntimeError('No cache found. Run train_v19.py first to build the cache.')


class BrainEncoder(nn.Module):
    def __init__(self, video_dim, audio_dim, fmri_dim,
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


def pearson_loss(pred, target):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    num = (pred_c * target_c).sum(dim=0)
    den = torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0) + 1e-8)
    return 1 - (num / den).mean()


def train_cluster(cluster_id, parcel_mask, train_loader, val_loader,
                  video_dim, audio_dim, device, seed, verbose=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    n_parcels = int(parcel_mask.sum())

    model = BrainEncoder(video_dim, audio_dim, n_parcels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    mse_fn = nn.MSELoss()

    best_val = float('inf')
    parcel_mask_t = torch.from_numpy(parcel_mask).to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for vb, ab, fb, sb in train_loader:
            vb, ab, fb, sb = vb.to(device), ab.to(device), fb.to(device), sb.to(device)
            fb_cluster = fb[:, parcel_mask_t]
            pred = model(vb, ab, sb)
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
            for vb, ab, fb, sb in val_loader:
                vb, ab, fb, sb = vb.to(device), ab.to(device), fb.to(device), sb.to(device)
                fb_cluster = fb[:, parcel_mask_t]
                val_loss += mse_fn(model(vb, ab, sb), fb_cluster).item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss

        if verbose and epoch % 20 == 0:
            print(f'    Epoch {epoch}/{EPOCHS} | Val: {val_loss:.6f}')

    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    return swa_model, best_val


def main():
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Ensemble size: {N_ENSEMBLE} models')

    all_video, all_audio, all_fmri, all_subject = load_data_cached()
    video_dim = all_video.shape[1]
    audio_dim = all_audio.shape[1]
    n_total = len(all_video)

    # Fixed train/val split (same across all ensemble members for fair averaging)
    np.random.seed(42)
    idx = np.random.permutation(n_total)
    split = int(0.9 * n_total)
    train_idx = idx[:split]
    val_idx = idx[split:]

    # Cluster parcels (same clusters across ensemble)
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
    fmri_t = torch.from_numpy(np.asarray(all_fmri))
    subject_t = torch.from_numpy(np.asarray(all_subject))

    full_ds = TensorDataset(video_t, audio_t, fmri_t, subject_t)
    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds = Subset(full_ds, val_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    n_val = len(val_ds)
    final_targets_val = np.zeros((n_val, N_PARCELS), dtype=np.float32)
    col = 0
    for vb, ab, fb, sb in val_loader:
        bs = fb.shape[0]
        final_targets_val[col:col + bs] = fb.numpy()
        col += bs

    # Accumulate predictions across ensemble + clusters
    ensemble_preds = np.zeros((n_val, N_PARCELS), dtype=np.float32)
    single_model_mse_history = []

    print(f'\n=== Training ensemble of {N_ENSEMBLE} models ===')

    for ens_i, seed in enumerate(SEEDS[:N_ENSEMBLE]):
        print(f'\n\n###### Ensemble member {ens_i+1}/{N_ENSEMBLE} (seed={seed}) ######')
        member_preds = np.zeros((n_val, N_PARCELS), dtype=np.float32)

        for cluster_id in range(N_CLUSTERS):
            parcel_mask = parcel_clusters == cluster_id
            n_parcels = int(parcel_mask.sum())
            print(f'  Cluster {cluster_id} ({n_parcels} parcels)...')

            swa_model, best_val = train_cluster(cluster_id, parcel_mask, train_loader, val_loader,
                                                video_dim, audio_dim, device, seed=seed + cluster_id)

            torch.save(swa_model.module.state_dict(),
                       f'{MODELS_DIR}/v20_ens{ens_i}_cluster{cluster_id}_swa.pt')

            swa_model.eval()
            col = 0
            parcel_indices = np.where(parcel_mask)[0]
            with torch.no_grad():
                for vb, ab, fb, sb in val_loader:
                    vb, ab, sb = vb.to(device), ab.to(device), sb.to(device)
                    pred = swa_model(vb, ab, sb).cpu().numpy()
                    bs = pred.shape[0]
                    member_preds[col:col + bs, parcel_indices] = pred
                    col += bs
            print(f'    Cluster {cluster_id} best val: {best_val:.6f}')

            del swa_model
            torch.cuda.empty_cache()
            gc.collect()

        # Report this member's single-model MSE
        member_mse = ((member_preds - final_targets_val) ** 2).mean()
        single_model_mse_history.append(member_mse)
        print(f'  Member {ens_i+1} single-model MSE: {member_mse:.6f} (R2={(1-member_mse)*100:.1f}%)')

        # Running ensemble average
        ensemble_preds = (ensemble_preds * ens_i + member_preds) / (ens_i + 1)
        ensemble_mse = ((ensemble_preds - final_targets_val) ** 2).mean()
        print(f'  Ensemble-so-far ({ens_i+1} models) MSE: {ensemble_mse:.6f} (R2={(1-ensemble_mse)*100:.1f}%)')

    # Final results
    final_mse = ((ensemble_preds - final_targets_val) ** 2).mean()
    final_r2 = (1 - final_mse) * 100

    print(f'\n\n=== FINAL ENSEMBLE RESULTS ===')
    print(f'Individual member MSEs: {[f"{m:.4f}" for m in single_model_mse_history]}')
    print(f'  Avg single: {np.mean(single_model_mse_history):.6f}')
    print(f'  Ensemble:   {final_mse:.6f} (R2={final_r2:.1f}%)')
    print(f'  v19 (single): 0.766 (R2=23.4%)')

    # Save final ensemble predictions
    np.save(f'{MODELS_DIR}/v20_ensemble_preds_val.npy', ensemble_preds)
    np.save(f'{MODELS_DIR}/v20_ensemble_targets_val.npy', final_targets_val)


if __name__ == '__main__':
    main()
