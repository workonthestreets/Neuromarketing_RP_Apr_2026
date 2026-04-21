# Redeploy Pod Instructions

After terminating the pod, to come back:

## 1. Deploy new pod with existing volume
- RunPod → Storage → your `neuroscore-data` volume → "Create pod with volume"
- GPU: RTX 4090
- Template: **RunPod Pytorch 2.4.0**
- Container disk: 20 GB
- Volume auto-mounts to `/workspace`
- Enable SSH + Jupyter

## 2. SSH in
The IP/port will change. Copy the new SSH command from RunPod "Connect" button.

```bash
ssh root@NEW_IP -p NEW_PORT -i ~/.ssh/id_ed25519
```

## 3. Reinstall dependencies (copy-paste all at once)

```bash
pip install transformers scipy scikit-learn h5py matplotlib pandas tqdm
apt-get update && apt-get install -y ffmpeg unzip
wget https://downloads.kitenet.net/git-annex/linux/current/git-annex-standalone-amd64.tar.gz
tar -xzf git-annex-standalone-amd64.tar.gz -C /usr/local/lib/
ln -sf /usr/local/lib/git-annex.linux/git-annex /usr/local/bin/git-annex
ln -sf /usr/local/lib/git-annex.linux/git-annex-shell /usr/local/bin/git-annex-shell
rm -f /workspace/neuroscore/git-annex-standalone-amd64.tar.gz
git config --global user.name "neuroscore"
git config --global user.email "neuroscore@example.com"
```

## 4. Continue where we left off

The current task: train v21 (video + audio + text) with 3 specialist models.

```bash
cd /workspace/neuroscore
python train_v21_with_text.py
```

## 5. If Cursor SSH stops working
Update `~/.ssh/config` on your Mac with the new IP and port:
```
Host runpod-neuroscore
    HostName NEW_IP
    Port NEW_PORT
    User root
    IdentityFile ~/.ssh/id_ed25519
```

## Status when you left off

- Best model: v20 ensemble, R²=24.0%
- Next: v21 with text features (just uploaded, not yet trained)
- All data + features + context in `/workspace/neuroscore/`
- Cost while pod running: $0.69/hr + $0.005/hr for volume
- Cost while terminated: $0.12/day for volume only
