# RunPod + Cursor Setup Guide

## Pod Configuration

| Setting | Value |
|---------|-------|
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Template search name | "RunPod Pytorch 2.4" |
| **GPU** | RTX 4090 (24GB) |
| Cloud type | Community Cloud |
| Disk size | 50GB |
| **Network Volume** | CREATE ONE — mount to `/workspace` |
| Exposed ports | 8888 (Jupyter), 22 (SSH) |

## Why these choices

- **PyTorch 2.4** — battle-tested with HuggingFace transformers, works with Whisper-large-v3 + DINOv2
- **CUDA 12.4** — required for RTX 4090 (Ada architecture)
- **Python 3.11** — stable, compatible with all our libraries
- **Network Volume** — persistent storage, data survives pod stop/start
- **Community Cloud** — ~30% cheaper than Secure, fine for single-GPU training

## Setup Steps

### 1. Sign up at runpod.io, add $20 credit

### 2. Create Network Volume FIRST (before the pod)
- Go to "Storage" → "Create Network Volume"
- Size: 50GB, same region as pod
- This persists data across pod restarts

### 3. Deploy Pod
- Template: `RunPod Pytorch 2.4`
- GPU: RTX 4090
- Attach the Network Volume, mount point `/workspace`
- Expose port 8888 and 22

### 4. Add SSH key in RunPod account settings (one-time)
Generate on Mac if you don't have one:
```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```
Copy the output, paste in RunPod "Settings" → "SSH Public Keys"

### 5. First-time install (one SSH command)
After connecting via SSH:
```bash
pip install transformers scipy scikit-learn h5py matplotlib datalad
apt-get update && apt-get install -y ffmpeg git-annex
```

### 6. Connect Cursor via SSH
- Install "Remote - SSH" extension in Cursor (usually preinstalled)
- RunPod → your pod → "Connect" → copy SSH command
- In Cursor: Cmd+Shift+P → "Remote-SSH: Connect to Host" → paste

### 7. Transfer your data from Google Drive
Option A: Use `rclone` to sync directly from Drive to pod
Option B: Download locally, then `scp` to pod
Option C: Re-download Algonauts dataset via datalad directly on pod (recommended — faster than Drive)

## Known Issues to Avoid

- **DON'T use PyTorch 2.8** — has regressions with Whisper-large-v3
- **DON'T use CUDA 11.x** — RTX 4090 won't run at full speed
- **DON'T install in `/root` or `/`** — won't persist across pod restarts
- **DO install everything in `/workspace`** (the Network Volume)
