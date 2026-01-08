#!/usr/bin/env python3
"""
Modified policy distillation script supporting multi-file loading AND progress bars.
"""

import argparse
import os
import pickle
import glob
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv

# === 新增：导入 tqdm ===
try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' not installed. Please install it via: pip install tqdm")
    exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


class RolloutDataset(Dataset):
    def __init__(self, path: str, fmt: str = "npz", obs_key: str = "obs", act_key: str = "acts"):
        self.path = path
        self.fmt = fmt
        self.obs = None
        self.acts = None
        self._load(path, fmt, obs_key, act_key)

    def _load(self, path, fmt, obs_key, act_key):
        file_list = []
        if os.path.isdir(path):
            file_list = sorted(glob.glob(os.path.join(path, "*.npz")))
            print(f"Detected directory. Found {len(file_list)} files.")
        else:
            file_list = [path]

        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found at {path}")

        all_obs = []
        all_acts = []

        print("Loading data...")
        for f_path in tqdm(file_list, desc="Loading Files"):
            if fmt == "npz":
                data = np.load(f_path)
                if obs_key not in data or act_key not in data:
                    continue
                obs = data[obs_key].astype(np.float32)
                acts = data[act_key].astype(np.float32)
            elif fmt == "pickle":
                with open(f_path, "rb") as f:
                    data = pickle.load(f)
                obs = np.asarray(data[obs_key], dtype=np.float32)
                acts = np.asarray(data[act_key], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported format: {fmt}")

            if obs.ndim == 1: obs = obs[None, :]
            if acts.ndim == 1: acts = acts[None, :]

            all_obs.append(obs)
            all_acts.append(acts)

        self.obs = np.concatenate(all_obs, axis=0)
        self.acts = np.concatenate(all_acts, axis=0)

        if len(self.obs) != len(self.acts):
            raise RuntimeError(f"Total obs length {len(self.obs)} != acts length {len(self.acts)}")
        
        print(f"Total samples loaded: {len(self.obs)}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.obs[idx], self.acts[idx]


class SequenceRolloutDataset(Dataset):
    def __init__(self, path: str, seq_len: int = 30, fmt: str = "npz", obs_key: str = "obs",
                 act_key: str = "acts", qpos_dim: int = 22):
        self.path = path
        self.seq_len = int(seq_len)
        self.fmt = fmt
        self.obs_key = obs_key
        self.act_key = act_key
        self.qpos_dim = int(qpos_dim)

        self._load(path, fmt, obs_key, act_key)

    def _load(self, path, fmt, obs_key, act_key):
        if fmt != 'npz':
            raise ValueError('Sequence dataset currently supports only npz format')

        file_list = []
        if os.path.isdir(path):
            file_list = sorted(glob.glob(os.path.join(path, "*.npz")))
            print(f"Detected directory. Found {len(file_list)} npz files.")
        else:
            file_list = [path]

        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found at {path}")

        all_obs = []
        all_acts = []
        all_dones = []

        print(f"Loading {len(file_list)} files...")
        
        # 使用 tqdm 显示加载进度
        for f_path in tqdm(file_list, desc="Loading Files"):
            try:
                data = np.load(f_path)
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                continue

            if obs_key not in data or act_key not in data:
                continue

            obs = data[obs_key].astype(np.float32)
            acts = data[act_key].astype(np.float32)

            if 'dones' in data:
                dones = data['dones'].astype(np.bool_)
            elif 'episode_starts' in data:
                ep_starts = data['episode_starts'].astype(np.bool_)
                dones = np.concatenate([np.zeros(1, dtype=bool), ep_starts[:-1]], axis=0)
            else:
                dones = np.zeros(len(obs), dtype=bool)

            if len(dones) > 0:
                dones[-1] = True

            all_obs.append(obs)
            all_acts.append(acts)
            all_dones.append(dones)

        self.obs = np.concatenate(all_obs, axis=0)
        self.acts = np.concatenate(all_acts, axis=0)
        self.dones = np.concatenate(all_dones, axis=0)

        if self.obs.ndim == 1: self.obs = self.obs[None, :]
        if self.acts.ndim == 1: self.acts = self.acts[None, :]

        print(f"Total Combined Data: {len(self.obs)} steps.")

        print("Computing valid indices (Vectorized)...")
        self.valid_indices = []
        T = len(self.obs)
        s = self.seq_len
        
        indices = np.arange(T)
        valid_mask = (indices >= s) 
        
        if self.dones is not None:
            dones_float = self.dones.astype(np.float32)
            dones_cumsum = np.cumsum(dones_float)
            dones_cumsum = np.insert(dones_cumsum, 0, 0)
            range_sums = dones_cumsum[indices + 1] - dones_cumsum[indices - s]
            no_done_mask = (range_sums == 0)
            
            final_mask = valid_mask.copy()
            final_mask[valid_mask] = no_done_mask[valid_mask]
            
            self.valid_indices = indices[final_mask]
        else:
            self.valid_indices = indices[valid_mask]

        print(f"Valid sequences available: {len(self.valid_indices)}")


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        s = self.seq_len
        q_start = t - s + 1
        q_end = t + 1
        qseq = self.obs[q_start:q_end, :self.qpos_dim].astype(np.float32)
        a_start = t - s
        a_end = t
        aseq = self.acts[a_start:a_end, :].astype(np.float32)
        seq = np.concatenate([qseq, aseq], axis=1)
        target = self.acts[t].astype(np.float32)
        return seq, target


class StudentMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class StudentTransformer(nn.Module):
    def __init__(self, seq_len: int, feat_dim: int, out_dim: int, trans_dim: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(feat_dim, trans_dim)
        self.pos_enc = PositionalEncoding(trans_dim, max_len=seq_len + 5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=trans_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(trans_dim, trans_dim),
            nn.ReLU(),
            nn.Linear(trans_dim, out_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        last = x[:, -1, :]
        return self.head(last)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.use_transformer:
        ds = SequenceRolloutDataset(args.rollout_file, seq_len=args.seq_len, fmt=args.format,
                                   obs_key=args.obs_key, act_key=args.act_key, qpos_dim=args.qpos_dim)
        act_dim = int(np.prod(ds.acts.shape[1:]))
        feat_dim = args.qpos_dim + act_dim
        obs_dim = feat_dim * args.seq_len
    else:
        ds = RolloutDataset(args.rollout_file, fmt=args.format, obs_key=args.obs_key, act_key=args.act_key)
        obs_dim = int(np.prod(ds.obs.shape[1:]))
        act_dim = int(np.prod(ds.acts.shape[1:]))

    if args.normalize:
        print("Normalizing data statistics...")
        obs_mean = ds.obs.mean(axis=0)
        obs_std = ds.obs.std(axis=0) + 1e-8
        ds.obs = (ds.obs - obs_mean) / obs_std

        act_mean = ds.acts.mean(axis=0)
        act_std = ds.acts.std(axis=0) + 1e-8
        ds.acts = (ds.acts - act_mean) / act_std
        print(f"Data normalized.")
    else:
        obs_mean = None
        obs_std = None

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # ... [Model Definition Code Skipped for Brevity - Same as before] ...
    # 为了完整性，这里放回模型定义代码
    teacher_model = None
    if args.use_a2c and not args.use_transformer:
         # build simple A2C-style network aligned with teacher (same as in collect_rollouts)
        class A2CNetwork(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.actor_mlp = nn.Sequential(
                    nn.Linear(obs_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                )
                self.mu = nn.Linear(256, act_dim)
                self.value = nn.Linear(256, 1)
            def forward(self, x):
                feats = self.actor_mlp(x)
                mu = self.mu(feats)
                return mu
            def latent(self, x):
                return self.actor_mlp(x)
        class StudentA2C(nn.Module):
            def __init__(self, obs_dim, act_dim):
                super().__init__()
                self.a2c_network = A2CNetwork(obs_dim, act_dim)
            def forward(self, x):
                return self.a2c_network(x)
            def latent(self, x):
                return self.a2c_network.latent(x)
        model = StudentA2C(obs_dim, act_dim).to(device)
        if args.teacher_ckpt:
            try:
                ck = torch.load(args.teacher_ckpt, map_location=device)
                sd = ck.get('model', ck.get('model_state', ck))
                teacher_model = StudentA2C(obs_dim, act_dim).to(device)
                try:
                    teacher_model.load_state_dict(sd, strict=False)
                except Exception:
                    try:
                        sd2 = {k.replace('a2c_network.', 'a2c_network.'): v for k, v in sd.items()}
                        teacher_model.load_state_dict(sd2, strict=False)
                    except Exception:
                        teacher_model = None
            except Exception:
                teacher_model = None
    else:
        if args.use_transformer:
            model = StudentTransformer(seq_len=args.seq_len, feat_dim=feat_dim, out_dim=act_dim,
                                       trans_dim=args.trans_dim, nhead=args.trans_heads, num_layers=args.trans_layers).to(device)
        else:
            model = StudentMLP(obs_dim, act_dim, hidden=tuple(args.hidden)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck.get("model_state", ck))
        opt.load_state_dict(ck.get("opt_state", opt.state_dict()))
        start_epoch = ck.get("epoch", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    log_dir = os.path.join(os.path.dirname(args.save_path) or '.', 'distill_logs')
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'loss.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as cf:
            w = csv.writer(cf)
            w.writerow(['epoch', 'loss'])

    writer = None
    if _HAS_TB:
        try:
            writer = SummaryWriter(log_dir=log_dir)
        except Exception:
            writer = None

    print(f"Starting training on device: {device}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        
        # === 修改点：使用 tqdm 包裹 dataloader ===
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for obs_batch, act_batch in pbar:
            if not torch.is_tensor(obs_batch):
                obs_batch = torch.from_numpy(np.stack(obs_batch) if isinstance(obs_batch, list) else obs_batch)
            if not torch.is_tensor(act_batch):
                act_batch = torch.from_numpy(np.stack(act_batch) if isinstance(act_batch, list) else act_batch)

            obs_batch = obs_batch.to(device).float()
            act_batch = act_batch.to(device).float()

            preds = model(obs_batch)
            loss = criterion(preds, act_batch)
            
            if args.latent_weight > 0.0 and teacher_model is not None and hasattr(model, 'latent'):
                with torch.no_grad():
                    teacher_lat = teacher_model.latent(obs_batch)
                student_lat = model.latent(obs_batch)
                latent_loss = nn.MSELoss()(student_lat, teacher_lat)
                loss = loss + args.latent_weight * latent_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            # === 修改点：实时更新进度条后缀显示当前 loss ===
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

            batch_size = obs_batch.shape[0]
            total_loss += loss.item() * batch_size
            n += batch_size

        avg_loss = total_loss / max(1, n)
        # 进度条结束后，打印最终平均 Loss
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        try:
            with open(csv_path, 'a', newline='') as cf:
                w = csv.writer(cf)
                w.writerow([epoch + 1, float(avg_loss)])
        except Exception:
            pass

        if writer is not None:
            try:
                writer.add_scalar('distill/loss', float(avg_loss), epoch + 1)
            except Exception:
                pass

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            ck = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "act_mean": ds.acts.mean(axis=0) if args.normalize else None,
                "act_std": ds.acts.std(axis=0) + 1e-8 if args.normalize else None
            }
            torch.save(ck, args.save_path)
            print(f"Saved checkpoint to {args.save_path}")
            if args.save_as_agent:
                agent_path = args.save_agent_path or os.path.join(os.path.dirname(args.save_path) or '.', 'student_agent.pth')
                agent_ck = {"model": model.state_dict()}
                try:
                    torch.save(agent_ck, agent_path)
                except Exception as e:
                    print(f"Warning: failed to save agent-style checkpoint: {e}")

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-file", required=True, help="Path to teacher rollouts (.npz file OR directory containing .npz files)")
    p.add_argument("--format", default="npz", choices=["npz", "npy", "pickle"], help="Format of rollout file")
    p.add_argument("--obs-key", default="obs", help="Key for observations in npz/pickle (default 'obs')")
    p.add_argument("--act-key", default="acts", help="Key for actions in npz/pickle (default 'acts')")
    p.add_argument("--loss", choices=["mse","l1"], default="mse", help="Loss to use for distillation (mse or l1)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256], help="Hidden layer sizes for student MLP")
    p.add_argument("--save-path", default="student_distill.pth", help="Where to save student checkpoint")
    p.add_argument("--save-interval", type=int, default=5)
    p.add_argument("--resume", default="", help="Path to resume checkpoint")
    p.add_argument("--normalize", action="store_true", help="Normalize observations and actions by dataset mean/std")
    p.add_argument("--save-as-agent", action='store_true', help='Also save an agent-style checkpoint containing {"model": state_dict} for train.py compatibility')
    p.add_argument("--save-agent-path", default='', help='Path to write agent-style checkpoint when --save-as-agent is used')
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    p.add_argument("--use-a2c", action='store_true', help='Use ActorCritic-style A2C network for student (larger network)')
    p.add_argument("--teacher-ckpt", default='', help='Path to teacher checkpoint for latent supervision (optional)')
    p.add_argument("--latent-weight", type=float, default=0.0, help='Weight for latent reconstruction loss (0 to disable)')
    p.add_argument("--use-transformer", action='store_true', help='Use temporal Transformer student consuming qpos+previous actions (seq input)')
    p.add_argument("--seq-len", type=int, default=30, help='Sequence length (number of time steps)')
    p.add_argument("--qpos-dim", type=int, default=22, help='Number of dimensions corresponding to qpos in the observation vector')
    p.add_argument("--trans-dim", type=int, default=128, help='Transformer model dimension')
    p.add_argument("--trans-heads", type=int, default=4, help='Transformer attention heads')
    p.add_argument("--trans-layers", type=int, default=2, help='Number of Transformer encoder layers')

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)