#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================================================
# 1. Dataset: Handles Frame Stacking automatically
# =========================================================================
class StackedMLPDataset(Dataset):
    def __init__(self, path, stack_num=4):
        # 注意：这里的 stack_num 参数实际上不再用于堆叠，仅用于兼容接口
        self.stack_num = stack_num
        self._load_data(path)

    def _load_data(self, path):
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.npz")))
        else:
            files = [path]
        
        all_obs = []
        all_acts = []

        print(f"Loading {len(files)} data files...")
        for f in tqdm(files):
            try:
                data = np.load(f)
                o = data['obs'] 
                a = data['actions']
                
                # --- 针对你特定数据形状的修复 ---
                # 你的数据是 (N, 1, 340)，我们需要变成 (N, 340)
                if o.ndim == 3 and o.shape[1] == 1:
                    o = o.squeeze(1) # 去掉中间的 1
                
                # 你的动作是 (N, 1, 22)，我们需要变成 (N, 22)
                if a.ndim == 3 and a.shape[1] == 1:
                    a = a.squeeze(1)
                # ------------------------------
                
                all_obs.append(o.astype(np.float32))
                all_acts.append(a.astype(np.float32))
                
            except Exception as e:
                print(f"Skipping {f}: {e}")

        self.raw_obs = np.concatenate(all_obs, axis=0)
        self.acts = np.concatenate(all_acts, axis=0)
        
        # 因为数据已经堆叠好了，所以所有索引都是有效的
        # 我们不再需要 valid_indices 来过滤跨越 Done 的窗口
        # (假设你的数据采集脚本在保存时已经处理好了 Done)
        self.valid_indices = np.arange(len(self.raw_obs))
        
        print(f"Total steps: {len(self.raw_obs)}. Shape: {self.raw_obs.shape}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 直接返回，不做任何堆叠
        return self.raw_obs[idx], self.acts[idx]

# =========================================================================
# 2. Model: Standard IsaacGym MLP
# =========================================================================
class SimpleMLP(nn.Module):
    # 修改默认 hidden_dims 以匹配你的 Teacher
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 256]): 
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU()) 
            prev = h
        self.head = nn.Linear(prev, output_dim)
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.head(self.net(x))

# =========================================================================
# 3. RL Games Compatibility Helpers
# =========================================================================
def load_teacher_weights(model, ckpt_path, device):
    """Loads weights from rl_games checkpoint into SimpleMLP"""
    print(f"Loading Teacher: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt['model']
    
    new_sd = {}
    # Mapping based on standard IsaacGym PPO structure
    new_sd['net.0.weight'] = sd['a2c_network.actor_mlp.0.weight']
    new_sd['net.0.bias']   = sd['a2c_network.actor_mlp.0.bias']
    new_sd['net.2.weight'] = sd['a2c_network.actor_mlp.2.weight']
    new_sd['net.2.bias']   = sd['a2c_network.actor_mlp.2.bias']
    new_sd['net.4.weight'] = sd['a2c_network.actor_mlp.4.weight']
    new_sd['net.4.bias']   = sd['a2c_network.actor_mlp.4.bias']
    new_sd['head.weight']  = sd['a2c_network.mu.weight']
    new_sd['head.bias']    = sd['a2c_network.mu.bias']
    
    model.load_state_dict(new_sd, strict=True)
    print("Teacher weights loaded successfully.")
    
    # Extract sigma for saving later (needed for rl_games)
    sigma = sd.get('a2c_network.sigma', None)
    return model, sigma

def save_fake_rlg_checkpoint(model, save_path, original_sigma=None):
    """Saves SimpleMLP back to rl_games format"""
    sd = model.state_dict()
    new_sd = {}
    
    # Reverse mapping
    new_sd['a2c_network.actor_mlp.0.weight'] = sd['net.0.weight']
    new_sd['a2c_network.actor_mlp.0.bias']   = sd['net.0.bias']
    new_sd['a2c_network.actor_mlp.2.weight'] = sd['net.2.weight']
    new_sd['a2c_network.actor_mlp.2.bias']   = sd['net.2.bias']
    new_sd['a2c_network.actor_mlp.4.weight'] = sd['net.4.weight']
    new_sd['a2c_network.actor_mlp.4.bias']   = sd['net.4.bias']
    new_sd['a2c_network.mu.weight']          = sd['head.weight']
    new_sd['a2c_network.mu.bias']            = sd['head.bias']
    
    if original_sigma is not None:
        new_sd['a2c_network.sigma'] = original_sigma
    else:
        # Dummy sigma if missing
        out_dim = sd['head.weight'].shape[0]
        new_sd['a2c_network.sigma'] = torch.zeros(out_dim)

    checkpoint = {
        'model': new_sd,
        'epoch': 0,
        'optimizer': {},
        'frame': 0
    }
    torch.save(checkpoint, save_path)
    print(f"Saved compatible checkpoint to: {save_path}")

# =========================================================================
# 4. Main Training
# =========================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    dataset = StackedMLPDataset(args.data_path, stack_num=args.stack_num)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Determine dimensions from data
    # raw_obs shape: [N, Obs_Dim]
    obs_dim = dataset.raw_obs.shape[1]
    act_dim = dataset.acts.shape[1]
    input_dim = obs_dim * args.stack_num
    
    print(f"Obs Dim: {obs_dim}, Stack: {args.stack_num}, Input Dim: {input_dim}, Act Dim: {act_dim}")
    
    print(f"Data Loaded: Obs Shape {dataset.raw_obs.shape}, Act Shape {dataset.acts.shape}")
    
    # 因为数据已经堆叠好了，所以 raw_obs.shape[1] 应该是 340 (即 85*4)
    # 我们不再强制检查 85
    input_dim = dataset.raw_obs.shape[1]
    act_dim = dataset.acts.shape[1]

    # Model
    model = SimpleMLP(
        input_dim, 
        act_dim, 
        hidden_dims=[512, 256, 256] # <--- 修正这里
    ).to(device)
    
    # Load Teacher
    model, sigma = load_teacher_weights(model, args.teacher_ckpt, device)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("Starting Fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for obs, act in pbar:
            obs, act = obs.to(device), act.to(device)
            
            pred = model(obs)
            loss = criterion(pred, act)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Save compatible checkpoint
        if (epoch+1) % args.save_interval == 0:
            save_path = f"finetuned_mlp_ep{epoch+1}.pth"
            save_fake_rlg_checkpoint(model, save_path, sigma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Folder containing real world .npz files")
    parser.add_argument("--teacher-ckpt", required=True, help="Path to original teacher .pth")
    parser.add_argument("--stack-num", type=int, default=4, help="Must match training stack (usually 4)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5)
    args = parser.parse_args()
    
    train(args)