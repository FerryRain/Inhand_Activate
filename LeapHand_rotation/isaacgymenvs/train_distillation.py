# train.py
# Script to train policies in Isaac Gym
# ... (Copyright header remains same) ...

import datetime
import isaacgym

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym
import sys
import numpy as np 
import torch       
import random
import atexit
import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reformat import omegaconf_to_dict, print_dict
from utils.utils import set_np_formatting, set_seed

# =================================================================================
# Data Collector Wrapper (Batch Saving Version + Target Recording)
# =================================================================================
class DataCollectorWrapper:
    """
    Wraps the environment to collect successful trajectories.
    Saves data in batches to avoid OOM and allow resume.
    Records: obs, actions, targets, dones
    """
    def __init__(self, env, total_samples=100000, save_path="collected_data.npz", 
                 threshold=700.0, samples_per_file=200000):
        self.env = env
        self.target_samples = total_samples
        self.threshold = threshold
        self.samples_per_file = samples_per_file
        
        # === 路径处理逻辑 ===
        input_dir = os.path.dirname(save_path)
        input_filename = os.path.basename(save_path)
        exp_name = os.path.splitext(input_filename)[0]

        if input_dir == "": input_dir = "."
            
        self.output_folder = os.path.join(input_dir, exp_name)
        os.makedirs(self.output_folder, exist_ok=True)

        self.base_path = os.path.join(self.output_folder, "data")
        
        self.part_counter = 0

        print(f"[DataCollector] Initialized.")
        print(f"  Target Total: {self.target_samples}")
        print(f"  Batch Size:   {self.samples_per_file}")
        print(f"  Output Dir:   {self.output_folder}/")
        
        # === 显式暴露 Gym 属性 ===
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        if hasattr(env, 'num_states'): self.num_states = env.num_states
        if hasattr(env, 'state_space'): self.state_space = env.state_space

        self.current_total_samples = 0
        self.collected_samples_in_memory = 0
        
        self.buffers = [[] for _ in range(self.num_envs)]
        self.current_episode_rewards = torch.zeros(self.num_envs, device=self.env.device)
        
        # === 数据存储列表 ===
        self.saved_obs = []
        self.saved_actions = []
        self.saved_targets = []  # [新增] 用于存储 target
        self.saved_dones = []
        
        self.last_obs = None
        self._is_saving = False

        # === 安全机制 ===
        atexit.register(self.flush_remaining)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n[DataCollector] Caught Ctrl+C! Saving remaining data...")
        self.flush_remaining()
        sys.exit(0)

    def reset(self):
        obs_dict = self.env.reset()
        if isinstance(obs_dict, dict) and 'obs' in obs_dict:
            self.last_obs = obs_dict['obs'].clone()
        else:
            self.last_obs = obs_dict.clone()
        return obs_dict

    def step(self, actions):
        # 1. Prepare current observation
        with torch.no_grad():
            current_obs_np = self.last_obs.cpu().numpy()
            actions_np = actions.cpu().numpy()
        
        # 2. Step environment
        obs_dict, rews, dones, info = self.env.step(actions)
        
        # [新增] 获取当前步的 Target
        # 在 IsaacGymEnvs 中，env 可能是被包裹的，我们需要找到底层的 task 来获取 cur_targets
        # 你的环境代码中 cur_targets 是 tensor
        raw_targets = None
        if hasattr(self.env, "cur_targets"):
            raw_targets = self.env.cur_targets
        elif hasattr(self.env, "task") and hasattr(self.env.task, "cur_targets"):
            # 如果被 RLGPUEnv 包裹
            raw_targets = self.env.task.cur_targets
        
        # 安全检查：如果没有找到 cur_targets (例如环境代码不匹配)，给一个零矩阵避免报错
        if raw_targets is None:
            # 假设 target 维度和 action 维度或者 dof 维度一致，这里暂用 action 维度兜底，最好确保环境中有 cur_targets
            print("[Warning] Could not find 'cur_targets' in env. Filling with zeros.")
            raw_targets = torch.zeros_like(actions) 

        # 3. Collect Data
        with torch.no_grad():
            targets_np = raw_targets.cpu().numpy() # 转为 numpy
            self.current_episode_rewards += rews
            dones_cpu = dones.cpu().numpy()
            
            for i in range(self.num_envs):
                # [修改] Buffer 现在存储 (obs, act, target)
                self.buffers[i].append((current_obs_np[i], actions_np[i], targets_np[i]))
                
                if dones_cpu[i]:
                    final_reward = self.current_episode_rewards[i].item()
                    
                    if final_reward > self.threshold:
                        # Success trajectory
                        traj_len = len(self.buffers[i])
                        # [修改] 解包三个变量
                        traj_obs, traj_acts, traj_targets = zip(*self.buffers[i])
                        
                        self.saved_obs.extend(traj_obs)
                        self.saved_actions.extend(traj_acts)
                        self.saved_targets.extend(traj_targets) # [新增] 保存 target
                        
                        dones_arr = np.zeros(traj_len, dtype=bool)
                        dones_arr[-1] = True
                        self.saved_dones.extend(dones_arr)
                        
                        self.collected_samples_in_memory += traj_len
                        
                        # LOGGING
                        total_now = self.current_total_samples + self.collected_samples_in_memory
                        if self.collected_samples_in_memory % 5000 < traj_len: 
                            print(f"[Collecting] Found Traj (R={final_reward:.1f}, Len={traj_len}). "
                                  f"Total Progress: {total_now}/{self.target_samples}")

                        # === BATCH SAVE CHECK ===
                        if len(self.saved_obs) >= self.samples_per_file:
                            self.save_batch()

                    self.buffers[i] = []
                    self.current_episode_rewards[i] = 0.0

        # 4. Check Exit
        total_collected = self.current_total_samples + self.collected_samples_in_memory
        if total_collected >= self.target_samples:
            print(f"[DataCollector] Target reached ({total_collected}). Exiting.")
            sys.exit(0) 

        # 5. Update last_obs
        if isinstance(obs_dict, dict) and 'obs' in obs_dict:
            self.last_obs = obs_dict['obs'].clone()
        else:
            self.last_obs = obs_dict.clone()

        return obs_dict, rews, dones, info

    def save_batch(self):
        """保存当前内存中的数据到一个新文件，并清空内存"""
        if self._is_saving or len(self.saved_obs) == 0:
            return
        
        self._is_saving = True
        
        filename = f"{self.base_path}_part_{self.part_counter}.npz"
        print(f"\n[DataCollector] >>> Saving Batch {self.part_counter} to {filename} ...")
        
        try:
            # [修改] 保存 targets 数组
            np.savez_compressed(
                filename, 
                obs=np.array(self.saved_obs), 
                acts=np.array(self.saved_actions), 
                targets=np.array(self.saved_targets), # [新增]
                dones=np.array(self.saved_dones)
            )
            print(f"[DataCollector] >>> Batch {self.part_counter} Saved. (Size: {len(self.saved_obs)})")
            
            # 更新计数器
            self.current_total_samples += len(self.saved_obs)
            self.part_counter += 1
            
            # 清空内存
            self.saved_obs = []
            self.saved_actions = []
            self.saved_targets = [] # [新增] 清空
            self.saved_dones = []
            self.collected_samples_in_memory = 0
            
        except Exception as e:
            print(f"[DataCollector] ERROR SAVING BATCH: {e}")
        
        self._is_saving = False

    def flush_remaining(self):
        """在程序退出时调用，保存内存中剩余的数据"""
        if self.collected_samples_in_memory > 0:
            print(f"\n[DataCollector] Flushing remaining {self.collected_samples_in_memory} samples...")
            self.save_batch()
        print("[DataCollector] Cleanup complete.")

    def __getattr__(self, name):
        return getattr(self.env, name)
# =================================================================================

## OmegaConf & Hydra Config
# ... (Below code remains exactly the same as your original script) ...

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # ... (Keep the rest of the original launch_rlg_hydra function unchanged) ...
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # Force Test Mode
    collect_data = cfg.get("collect_data", False)
    if collect_data:
        print("\n" + "="*50)
        print(" DATA COLLECTION MODE ACTIVATED")
        print("="*50 + "\n")
        
        cfg.test = True 
        cfg.train.params.config.minibatch_size = cfg.task.env.numEnvs 
        
        if not cfg.checkpoint:
            print("ERROR: Data collection requires a pre-trained checkpoint.")
            sys.exit(1)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

        if collect_data:
            target_samples = cfg.get("collect_samples", 10000000) 
            save_name = cfg.get("save_name", f"data_{cfg.task_name}.npz")
            reward_threshold = cfg.get("reward_threshold", 700.0) 
            samples_per_file = cfg.get("samples_per_file", 500000) 

            envs = DataCollectorWrapper(
                envs, 
                total_samples=target_samples, 
                save_path=save_name,
                threshold=reward_threshold,
                samples_per_file=samples_per_file
            )

        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {'vecenv_type': 'RLGPU', 'env_creator': create_env_thunk})

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
        return runner

    time_prefix = time_str + '-' + str(random.randint(0, 100000))
    need_set_prefix = False

    if hasattr(cfg.train.params.config, 'user_prefix'):
        prefix = cfg.train.params.config.prefix = cfg.train.params.config.user_prefix \
                                                  + cfg.train.params.config.auto_prefix \
                                                  + time_prefix
    else:
        prefix = time_prefix
        need_set_prefix = True

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    if need_set_prefix:
        rlg_config_dict['params']['config']['prefix'] = prefix
    
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    experiment_dir = os.path.join(experiment_dir, prefix)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint' : cfg.checkpoint,
        'sigma' : None
    })

    if cfg.wandb_activate and rank == 0:
        wandb.finish()

if __name__ == "__main__":
    launch_rlg_hydra()