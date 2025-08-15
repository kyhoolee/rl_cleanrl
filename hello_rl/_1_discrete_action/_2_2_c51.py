import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    log_folder: str = "runs"
    """the folder to save all the logs and models"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 101
    """the number of atoms for the value distribution"""
    v_min: float = -100
    """the lower bound of the value distribution"""
    v_max: float = 100
    """the upper bound of the value distribution"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 500
    """the timesteps to update the target network"""
    batch_size: int = 128
    """the batch size for training"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total_timesteps` for epsilon to decay"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Tạo và cấu hình môi trường gym.
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

class QNetwork(nn.Module):
    """
    Mạng nơ-ron cho C51. Nó dự đoán một phân phối xác suất (PMF)
    cho mỗi hành động, thay vì một giá trị Q duy nhất.
    """
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        # Các "nguyên tử" (atoms) là các giá trị rời rạc mà phân phối giá trị có thể nhận
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n_actions = env.single_action_space.n
        
        # Mạng chính để tính toán logits cho mỗi hành động và mỗi nguyên tử
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n_actions * n_atoms),
        )

    def forward(self, x):
        """Tính toán logits cho tất cả các hành động và nguyên tử."""
        return self.network(x)

    def get_action_and_pmf(self, x, action=None):
        """
        Dựa vào đầu vào `x` (trạng thái), hàm này trả về hành động tốt nhất
        và phân phối xác suất (PMF) tương ứng.
        """
        logits = self.network(x)
        # Chuyển đổi logits thành PMF (phân phối xác suất)
        pmfs = torch.softmax(logits.view(len(x), self.n_actions, self.n_atoms), dim=2)
        # Tính toán Q-values bằng cách lấy tổng của PMF * atoms
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            # Chọn hành động có Q-value cao nhất
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Lịch trình giảm epsilon tuyến tính cho chiến lược epsilon-greedy.
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def projection(rewards, dones, next_pmfs, atoms, v_min, v_max, n_atoms, gamma):
    """
    Hàm chiếu (projection) chính trong C51.
    Nó chiếu phân phối giá trị của trạng thái tiếp theo về phân phối của trạng thái hiện tại.
    """
    with torch.no_grad():
        # Tính toán giá trị đích (target values)
        next_atoms = rewards + gamma * atoms * (1 - dones)
        
        # Clamp các giá trị vào khoảng [v_min, v_max]
        tz = next_atoms.clamp(v_min, v_max)

        # Tính toán các chỉ số rời rạc cho phép chiếu
        delta_z = atoms[1] - atoms[0]
        b = (tz - v_min) / delta_z
        l = b.floor().clamp(0, n_atoms - 1).long()
        u = b.ceil().clamp(0, n_atoms - 1).long()
        
        # Tính toán trọng số để phân tán xác suất (weights for distributing probabilities)
        d_m_l = (u.float() + (l == u).float() - b) * next_pmfs
        d_m_u = (b - l.float()) * next_pmfs
        
        # Tạo phân phối đích (target distribution) bằng cách phân tán xác suất
        target_pmfs = torch.zeros_like(next_pmfs)
        for i in range(target_pmfs.size(0)):
            target_pmfs[i].index_add_(0, l[i], d_m_l[i])
            target_pmfs[i].index_add_(0, u[i], d_m_u[i])
    return target_pmfs

def debug_print(global_step, loss, q_values, sps):
    """
    In các thông tin debug ra console.
    """
    print(f"--- Debug at step {global_step} ---")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Avg Q-value: {q_values.mean().item():.4f}")
    print(f"  Steps per Second (SPS): {sps}")
    print("-" * 30)

def main():
    args = tyro.cli(Args)
    assert args.num_envs == 1, "Vectorized envs are not supported at the moment."
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # SETUP LOGGING
    writer = SummaryWriter(f"{args.log_folder}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # SETUP SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # SETUP ENV
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action space is supported."

    # SETUP AGENT
    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # SETUP REPLAY BUFFER
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    
    # START THE GAME LOOP
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # 1. EXPLORE: Hành động dựa trên chính sách epsilon-greedy
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = q_network.get_action_and_pmf(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # 2. STORE TRANSITION: Thực hiện hành động và lưu trữ dữ liệu vào replay buffer
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Ghi lại phần thưởng nếu một episode kết thúc
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        
        obs = next_obs

        # 3. TRAINING: Huấn luyện mạng Q
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                # Tính toán phân phối đích (target distribution)
                with torch.no_grad():
                    # Lấy PMF của hành động tốt nhất từ target network
                    _, next_pmfs = target_network.get_action_and_pmf(data.next_observations)
                    
                    # Áp dụng projection (chiếu) để có target_pmfs
                    target_pmfs = projection(
                        data.rewards, data.dones, next_pmfs, target_network.atoms, 
                        args.v_min, args.v_max, args.n_atoms, args.gamma
                    )

                # Tính toán loss (KL-divergence)
                # Dùng old_pmfs từ q_network để so sánh với target_pmfs
                _, old_pmfs = q_network.get_action_and_pmf(data.observations, data.actions.flatten())
                # Dùng KL-divergence thay vì cross-entropy để tính loss
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                # Cập nhật mạng chính
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # LOGGING
                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    writer.add_scalar("charts/SPS", sps, global_step)
                    debug_print(global_step, loss, old_val, sps)

            # 4. CẬP NHẬT TARGET NETWORK
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # CLEANUP & SAVE MODEL
    envs.close()
    writer.close()
    
    if args.save_model:
        model_path = f"{args.log_folder}/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # EVALUATION (nếu có)
        # from cleanrl_utils.evals.c51_eval import evaluate
        # episodic_returns = evaluate(...)
        # ...
        
        # UPLOAD MODEL (nếu có)
        # from cleanrl_utils.huggingface import push_to_hub
        # ...

if __name__ == "__main__":
    main()