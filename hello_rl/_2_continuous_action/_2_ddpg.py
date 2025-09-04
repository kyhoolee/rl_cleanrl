import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """tên của experiment"""
    seed: int = 1
    """seed của experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda sẽ được bật theo mặc định"""
    track: bool = False
    """if toggled, experiment sẽ được theo dõi với Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """tên project của wandb"""
    wandb_entity: str = None
    """thực thể (team) của wandb"""
    capture_video: bool = False
    """có ghi lại video performance của agent không"""
    save_model: bool = False
    """có lưu model không"""
    upload_model: bool = False
    """có upload model lên Hugging Face không"""
    hf_entity: str = ""
    """tên người dùng hoặc tổ chức trên Hugging Face Hub"""
    log_folder: str = "runs"
    """thư mục để lưu log TensorBoard"""

    # Các đối số cụ thể cho thuật toán
    env_id: str = "Hopper-v4"
    """id của môi trường"""
    total_timesteps: int = 1000000
    """tổng số timesteps của experiment"""
    learning_rate: float = 3e-4
    """learning rate của optimizer"""
    buffer_size: int = int(1e6)
    """kích thước buffer replay memory"""
    gamma: float = 0.99
    """hệ số discount factor gamma"""
    tau: float = 0.005
    """hệ số làm mượt cho mạng target (soft update)"""
    batch_size: int = 256
    """kích thước batch lấy mẫu từ replay memory"""
    exploration_noise: float = 0.1
    """mức độ nhiễu khám phá (exploration noise)"""
    learning_starts: int = 25e3
    """timesteps để bắt đầu học"""
    policy_frequency: int = 2
    """tần suất cập nhật policy (delayed policy update)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Hàm factory để tạo môi trường.
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
    Critic Network (Mạng Q).

    Đầu vào: trạng thái (state) và hành động (action).
    Đầu ra: giá trị Q (Q-value).
    """
    def __init__(self, env):
        super().__init__()
        # Kích thước đầu vào = kích thước trạng thái + kích thước hành động
        input_dim = np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    """
    Actor Network (Mạng chính sách - Policy).

    Đầu vào: trạng thái (state).
    Đầu ra: hành động (action) **mang tính xác định**.
    """
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        
        # Rescaling hành động để phù hợp với không gian hành động của môi trường
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))  # tanh để đưa giá trị về [-1, 1]
        return x * self.action_scale + self.action_bias # Rescale về không gian hành động thực tế


def setup_ddpg(args, device):
    """
    Khởi tạo các thành phần chính của DDPG: môi trường, các mạng, buffer, và optimizers.
    """
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # Thiết lập logging
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
    writer = SummaryWriter(f"{args.log_folder}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Thiết lập seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Thiết lập môi trường
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "DDPG chỉ hỗ trợ không gian hành động liên tục"

    # Khởi tạo các mạng
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    
    # Khởi tạo mạng target với cùng trọng số của mạng chính
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    # Khởi tạo optimizers
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    # Khởi tạo Replay Buffer
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    
    return envs, actor, qf1, qf1_target, target_actor, q_optimizer, actor_optimizer, rb, writer, run_name


def train_ddpg(
        args, envs, actor, qf1, qf1_target, 
        target_actor, q_optimizer, actor_optimizer, 
        rb, writer, device
        ):
    """
    Vòng lặp huấn luyện chính của DDPG.
    """
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Bước 1: Chọn hành động (exploration)
        if global_step < args.learning_starts:
            # Chọn hành động ngẫu nhiên để khởi tạo buffer
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Chọn hành động từ Actor Network và thêm nhiễu
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                # Clip hành động để đảm bảo nằm trong giới hạn của môi trường
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # Bước 2: Thực hiện hành động và lưu trữ kinh nghiệm
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Ghi lại phần thưởng (rewards) của episode
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break
        
        # Xử lý next_obs cho trường hợp episode kết thúc do timeout
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        
        # Thêm kinh nghiệm vào replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Bước 3: Học từ Replay Buffer (chỉ sau khi đủ dữ liệu)
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            
            # Tính Q-value target
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
            
            # Cập nhật Critic (QNetwork)
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            # Cập nhật Actor (Policy Network)
            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update các mạng target
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Ghi log
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

def main():
    args = tyro.cli(Args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    envs, actor, qf1, qf1_target, target_actor, q_optimizer, actor_optimizer, rb, writer, run_name = setup_ddpg(args, device)
    
    train_ddpg(
        args, envs, 
        actor, 
        qf1, qf1_target, target_actor, 
        q_optimizer, actor_optimizer, 
        rb, writer,
        device
        )
    
    # Xử lý sau khi huấn luyện (lưu model, upload,...)
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        # ... (phần eval và upload model) ...
    
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()