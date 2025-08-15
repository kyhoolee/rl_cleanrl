# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
from cleanrl_utils.evals.dqn_eval import evaluate
from cleanrl_utils.huggingface import push_to_hub


@dataclass
class Args:
    """Class để định nghĩa các tham số (hyperparameters) của thuật toán."""
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

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Tạo môi trường Gym.
    Hàm này được dùng để khởi tạo môi trường và các wrapper (ví dụ: RecordVideo).
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

# LOGIC THUẬT TOÁN: Khởi tạo agent
class QNetwork(nn.Module):
    """
    Định nghĩa kiến trúc mạng nơ-ron cho Q-Network.
    Nó nhận trạng thái (state) và trả về Q-value cho mỗi hành động.
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """
    Hàm này tính toán giá trị epsilon giảm dần theo thời gian.
    Epsilon là xác suất để tác tử thực hiện hành động ngẫu nhiên (khám phá).
    """
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def setup_training(args: Args):
    """
    Tập hợp tất cả các bước khởi tạo cần thiết.
    - Thiết lập W&B và TensorBoard.
    - Thiết lập random seed để đảm bảo tính tái lập.
    - Khởi tạo môi trường, Q-Network, Target Network, Replay Buffer.
    """
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "chỉ hỗ trợ không gian hành động rời rạc"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    return envs, q_network, optimizer, target_network, rb, device, writer, run_name

def get_action_and_explore(q_network, obs, epsilon, envs, device):
    """
    Lựa chọn hành động dựa trên chiến lược epsilon-greedy.
    - Với xác suất epsilon, tác tử chọn hành động ngẫu nhiên (khám phá).
    - Với xác suất 1 - epsilon, tác tử chọn hành động tốt nhất theo Q-Network (khai thác).
    """
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # print(f"  > Khám phá: Chọn hành động ngẫu nhiên: {actions}")
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        # print(f"  > Khai thác: Chọn hành động tốt nhất: {actions}")
    return actions

def train_q_network(q_network, optimizer, target_network, rb, args, device, writer, global_step):
    """
    Thực hiện một bước huấn luyện cho Q-Network.
    - Lấy một lô (batch) dữ liệu từ replay buffer.
    - Tính toán TD Target (phần thưởng cộng với giá trị của trạng thái tiếp theo).
    - Tính toán hàm mất mát (loss) MSE giữa TD Target và Q-value hiện tại.
    - Thực hiện backpropagation và cập nhật trọng số.
    """
    data = rb.sample(args.batch_size)
    with torch.no_grad():
        # Lấy Q-value lớn nhất từ Target Network
        target_max, _ = target_network(data.next_observations).max(dim=1)
        # Tính toán TD Target (Mục tiêu của hàm Q)
        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
    
    # Lấy Q-value hiện tại từ Q-Network chính
    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
    loss = F.mse_loss(td_target, old_val)
    
    if global_step % 100 == 0:
        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        # print(f"  > BƯỚC HUẤN LUYỆN: global_step={global_step}, loss={loss.item():.4f}")
    
    # Tối ưu hóa mô hình
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def update_target_network(target_network, q_network, args, global_step):
    """
    Cập nhật trọng số của Target Network.
    - Target Network được cập nhật định kỳ để ổn định quá trình huấn luyện.
    """
    if global_step % args.target_network_frequency == 0:
        for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
            target_param.data.copy_(
                args.tau * q_param.data + (1.0 - args.tau) * target_param.data
            )
        # print(f"  > CẬP NHẬT: Target network đã được cập nhật ở global_step={global_step}")

def run_training_loop(args, envs, q_network, optimizer, target_network, rb, device, writer, run_name):
    """
    Vòng lặp chính để huấn luyện tác tử.
    - Lặp qua từng bước thời gian (global_step).
    - Chọn hành động, tương tác với môi trường, và lưu dữ liệu.
    - Cập nhật trọng số Q-Network và Target Network.
    """
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # 1. Lựa chọn hành động dựa trên epsilon-greedy
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        actions = get_action_and_explore(q_network, obs, epsilon, envs, device)

        # 2. Thực thi hành động và quan sát kết quả
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # 3. Ghi lại kết quả và lưu vào replay buffer
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Thay đổi cốt lõi: Bây giờ info đã là một dictionary hợp lệ
                # và `episode` đã là một dictionary bên trong nó.
                if info and "episode" in info:
                    # Sửa lỗi Type Error ở đây:
                    # Chúng ta truy cập giá trị đầu tiên của các mảng NumPy.
                    episodic_return = info['episode']['r'][0]
                    episodic_length = info['episode']['l'][0]
                    
                    print(f"global_step={global_step}, episodic_return={episodic_return:.2f}, episodic_length={episodic_length}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
        
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # 4. Huấn luyện Q-Network nếu đã vượt qua ngưỡng khởi đầu
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                train_q_network(q_network, optimizer, target_network, rb, args, device, writer, global_step)
            
            # 5. Cập nhật Target Network
            update_target_network(target_network, q_network, args, global_step)

        if global_step > args.learning_starts and global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            print(f"SPS (Steps per second): {sps}")


def evaluate_and_save_model(args, envs, q_network, device, run_name):
    """
    Đánh giá và lưu mô hình cuối cùng.
    """
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"Mô hình đã được lưu tại: {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        # Ghi kết quả đánh giá vào TensorBoard
        writer = SummaryWriter(f"runs/{run_name}")
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        
        if args.upload_model:
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    envs, q_network, optimizer, target_network, rb, device, writer, run_name = setup_training(args)
    run_training_loop(args, envs, q_network, optimizer, target_network, rb, device, writer, run_name)
    evaluate_and_save_model(args, envs, q_network, device, run_name)

    envs.close()
    writer.close()
