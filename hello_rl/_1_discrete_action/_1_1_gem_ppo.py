# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import itertools
import pandas as pd

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
    log_folder: str = "runs"
    """thư mục để lưu log của TensorBoard"""
    run_hyperparam_search: bool = False
    """Toggles whether to run a hyperparameter search instead of a single run"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    """Creates a single environment instance with wrappers."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initializes a layer's weights and biases."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """The PPO agent network, combining a critic (value) and an actor (policy)."""
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(envs.single_action_space.n)

        # The critic network estimates the value function V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # The actor network outputs the policy distribution pi(a|s)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        """Returns the state value V(s) from the critic network."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Calculates the action, log probability, entropy, and value for a given state.
        If an action is provided, it calculates its log probability instead of sampling.
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_dist(self, x):
        """A convenience method to get the action distribution for diagnostics."""
        logits = self.actor(x)
        return Categorical(logits=logits)


def run_rollout(agent, envs, args, device, obs, actions, logprobs, rewards, dones, values, global_step, writer, next_obs, next_done):
    """
    Performs the rollout phase to collect a batch of experiences from the environment.
    This function fills the `obs`, `actions`, `logprobs`, `rewards`, `dones`, and `values` tensors.
    """
    episodic_returns = []
    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # Policy diagnostics during rollout
        with torch.no_grad():
            dist = agent.get_dist(next_obs)
            if hasattr(dist, "probs"):
                probs = dist.probs
                max_prob = probs.max(dim=-1).values.mean().item()
                writer.add_scalar("policy/max_action_prob", max_prob, global_step)
                writer.add_scalar("policy/entropy_rollout", dist.entropy().mean().item(), global_step)
            writer.add_histogram("policy/action_hist", action.detach().cpu(), global_step)

        # Execute the game and log data.
        next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done_np = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward_np).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs_np).to(device), torch.Tensor(next_done_np).to(device)

        # Log episode returns
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    episodic_returns.append(info["episode"]["r"])
    return global_step, next_obs, next_done, episodic_returns


def calculate_advantages_and_returns(agent, next_obs, next_done, rewards, values, dones, args, device):
    """
    Calculates the advantages using Generalized Advantage Estimation (GAE)
    and the returns.
    """
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    return advantages, returns


def update_policy(agent, optimizer, args, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, global_step, writer):
    """
    Performs the policy and value network optimization using minibatches.
    """
    b_inds = np.arange(args.batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            # Get new logprobs, entropy, and values from the current policy
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            
            with torch.no_grad():
                # Calculate approx_kl for early stopping
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            # Normalize advantages if configured
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss (PPO-clip)
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # Total loss
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break
            
    # Logging final metrics for the iteration
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)

def run_single_experiment(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    final_returns = []

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            writer.add_scalar("charts/learning_rate", lrnow, global_step)

        # Rollout phase
        global_step, next_obs, next_done, episodic_returns = run_rollout(
            agent, envs, args, device, obs, actions, logprobs, rewards, dones, values, global_step, writer, next_obs, next_done
        )
        final_returns.extend(episodic_returns)
        
        # GAE calculation
        advantages, returns = calculate_advantages_and_returns(
            agent, next_obs, next_done, rewards, values, dones, args, device
        )

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # Policy update phase
        update_policy(
            agent, optimizer, args, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, global_step, writer
        )

        # Log SPS
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
    
    # Return the average of the last 10 episode returns as the final result
    if final_returns:
        return np.mean(final_returns[-10:])
    return 0.0

def run_hyperparameter_search(args):
    results = []
    hyperparam_grid = {
    "learning_rate": [2.5e-4, 1e-4],
    "num_minibatches": [4, 8],
    "update_epochs": [4, 8],
    "clip_coef": [0.1, 0.2],
    "ent_coef": [0.01, 0.005],
    "gamma": [0.99, 0.95],
    "gae_lambda": [0.95, 0.98],
}   

    keys = hyperparam_grid.keys()
    # Sử dụng itertools.product để tạo tất cả các tổ hợp
    param_combinations = list(itertools.product(*hyperparam_grid.values()))

    # Lặp qua tất cả các tổ hợp
    for i, values in enumerate(param_combinations):
        params = dict(zip(keys, values))
        
        # Tạo đối tượng args mới cho mỗi lần chạy
        current_args = Args(**vars(args)) # Bắt đầu với các giá trị mặc định
        for k, v in params.items():
            setattr(current_args, k, v)
        
        current_args.exp_name = f"search_{i+1:02d}"
        
        print(f"\n=======================================================")
        print(f"--- Bắt đầu chạy tìm kiếm {i+1}/{len(param_combinations)} ---")
        print(f"Tham số: {params}")
        print(f"=======================================================")
        
        # Chạy thí nghiệm và lấy kết quả cuối cùng
        final_return = run_single_experiment(current_args)
        
        # Lưu kết quả
        params["final_return"] = final_return
        results.append(params)
    
    # Lưu toàn bộ kết quả vào một file CSV
    df = pd.DataFrame(results)
    output_filename = f"hyperparam_search_results_{int(time.time())}.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n--- Tìm kiếm hoàn tất. Kết quả được lưu tại {output_filename} ---")
    print(df)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.run_hyperparam_search:
        run_hyperparameter_search(args)
    else:
        run_single_experiment(args)
