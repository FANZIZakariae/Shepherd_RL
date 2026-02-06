import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

from collections import deque
import random
import cv2

from torchvision import transforms
import os

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((84, 84)),
    transforms.ToTensor()  # converts to [C,H,W] float32
])

N_ACTIONS = 64
ANGLES = np.linspace(-np.pi, np.pi, N_ACTIONS)

def render_env_to_rgb(env, img_size=84):
    """
    Render a ShepherdEnv object to an RGB image (numpy array)
    
    Parameters:
    - env: ShepherdEnv instance
    - img_size: output image size (square)
    
    Returns:
    - image: np.array of shape (img_size, img_size, 3), dtype=np.uint8
    """
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # white background

    def world_to_pixel(pos):
        """
        Convert environment coordinates (-1 to 1) to pixel coordinates
        """
        x = int((pos[0] + 1) / 2 * (img_size - 1))
        y = int((1 - (pos[1] + 1) / 2) * (img_size - 1))  # flip y-axis for image
        return x, y

    # Draw goal
    gx, gy = world_to_pixel(env.goal)
    cv2.circle(canvas, (gx, gy), radius=5, color=(0, 255, 0), thickness=-1)  # green

    # Draw shepherd
    sx, sy = world_to_pixel(env.shepherd)
    cv2.circle(canvas, (sx, sy), radius=5, color=(255, 0, 0), thickness=-1)  # red

    # Draw obstacle
    ox, oy = world_to_pixel(env.obstacle)
    cv2.circle(canvas, (ox, oy), radius=5, color=(0, 0, 0), thickness=-1)  # black

    # Draw sheep
    for s in env.sheep:
        x, y = world_to_pixel(s)
        cv2.circle(canvas, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # blue

    return canvas

class ImageQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = x / 255.0  # normalize image
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure torch tensors and correct layout
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)

        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state)

        # Convert HWC → CHW if needed
        if state.ndim == 3 and state.shape[-1] == 3:
            state = state.permute(2, 0, 1)

        if next_state.ndim == 3 and next_state.shape[-1] == 3:
            next_state = next_state.permute(2, 0, 1)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)
    

class ImageDQNAgent:
    def __init__(
        self,
        n_actions,
        lr=1e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=200_000,
        device="cpu"
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma

        self.q_net = ImageQNetwork(n_actions).to(device)
        self.target_net = ImageQNetwork(n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

    def select_action(self, state,writer=None):
        self.steps += 1

        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps / self.eps_decay)

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0).to(self.device))

        if writer is not None:
            writer.add_scalar("train/epsilon", eps, self.steps)
        return q_values.argmax(dim=1).item()
    
    def train_step(self, batch_size=32,writer=None):
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions)

        # Bellman target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        writer.add_scalar(
                            "debug/q_mean",
                            q_values.mean().item(),
                            self.steps
                        )

        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


@torch.no_grad()
def evaluate_agent(env, agent, episodes=5):
    total_rewards = []
    total_lengths = []

    for _ in range(episodes):
        state = env.reset()
        state = render_env_to_rgb(env)
        state = torch.from_numpy(state).float().permute(2,0,1)
        done = False
        ep_reward = 0
        ep_len = 0

        while not done:
            q_values = agent.q_net(state.unsqueeze(0).to(agent.device))
            action_idx = q_values.argmax(dim=1).item()
            angle = ANGLES[action_idx]

            # state, reward, done, _ = env.step(angle)

            _, reward, done, _ = env.step([angle])
            next_state = render_env_to_rgb(env)
            next_state = torch.from_numpy(next_state).float().permute(2,0,1)
            state = next_state

            ep_reward += reward
            ep_len += 1

        total_rewards.append(ep_reward)
        total_lengths.append(ep_len)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(total_lengths),
    }



def train_image_dqn(
    env,
    eval_env,
    agent,
    episodes=1000,
    batch_size=64,
    target_update=1000,
    eval_every=1,
    eval_episodes=10
    ):

    log_dir = f"logs/image_dqn_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    # Create models folder if it does not exist
    os.makedirs("models", exist_ok=True)

    step_count = 0
    global_step = 0
    mean_length = float('inf')

    for ep in range(episodes):
        state = env.reset()
        state = render_env_to_rgb(env)           # H x W x C numpy
        state = torch.from_numpy(state).float().permute(2,0,1)  # C x H x W
        done = False
        ep_reward = 0
        ep_losses = []

        while not done:
            action_idx = agent.select_action(state,writer)
            angle = ANGLES[action_idx]

            _, reward, done, _ = env.step([angle])

            next_state = render_env_to_rgb(env)       # H x W x C
            next_state = torch.from_numpy(next_state).float().permute(2,0,1)  # C x H x W

            # Push image states to replay buffer
            agent.replay.push(state, action_idx, reward, next_state, done)

            # Update current state
            state = next_state


            loss = agent.train_step(batch_size,writer)

            if loss is not None:
                ep_losses.append(loss)
                writer.add_scalar("train/loss", loss, global_step)

            ep_reward += reward
            global_step += 1
            step_count += 1

            if step_count % target_update == 0:
                agent.update_target()

        # --- Training episode logs ---
        writer.add_scalar("train/episode_reward", ep_reward, ep)
        if ep_losses:
            writer.add_scalar(
                "train/episode_loss_mean",
                np.mean(ep_losses),
                ep
            )

        # --- Evaluation ---
        if ep % eval_every == 0:
            eval_stats = evaluate_agent(
                eval_env, agent, eval_episodes
            )

            writer.add_scalar(
                "eval/mean_reward",
                eval_stats["mean_reward"],
                ep
            )
            writer.add_scalar(
                "eval/std_reward",
                eval_stats["std_reward"],
                ep
            )
            writer.add_scalar(
                "eval/mean_ep_length",
                eval_stats["mean_length"],
                ep
            )

            print(
                f"[Eval]\t Ep {ep:4d} | "
                f"Reward: {eval_stats['mean_reward']:.2f} ± "
                f"{eval_stats['std_reward']:.2f}"
                f" | Length: {eval_stats['mean_length']:.0f}"
            )

            if eval_stats['mean_length'] < mean_length:
                mean_length = eval_stats['mean_length']
                torch.save(agent.q_net.state_dict(), f"models/agent_DQN_best.pth")

        print(f"[Train] Ep {ep:4d} | Reward: {ep_reward:.2f}")

        # Save the agent's Q-network
        torch.save(agent.q_net.state_dict(), f"models/agent_DQN_last.pth")

    return agent.q_net

    writer.close()