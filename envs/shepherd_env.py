import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class ShepherdEnv(gym.Env):
    def __init__(
        self,
        n_sheep=1,
        world_size=1.0,
        goal_radius=0.7,
        obstacle_radius=0,
        sheep_repulsion_radius=0.2,
        shepherd_speed=0.05,
        max_steps=500,
        n_sheep_in_goal=0,
        sheep_jitter=0.0  # <--- NEW: Controls "Shaky" behavior (0.0 = stable, 0.1 = very shaky)
    ):
        super().__init__()

        self.n_sheep = n_sheep
        self.n_sheep_in_goal = min(n_sheep_in_goal, n_sheep)
        self.world_size = world_size
        self.goal_radius = goal_radius
        
        if obstacle_radius > 0.3:
            print("Warning: obstacle_radius too large, setting to 0.3")
            self.obstacle_radius = 0.3
        else:
            self.obstacle_radius = obstacle_radius
            
        self.repulsion_radius = sheep_repulsion_radius
        self.shepherd_speed = shepherd_speed
        self.max_steps = max_steps
        self.sheep_jitter = sheep_jitter # Store jitter strength

        # Action: shepherd orientation angle
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation: [sheep_rel_pos * n, goal_rel, obstacle_rel]
        obs_dim = 4 * self.n_sheep + 2 + 2 
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.steps = 0

        self.shepherd = np.random.uniform(-0.8, 0.8, size=2)
        self.obstacle = np.random.uniform(-0.8, 0.8, size=2)
        self.goal = np.random.uniform(-0.8, 0.8, size=2)

        while np.linalg.norm(self.obstacle - self.goal) < (self.goal_radius + self.obstacle_radius + 0.1):
            self.goal = np.random.uniform(-0.8, 0.8, size=2)
            self.obstacle = np.random.uniform(-0.8, 0.8, size=2)

        self.sheep = []
        self.sheep_locked_status = [False] * self.n_sheep 

        # 1. Spawn "Free" Sheep
        self.num_free_sheep = self.n_sheep - self.n_sheep_in_goal
        for i in range(self.num_free_sheep):
            s = np.random.uniform(-0.8, 0.8, size=2)
            while np.linalg.norm(s - self.goal) < self.goal_radius:
                s = np.random.uniform(-0.8, 0.8, size=2)
            self.sheep.append(s)
            
        # 2. Spawn "Pre-Solved" Sheep
        for i in range(self.n_sheep_in_goal):
            offset = np.random.uniform(-0.1, 0.1, size=2)
            s = self.goal + offset 
            self.sheep.append(s)
            self.sheep_locked_status[self.num_free_sheep + i] = True

        self.prev_goal_dist = self._mean_active_sheep_dist()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for s in self.sheep:
            sheep_rel = s - self.shepherd
            goal_rel_s = self.goal - s
            obs.extend(sheep_rel)
            obs.extend(goal_rel_s)

        goal_rel = self.goal - self.shepherd
        obs.extend(goal_rel)
        obstacle_rel = self.obstacle - self.shepherd
        obs.extend(obstacle_rel)

        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _mean_active_sheep_dist(self):
        active_dists = []
        for i, s in enumerate(self.sheep):
            if not self.sheep_locked_status[i]:
                active_dists.append(np.linalg.norm(s - self.goal))
        
        if not active_dists:
            return 0.0
        return np.mean(active_dists)

    def step(self, action):
        self.steps += 1
        self.prev_shepherd = self.shepherd.copy()

        # --- Shepherd dynamics ---
        angle_deg = float(np.clip(action[0]*180, -180.0, 180.0))
        angle_rad = np.deg2rad(angle_deg)
        move = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * self.shepherd_speed
        self.shepherd += move
        self.shepherd = np.clip(self.shepherd, -1.0, 1.0)

        # --- Sheep dynamics ---
        newly_locked_count = 0

        for i, s in enumerate(self.sheep):
            # If sheep is locked, it STAYS PUT.
            if self.sheep_locked_status[i]:
                continue

            # Check if it just entered the goal
            dist_to_goal = np.linalg.norm(s - self.goal)
            if dist_to_goal <= self.goal_radius:
                self.sheep_locked_status[i] = True
                newly_locked_count += 1
                continue 

            # Normal movement for active sheep
            sheep_move = np.zeros(2)
            vec = s - self.shepherd
            dist = np.linalg.norm(vec)
            
            # Repulsion from shepherd
            if dist < self.repulsion_radius:
                sheep_move += (vec / (dist + 1e-6)) * 0.05
            
            # --- NEW: APPLY JITTER (Level 3+) ---
            if self.sheep_jitter > 0:
                # Add random noise to movement (X and Y)
                noise = np.random.uniform(-self.sheep_jitter, self.sheep_jitter, size=2)
                sheep_move += noise
            # ------------------------------------

            new_pos = s + sheep_move
            if self.obstacle_radius > 0:
                if np.linalg.norm(np.clip(new_pos, -0.9, 0.9) - self.obstacle) <= (self.obstacle_radius + 0.05):
                     self.sheep[i] = np.clip(s - sheep_move * 0.5, -0.9, 0.9)
                else:
                     self.sheep[i] = np.clip(new_pos, -0.9, 0.9)
            else:
                self.sheep[i] = np.clip(new_pos, -0.9, 0.9)

        # ---------------------------------------------------------
        # --- REWARD SYSTEM ---
        # ---------------------------------------------------------
        reward = 0.0
        reward_components = {
            "Progress (Push)": 0.0,
            "Target Proximity": 0.0,
            "Goal Entry Bonus": 0.0,
            "Movement Cost": 0.0,
            "Step Cost": 0.0,
            "Win Bonus": 0.0,
            "Fail Penalty": 0.0
        }

        # 1. Goal Entry Bonus
        if newly_locked_count > 0:
            entry_bonus = 100.0 * newly_locked_count
            reward += entry_bonus
            reward_components["Goal Entry Bonus"] = entry_bonus

        # 2. Progress Reward
        curr_dist = self._mean_active_sheep_dist()
        if curr_dist > 0 and newly_locked_count == 0: 
            prog_reward = (self.prev_goal_dist - curr_dist) * 300.0
            reward += prog_reward
            reward_components["Progress (Push)"] = prog_reward
        self.prev_goal_dist = curr_dist

        # 3. Proximity Reward
        active_indices = [i for i, locked in enumerate(self.sheep_locked_status) if not locked]
        if active_indices:
            active_dists = [np.linalg.norm(self.sheep[i] - self.goal) for i in active_indices]
            furthest_active_idx = active_indices[np.argmax(active_dists)]
            target_sheep = self.sheep[furthest_active_idx]
            
            dist_shepherd_target = np.linalg.norm(target_sheep - self.shepherd)
            prox_reward = 5.0 * np.exp(-5.0 * dist_shepherd_target)
            reward += prox_reward
            reward_components["Target Proximity"] = prox_reward
        else:
            reward_components["Target Proximity"] = 0.0

        # 4. Movement Penalty
        shepherd_move_dist = np.linalg.norm(self.shepherd - self.prev_shepherd)
        move_penalty = -1.0 * np.exp(-100.0 * shepherd_move_dist)
        reward += move_penalty
        reward_components["Movement Cost"] = move_penalty

        # --- Termination ---
        terminated = False
        truncated = False

        if all(self.sheep_locked_status):
            win_bonus = 200.0 * self.n_sheep + 5 * (self.max_steps - self.steps)
            reward += win_bonus
            reward_components["Win Bonus"] = win_bonus
            terminated = True
            
        elif self.steps >= self.max_steps:
            truncated = True
            fail_pen = -10.0
            reward += fail_pen
            reward_components["Fail Penalty"] = fail_pen
        else:
            step_cost = -0.02
            reward += step_cost
            reward_components["Step Cost"] = step_cost

        return self._get_obs(), reward, terminated, truncated, {"reward_breakdown": reward_components}

    def render(self, mode='human'):
        if not hasattr(self, "screen") or self.screen is None:
            pygame.init()
            self.screen_size_px = 600
            self.screen = pygame.display.set_mode((self.screen_size_px, self.screen_size_px))
            pygame.display.set_caption("Shepherd Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        self.screen.fill((255, 255, 255))

        def to_px(pos):
            return ((pos + 1) * self.screen_size_px / 2).astype(int)

        # Draw goal
        pygame.draw.circle(
            self.screen, (0, 200, 0),
            to_px(self.goal), int(self.goal_radius * self.screen_size_px/2), width=2
        )

        # Draw obstacle
        if self.obstacle_radius > 0:
            pygame.draw.circle(
                self.screen, (100, 100, 100),
                to_px(self.obstacle), int(self.obstacle_radius * self.screen_size_px/ 2), width=0
            )

        # Draw sheep
        for i, s in enumerate(self.sheep):
            color = (0, 255, 0) if self.sheep_locked_status[i] else (0, 0, 0)
            pygame.draw.circle(
                self.screen, color,
                to_px(s), int(0.02 * self.screen_size_px), width=0
            )

        # Draw shepherd
        pygame.draw.circle(
            self.screen, (200, 0, 0),
            to_px(self.shepherd), int(0.03 * self.screen_size_px), width=0
        )

        text = f"Step: {self.steps}/{self.max_steps}"
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)