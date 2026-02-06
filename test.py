import argparse
import numpy as np
import os
import torch

from stable_baselines3 import PPO, A2C, TD3

from envs.shepherd_env import ShepherdEnv
from agents.rule_based_agent import RuleBasedShepherd,TipsyShepherd,LazyShepherd

from agents.CNN_QN import ImageDQNAgent,N_ACTIONS,render_env_to_rgb,ANGLES,transform


def load_agent(agentType: str, model_name: str, env: ShepherdEnv):
    """Load the selected agent."""
    if agentType == "ruleBase":
        print("A Rule-Based Shepherd Agent ...")
        return RuleBasedShepherd()
    if agentType == "tipsy":
        print("A Tipsy Shepherd Agent ...")
        return TipsyShepherd()
    if agentType == "lazy":
        print("A Lazy Shepherd Agent ...")
        return LazyShepherd()
    if agentType == "DQN" and os.path.exists(model_name):
        print("Using DQN Agent.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        agent = ImageDQNAgent(
                            n_actions=N_ACTIONS,
                            lr=1e-4,
                            gamma=0.99,
                            device=device
                        )
        agent.q_net.load_state_dict(torch.load(model_name, map_location=device))
        agent.q_net.eval()
        return agent

    if os.path.exists(model_name):
        print(f"Using {agentType} Agent.")
        return globals()[agentType].load(
            model_name,
            env=env,
            device="cpu",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def run_episode(env: ShepherdEnv, agent, model_type: str,display_flag=False) -> float:
    obs = env.reset()
    done = False
    reward = 0.0

    while not done:
        if model_type in ["ruleBase","lazy","tipsy"]:
            actions = agent.act(obs)
        elif model_type == "DQN":
            state = transform(render_env_to_rgb(env))
            with torch.no_grad():
                action_idx = agent.select_action(state)
                actions = [ANGLES[action_idx]]
        else:
            actions, _ = agent.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(actions)
        if display_flag :
            # print(f"Step {env.steps}: Action:{actions}, Reward: {reward:.2f}")
            env.render()

    return reward


def main():
    parser = argparse.ArgumentParser(description="Shepherd Environment Test Runner")

    parser.add_argument(
        "-a", "--agent_dir",
        type=str,
        default="models/",
        help="Agent model type or path to a saved model. Options: 'ruleBase', 'lazy', 'tipsy', or provide a valid file path.",
    )

    parser.add_argument(
        "-t", "--agentType",
        type=str,
        choices=["ruleBase", "lazy", "tipsy", "PPO", "DQN", "TD3"],
        default="ruleBase",
        help="Agent model type. Choose from: 'ruleBase', 'lazy', 'tipsy', 'PPO', 'A2C', 'TD3'.",
    )

    parser.add_argument(
        "-n", "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run. If greater than 1, renders are disabled and statistics are reported.",
    )

    parser.add_argument(
        "-s", "--num_sheep",
        type=int,
        default=1,
        help="Number of sheep in the environment.",
    )

    parser.add_argument(
        "-m", "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode.",
    )

    parser.add_argument(
        "-r", "--obstacle_radius",
        type=float,
        default=0.0,
        help="Radius of obstacles in the environment.",
    )

    parser.add_argument(
        "-g", "--goal_radius",
        type=float,
        default=0.7,
        help="Radius of the goal area in the environment.",
    )

    args = parser.parse_args()

    rewards = []

    for eps in range(1, args.num_episodes + 1):
        env = ShepherdEnv(n_sheep=args.num_sheep,
                        max_steps=args.max_steps,
                        obstacle_radius=args.obstacle_radius,
                        goal_radius=args.goal_radius)
        print(
            f"Environment initialized with "
            f"{env.n_sheep} sheep for episode {eps}"
        )

        agent = load_agent(args.agentType,args.agent_dir, env)
        final_reward = run_episode(env, agent, args.agentType,display_flag=(args.num_episodes==1))

        rewards.append(final_reward)
        if args.num_episodes == 1:
            print(f"Episode {eps} finished. Final reward: {final_reward:.2f}")
        env.close()
        
    if args.num_episodes > 1:
        print(f"\nRan {args.num_episodes} episodes.")
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        ci = 1.96 * (std_reward / np.sqrt(len(rewards)))  # 95% confidence interval

        print(f"Average reward: {mean_reward:.2f} ± {ci:.2f} (95% CI)")


if __name__ == "__main__":
    main()
