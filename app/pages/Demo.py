import streamlit as st
import time
import os
import re
from PIL import Image
import numpy as np
import torch
import pygame
import sys

# Add path to sys to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from envs.shepherd_env import ShepherdEnv
try:
    from agents.rule_based_agent import RuleBasedShepherd
    from agents.CNN_QN_agent import ImageDQNAgent, N_ACTIONS, render_env_to_rgb, ANGLES, transform
except ImportError:
    pass
    
from stable_baselines3 import PPO, TD3

st.set_page_config(page_title="Agent Demo", page_icon="🎮", layout="wide")

# -------------------------
# Sidebar Settings
# -------------------------
with st.sidebar:
    st.header("🎮 Demo Controls")
    
    st.subheader("1. Environment")
    num_sheep = st.slider("Number of sheep", 1, 5, 1)
    obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
    goal_radius = st.slider("Goal radius", 0.1, 2.0, 0.7, 0.1)
    max_steps = st.number_input("Max steps", value=500)

    st.subheader("2. Visual Settings")
    render_scale = st.slider("View Scale", 0.1, 2.0, 0.65, 0.05)

    st.markdown("---")
    st.subheader("3. Agent Selection")
    
    # A. Choose Type
    agent_type = st.selectbox("Choose Agent Architecture", ["ruleBase", "PPO", "TD3", "DQN"])
    checkpoint_path = None
    
    # B. Dynamic Model Selector with LEVELS
    if agent_type != "ruleBase":
        # New Step: Select Level
        selected_level = st.selectbox("Select Training Level", ["level_1", "level_2", "level_3"])
        
        # Path: models/level_X/agent_type/
        base_model_dir = os.path.join("models", selected_level, agent_type.lower())
        
        available_runs = []
        if os.path.exists(base_model_dir):
            files = [f for f in os.listdir(base_model_dir) if f.endswith(".zip")]
            for f in files:
                clean_name = f.replace("_model.zip", "")
                available_runs.append(clean_name)
            # Sort naturally
            available_runs.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x)

        if not available_runs:
            st.warning(f"No models found in `{base_model_dir}`")
            selected_run = "None"
        else:
            selected_run = st.selectbox("Select Trained Model", available_runs)

        if selected_run != "None":
            checkpoint_path = os.path.join(base_model_dir, f"{selected_run}_model.zip")
    
    st.markdown("---")
    
    disable_button = (agent_type != "ruleBase" and (not checkpoint_path or not os.path.exists(checkpoint_path)))
    run_demo = st.button("▶ Run Episode", type="primary", use_container_width=True, disabled=disable_button)

# -------------------------
# Main Page Layout
# -------------------------
st.title("🎮 Agent Demonstration")
st.markdown("Visualize the behavior of your trained agents in real-time.")

def get_pygame_frame(env, scale=0.65):
    if not pygame.get_init():
        pygame.init()
    env.render()
    screen = pygame.display.get_surface()
    if screen is None:
        return None
    frame = np.transpose(pygame.surfarray.array3d(screen), (1,0,2))
    h, w, _ = frame.shape
    new_size = (int(w*scale), int(h*scale))
    return np.array(Image.fromarray(frame).resize(new_size, Image.Resampling.BILINEAR))

if run_demo:
    env = ShepherdEnv(n_sheep=num_sheep, max_steps=max_steps,
                      obstacle_radius=obstacle_radius, goal_radius=goal_radius)

    agent = None
    with st.status(f"Loading {agent_type} agent...", expanded=True) as status:
        try:
            if agent_type == "ruleBase":
                agent = RuleBasedShepherd()
                status.write("Rule-Based Logic Loaded.")
            elif agent_type in ["PPO", "TD3"]:
                if checkpoint_path and os.path.exists(checkpoint_path):
                    status.write(f"Loading `{os.path.basename(checkpoint_path)}` from `{selected_level}`...")
                    if agent_type == "PPO":
                        agent = PPO.load(checkpoint_path, env=env, device="cpu")
                    else:
                        agent = TD3.load(checkpoint_path, env=env, device="cpu")
            elif agent_type == "DQN":
                if checkpoint_path and os.path.exists(checkpoint_path):
                    status.write(f"Loading `{os.path.basename(checkpoint_path)}`...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    agent = ImageDQNAgent(n_actions=N_ACTIONS, lr=1e-4, gamma=0.99, device=device)
                    agent.q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
                    agent.q_net.eval()
            
            status.update(label="Agent Ready!", state="complete")
        except Exception as e:
            status.update(label="Loading Failed", state="error")
            st.error(f"Error: {e}")
            st.stop()

    col_video, col_stats = st.columns([3, 1])
    with col_stats:
        st.markdown("### Live Stats")
        reward_metric = st.empty()
        step_metric = st.empty()
    
    with col_video:
        frame_placeholder = st.empty()

    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if agent_type == "ruleBase":
            action = agent.act(obs)
        elif agent_type == "DQN":
            state = transform(render_env_to_rgb(env))
            action_idx = agent.select_action(state)
            action = [ANGLES[action_idx]]
        else:
            action, _ = agent.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        frame = get_pygame_frame(env, scale=render_scale)
        if frame is not None:
            frame_placeholder.image(Image.fromarray(frame), caption="Agent View", use_container_width=False)

        reward_metric.metric("Total Reward", f"{total_reward:.2f}")
        step_metric.metric("Current Step", f"{env.steps}")
        time.sleep(0.05)

    st.success(f"Episode Finished! Final Reward: **{total_reward:.2f}**")