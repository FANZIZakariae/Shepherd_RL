import streamlit as st
import time
import os
import re
from PIL import Image
import numpy as np
import torch
import pygame
import sys
import pandas as pd

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
    num_sheep = st.slider("Total Number of Sheep", 1, 5, 1)
    
    pre_solved_sheep = st.slider("Pre-Solved Sheep (Spawn in Goal)", 0, num_sheep, 0)
    
    obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
    
    # --- NEW: JITTER ---
    sheep_jitter = st.slider("Sheep Jitter (Noise)", 0.0, 0.2, 0.0, 0.01)
    
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
        # Updated Levels
        selected_level = st.selectbox("Select Training Level", ["level_1", "level_2", "level_3", "level_4"])
        base_model_dir = os.path.join("models", selected_level, agent_type.lower())
        
        available_runs = []
        if os.path.exists(base_model_dir):
            for f in os.listdir(base_model_dir):
                full_p = os.path.join(base_model_dir, f)
                if os.path.isdir(full_p):
                     available_runs.append(f)
                elif f.endswith(".zip") or f.endswith(".pth"):
                     clean_name = f.replace("_model.zip", "").replace(".pth", "")
                     available_runs.append(clean_name)
            
            # Sort naturally
            available_runs.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x)

        if not available_runs:
            st.warning(f"No models found in `{base_model_dir}`")
            selected_run = "None"
        else:
            selected_run = st.selectbox("Select Trained Model", available_runs)

        if selected_run != "None":
            folder_path = os.path.join(base_model_dir, selected_run)
            
            if os.path.isdir(folder_path):
                # Look inside folder
                if agent_type == "DQN":
                     if os.path.exists(os.path.join(folder_path, "best_model.pth")):
                        checkpoint_path = os.path.join(folder_path, "best_model.pth")
                     elif os.path.exists(os.path.join(folder_path, "final_model.pth")):
                        checkpoint_path = os.path.join(folder_path, "final_model.pth")
                else: # PPO / TD3
                     if os.path.exists(os.path.join(folder_path, "best_model.zip")):
                        checkpoint_path = os.path.join(folder_path, "best_model.zip")
                     elif os.path.exists(os.path.join(folder_path, "final_model.zip")):
                        checkpoint_path = os.path.join(folder_path, "final_model.zip")
            else:
                # Legacy file support
                if agent_type == "DQN":
                    checkpoint_path = os.path.join(base_model_dir, f"{selected_run}.pth")
                else:
                    checkpoint_path = os.path.join(base_model_dir, f"{selected_run}_model.zip")
    
    st.markdown("---")
    
    disable_button = (agent_type != "ruleBase" and (not checkpoint_path or not os.path.exists(checkpoint_path)))
    run_demo = st.button("▶ Run Episode", type="primary", use_container_width=True, disabled=disable_button)

# -------------------------
# Main Page Layout
# -------------------------
st.title("🎮 Agent Demonstration")

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
    # --- 1. Init Environment ---
    env = ShepherdEnv(
        n_sheep=num_sheep, 
        max_steps=max_steps,
        obstacle_radius=obstacle_radius, 
        goal_radius=goal_radius,
        n_sheep_in_goal=pre_solved_sheep,
        sheep_jitter=sheep_jitter # <--- PASSED HERE
    )

    agent = None
    with st.status(f"Loading {agent_type} agent...", expanded=True) as status:
        try:
            if agent_type == "ruleBase":
                agent = RuleBasedShepherd()
                status.write("Rule-Based Logic Loaded.")
            elif agent_type in ["PPO", "TD3"]:
                if checkpoint_path and os.path.exists(checkpoint_path):
                    status.write(f"Loading `{os.path.basename(checkpoint_path)}`...")
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

    # --- 2. Live Dashboard Layout ---
    col_video, col_graph = st.columns([1.5, 2])
    
    with col_video:
        st.markdown("### 🎥 Agent View")
        frame_placeholder = st.empty()
        reward_metric = st.empty()

    with col_graph:
        st.markdown("### 📈 Cumulative Reward History")
        chart_placeholder = st.empty()

    # --- 3. Simulation Loop ---
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    # Data Tracking
    cumulative_breakdown = {
        "Progress (Push)": 0.0,
        "Target Proximity": 0.0,
        "Goal Entry Bonus": 0.0,
        "Movement Cost": 0.0,
        "Step Cost": 0.0,
        "Win Bonus": 0.0,
        "Fail Penalty": 0.0
    }
    
    history_df = pd.DataFrame(columns=list(cumulative_breakdown.keys()))
    
    while not done:
        # A. Action
        if agent_type == "ruleBase":
            action = agent.act(obs)
        elif agent_type == "DQN":
            state = torch.from_numpy(render_env_to_rgb(env)).float().permute(2, 0, 1)
            action_idx = agent.select_action(state, training=False)
            action = [ANGLES[action_idx]]
        else:
            action, _ = agent.predict(obs, deterministic=True)

        # B. Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # C. Update Data
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                if k in cumulative_breakdown:
                    cumulative_breakdown[k] += v
        
        new_row = pd.DataFrame([cumulative_breakdown])
        history_df = pd.concat([history_df, new_row], ignore_index=True)

        # D. Render Video
        frame = get_pygame_frame(env, scale=render_scale)
        if frame is not None:
            frame_placeholder.image(Image.fromarray(frame), use_container_width=True)

        # E. Render Graph & Metrics
        reward_metric.metric("Total Reward", f"{total_reward:.2f}", f"Step: {env.steps}")
        chart_placeholder.line_chart(history_df)

        time.sleep(0.05)

    if cumulative_breakdown.get("Win Bonus", 0) > 0:
        st.success(f"🏆 Episode Won! Final Reward: **{total_reward:.2f}**")
    else:
        st.error(f"💀 Episode Failed. Final Reward: **{total_reward:.2f}**")