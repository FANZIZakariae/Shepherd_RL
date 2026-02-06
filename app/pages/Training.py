import streamlit as st
import os
import sys
import subprocess
import time
import socket
import torch
from datetime import datetime

# Add the project root (rl_shepherd-main) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from envs.shepherd_env import ShepherdEnv

# -------------------------------------------------------------------
# IMPORT TRAINING FUNCTIONS
# -------------------------------------------------------------------
try:
    # Import PPO (and TD3 if available)
    from agents.ppo_agent import train_rl_agent_ppo_mlp
    # Import DQN
    from agents.CNN_QN_agent import train_image_dqn, ImageDQNAgent, N_ACTIONS
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}. Please ensure `agents/ppo_agent.py` and `agents/CNN_QN.py` exist.")
    # Fallbacks to prevent crash
    def train_rl_agent_ppo_mlp(*args, **kwargs): pass
    def train_image_dqn(*args, **kwargs): pass
    N_ACTIONS = 64

st.set_page_config(page_title="Training Deck", page_icon="🏋️", layout="wide")
st.title("🏋️ Shepherd RL Training Deck")

# -------------------------------------------------------------------
# HELPER: Determine Level
# -------------------------------------------------------------------
def get_level_name(n_sheep, obs_radius):
    if n_sheep == 1 and obs_radius == 0.0:
        return "level_1"
    elif n_sheep == 1 and obs_radius > 0.0:
        return "level_2"
    else:
        return "level_3"

# -------------------------------------------------------------------
# Automatic TensorBoard Manager
# -------------------------------------------------------------------
@st.cache_resource
def launch_tensorboard(log_dir):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 6006))
    sock.close()
    
    if result == 0:
        return "TensorBoard is already running on port 6006."
    
    # Launch TB pointing to the root 'logs' folder so it sees all levels/agents
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3) 
    return "TensorBoard started automatically on port 6006."

if not os.path.exists("logs"):
    os.makedirs("logs")

tb_status = launch_tensorboard("logs")

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Agent Architecture
    st.subheader("1. Agent Architecture")
    agent_type = st.selectbox("Choose Algorithm", ["PPO", "DQN"], index=0)

    # 2. Environment
    with st.expander("🌍 Environment Settings", expanded=True):
        num_sheep = st.slider("Number of sheep", 1, 5, 1)
        obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
        goal_radius = st.slider("Goal radius", 0.1, 2.0, 0.7, 0.1)
        max_steps = st.number_input("Max steps per episode", value=500)
        
        current_level = get_level_name(num_sheep, obstacle_radius)
        if current_level == "level_1":
            st.success(f"📍 **Mode: {current_level}** (Easy)")
        elif current_level == "level_2":
            st.warning(f"📍 **Mode: {current_level}** (Medium)")
        else:
            st.error(f"📍 **Mode: {current_level}** (Hard)")

    # 3. Hyperparameters
    with st.expander("🧠 Hyperparameters", expanded=True):
        timesteps = st.number_input("Total Timesteps (approx)", value=100_000, step=10_000)
        curriculum = st.checkbox("Enable Curriculum", value=False)
        st.caption(f"For DQN: ≈ {int(timesteps/max_steps)} episodes")

    st.markdown("---")
    
    # 4. Load Checkpoint
    st.subheader("📂 Load Checkpoint")
    load_level = st.selectbox("Select Level to Load From", ["level_1", "level_2", "level_3"])
    
    # Path logic varies by agent (PPO=zip file, DQN=folder or .pth)
    models_folder = os.path.join("models", load_level, agent_type.lower())
    
    available_models = []
    if os.path.exists(models_folder):
        # List all files/folders that look like model checkpoints
        # PPO: .zip files
        # DQN: folders (if saving multiple .pth) or .pth files
        for f in os.listdir(models_folder):
            if f.endswith(".zip") or f.endswith(".pth") or os.path.isdir(os.path.join(models_folder, f)):
                available_models.append(f)
        available_models.sort()
    
    selected_model_name = st.selectbox("Select Run", ["None"] + available_models)
    
    checkpoint_path = None
    if selected_model_name != "None":
        full_path = os.path.join(models_folder, selected_model_name)
        # If DQN selected a folder, assume we want 'best_model.pth' inside
        if os.path.isdir(full_path) and agent_type == "DQN":
             potential_pth = os.path.join(full_path, "best_model.pth")
             if os.path.exists(potential_pth):
                 checkpoint_path = potential_pth
             else:
                 st.warning("Selected folder does not contain 'best_model.pth'")
        else:
            checkpoint_path = full_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            st.success(f"Ready: {os.path.basename(checkpoint_path)}")

    st.write("")
    start_training = st.button(f"▶ Start {current_level} {agent_type}", type="primary", use_container_width=True)


# -------------------------
# Main Dashboard
# -------------------------
st.markdown("### 📊 Live Training Dashboard")
st.caption(f"Backend: {tb_status}")

# TensorBoard Filtering
# We filter logs by "level_X/agent_type" to keep the view clean
tb_regex = f"{current_level}/{agent_type.lower()}"
tb_url = f"http://localhost:6006/#scalars&regex={tb_regex}&time=wall_relative"

st.markdown(
    f'<iframe src="{tb_url}" width="100%" height="700" frameborder="0"></iframe>',
    unsafe_allow_html=True
)
st.divider()

# -------------------------
# Training Logic
# -------------------------
if start_training:
    # 1. Init Envs
    env = ShepherdEnv(n_sheep=num_sheep, max_steps=max_steps, obstacle_radius=obstacle_radius, goal_radius=goal_radius)
    eval_env = ShepherdEnv(n_sheep=num_sheep, max_steps=max_steps, obstacle_radius=obstacle_radius, goal_radius=goal_radius)

    # 2. Setup Paths
    # Unique ID: DQN_1707050000
    run_id = f"{agent_type}_{int(time.time())}"
    
    # Log Path: logs/level_1/dqn/DQN_timestamp/
    log_dir = os.path.join("logs", current_level, agent_type.lower(), run_id)
    
    # Save Path: models/level_1/dqn/DQN_timestamp/
    # (We put DQN in a subfolder because it saves multiple .pth files)
    save_dir = os.path.join("models", current_level, agent_type.lower())
    if agent_type == "DQN":
        save_dir = os.path.join(save_dir, run_id)

    col_status, col_spinner = st.columns([2, 1])
    with col_status:
        st.info(f"**Training:** `{agent_type}` | **Level:** `{current_level}`")
        st.write(f"📂 Logs: `{log_dir}`")
        if checkpoint_path:
            st.write(f"🔄 Continuing from `{os.path.basename(checkpoint_path)}`")
        else:
            st.write("🆕 Starting fresh run")

    with st.spinner(f"🚀 Training {agent_type}..."):
        try:
            # --- PPO TRAINING ---
            if agent_type == "PPO":
                # PPO uses steps
                model = train_rl_agent_ppo_mlp(
                    env=env, 
                    eval_env=eval_env, 
                    timesteps=timesteps, 
                    checkpoint_dir=checkpoint_path, 
                    criculam_learning=curriculum
                )
                
                # Manual Saving for PPO (assuming function returns model but doesn't handle specific naming)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{run_id}_model")
                model.save(save_path)
                
                st.success(f"✅ Training Complete! Model saved to `{save_path}.zip`")

            # --- DQN TRAINING ---
            elif agent_type == "DQN":
                # DQN uses episodes, convert approx steps -> episodes
                approx_episodes = max(1, int(timesteps / max_steps))
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                agent = ImageDQNAgent(n_actions=N_ACTIONS, device=device)
                
                # Call the custom training loop
                # This function handles the loop, logging, and saving internally based on args
                train_image_dqn(
                    env=env, 
                    eval_env=eval_env, 
                    agent=agent, 
                    episodes=approx_episodes, 
                    log_dir=log_dir, 
                    save_dir=save_dir,
                    checkpoint_path=checkpoint_path
                )
                
                st.success(f"✅ Training Complete! Model saved to `{save_dir}/final_model.pth`")

        except Exception as e:
            st.error(f"Training Error: {e}")
            raise e