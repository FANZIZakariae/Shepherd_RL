import streamlit as st
import os
import sys
import subprocess
import time
import socket
from datetime import datetime

# Add the project root (rl_shepherd-main) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from envs.shepherd_env import ShepherdEnv

# -------------------------------------------------------------------
# IMPORT TRAINING FUNCTIONS
# -------------------------------------------------------------------
try:
    from agents.rl_agent import train_rl_agent_ppo_mlp, train_rl_agent_td3_mlp
except ImportError:
    st.error("⚠️ Could not import training functions. Please ensure `agents/rl_agents.py` exists.")
    def train_rl_agent_ppo_mlp(*args, **kwargs): pass
    def train_rl_agent_td3_mlp(*args, **kwargs): pass

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
        # Covers multiple sheep scenarios
        return "level_3"

# -------------------------------------------------------------------
# 1. Automatic TensorBoard Manager
# -------------------------------------------------------------------
@st.cache_resource
def launch_tensorboard(log_dir):
    # Check if port 6006 is free
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 6006))
    sock.close()
    
    if result == 0:
        return "TensorBoard is already running on port 6006."
    
    # Start TensorBoard pointing to the ROOT 'logs' directory
    # This allows TB to see logs/level_1/ppo, logs/level_2/ppo, etc.
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3) 
    return "TensorBoard started automatically on port 6006."

if not os.path.exists("logs"):
    os.makedirs("logs")

tb_status = launch_tensorboard("logs")

# -------------------------
# 2. Sidebar Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("1. Agent Architecture")
    agent_type = st.selectbox("Choose Algorithm", ["PPO", "TD3"], index=0)

    with st.expander("🌍 Environment Settings", expanded=True):
        num_sheep = st.slider("Number of sheep", 1, 5, 1)
        obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
        goal_radius = st.slider("Goal radius", 0.1, 2.0, 0.7, 0.1)
        max_steps = st.number_input("Max steps per episode", value=500)
        
        # --- DYNAMIC LEVEL DISPLAY ---
        current_level = get_level_name(num_sheep, obstacle_radius)
        if current_level == "level_1":
            st.success(f"📍 **Current Mode: {current_level.upper().replace('_', ' ')}**\n\n(Single Sheep, No Obstacles)")
        elif current_level == "level_2":
            st.warning(f"📍 **Current Mode: {current_level.upper().replace('_', ' ')}**\n\n(Single Sheep, With Obstacles)")
        else:
            st.error(f"📍 **Current Mode: {current_level.upper().replace('_', ' ')}**\n\n(Multi-Sheep / Hard)")

    with st.expander("🧠 Hyperparameters", expanded=True):
        timesteps = st.number_input("Total Timesteps", value=2_000_000, step=100_000)
        curriculum = st.checkbox("Enable Curriculum Learning", value=False)

    st.markdown("---")
    
    st.subheader("📂 Load Checkpoint")
    # Select Level
    load_level = st.selectbox("Select Level to Load From", ["level_1", "level_2", "level_3"])
    
    # Path: models/level_X/agent_type/
    models_folder = os.path.join("models", load_level, agent_type.lower())
    
    available_models = []
    if os.path.exists(models_folder):
        available_models = [f.replace("_model.zip", "") for f in os.listdir(models_folder) if f.endswith(".zip")]
        import re
        available_models.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else x)
    
    selected_model_name = st.selectbox("Select Run", ["None"] + available_models)
    
    checkpoint_path = None
    if selected_model_name != "None":
        checkpoint_path = os.path.join(models_folder, f"{selected_model_name}_model.zip")
        if os.path.exists(checkpoint_path):
            st.success(f"Ready to load: {selected_model_name}")

    st.write("")
    start_training = st.button(f"▶ Start {current_level.upper().replace('_', ' ')} Training", type="primary", use_container_width=True)


# -------------------------
# 3. Main Dashboard (TensorBoard)
# -------------------------
st.markdown("### 📊 Live Training Dashboard")
st.caption(f"Backend Status: {tb_status}")

# Filter regex example: level_1/ppo
tb_regex = f"{current_level}/{agent_type.lower()}"
tb_url = f"http://localhost:6006/#scalars&regex={tb_regex}&time=wall_relative"

st.markdown(
    f'<iframe src="{tb_url}" width="100%" height="700" frameborder="0"></iframe>',
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# 4. Training Logic
# -------------------------
if start_training:
    # 1. Environment Setup
    env = ShepherdEnv(n_sheep=num_sheep, max_steps=max_steps, obstacle_radius=obstacle_radius, goal_radius=goal_radius)
    eval_env = ShepherdEnv(n_sheep=num_sheep, max_steps=max_steps, obstacle_radius=obstacle_radius, goal_radius=goal_radius)

    # 2. Status UI
    col_status, col_spinner = st.columns([2, 1])
    with col_status:
        st.info(f"**Training Level:** `{current_level}` | **Agent:** `{agent_type}`")
        if checkpoint_path:
            st.write(f"🔄 Continuing from: `{os.path.basename(checkpoint_path)}`")
        else:
            st.write("🆕 Starting fresh session")

    # 3. Blocking Training Call
    with st.spinner(f"🚀 Training {agent_type} in {current_level}..."):
        try:
            model = None
            if agent_type == "PPO":
                model = train_rl_agent_ppo_mlp(
                    env=env, 
                    eval_env=eval_env, 
                    timesteps=timesteps, 
                    checkpoint_dir=checkpoint_path, 
                    criculam_learning=curriculum
                )
            elif agent_type == "TD3":
                model = train_rl_agent_td3_mlp(
                    env=env, 
                    eval_env=eval_env, 
                    timesteps=timesteps, 
                    checkpoint_dir=checkpoint_path, 
                    criculam_learning=curriculum
                )
            
            # -------------------------------------------------------
            # 4. SYNC MODEL NAME WITH LOG NAME (LEVEL AWARE)
            # -------------------------------------------------------
            if model:
                # 1. Find the log in the standard location (./logs/ppo)
                default_log_root = os.path.join("logs", agent_type.lower())
                
                run_name = f"{agent_type}_{int(time.time())}" # Fallback
                if os.path.exists(default_log_root):
                    try:
                        subdirs = [os.path.join(default_log_root, d) for d in os.listdir(default_log_root) if os.path.isdir(os.path.join(default_log_root, d))]
                        if subdirs:
                            latest_subdir = max(subdirs, key=os.path.getmtime)
                            run_name = os.path.basename(latest_subdir)
                    except:
                        pass

                # 2. Construct the LEVEL-specific save path
                # Path: models/level_1/ppo/PPO_1_model.zip
                save_dir = os.path.join("models", current_level, agent_type.lower())
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"{run_name}_model")
                
                model.save(save_path)
                
                st.success(f"✅ Training finished! Model saved to: `{save_path}.zip`")
                st.caption(f"Linked Log ID: `{run_name}`")
            else:
                st.error("Training function returned None.")

        except Exception as e:
            st.error(f"An error occurred: {e}")