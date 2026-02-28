import streamlit as st
import os
import sys
import subprocess
import time
import socket
import torch
import re 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from envs.shepherd_env import ShepherdEnv

try:
    from agents.ppo_agent import train_rl_agent_ppo_mlp
    from agents.TD3_agent import train_rl_agent_td3_mlp
    from agents.CNN_QN_agent import train_image_dqn, ImageDQNAgent, N_ACTIONS
except ImportError as e:
    st.error(f"⚠️ Import Error: {e}")
    def train_rl_agent_ppo_mlp(*args, **kwargs): pass
    def train_rl_agent_td3_mlp(*args, **kwargs): pass
    def train_image_dqn(*args, **kwargs): pass
    N_ACTIONS = 64

st.set_page_config(page_title="Training Deck", page_icon="🏋️", layout="wide")
st.title("🏋️ Shepherd RL Training Deck")

# -------------------------------------------------------------------
# HELPER: Determine Level & Naming
# -------------------------------------------------------------------
def get_level_name(n_sheep, obs_radius, jitter, pre_solved=0):
    """
    Determines the level based on 'Active Sheep' (Total - PreSolved).
    This enables curriculum learning: A Level 4 setup (3 sheep) 
    with 2 pre-solved behaves like Level 1/2/3.
    """
    # Calculate how many sheep are actually outside the goal
    active_sheep = n_sheep - pre_solved

    # --- CASE A: MULTI-AGENT TASK ---
    if active_sheep > 1:
        return "level_4"
    
    # --- CASE B: SINGLE AGENT TASK (Curriculum) ---
    elif active_sheep == 1:
        # Level 1: 1 Active Sheep, No Obstacle, No Jitter
        if obs_radius == 0.0 and jitter == 0.0:
            return "level_1"
        
        # Level 2: 1 Active Sheep, Obstacle, No Jitter
        elif obs_radius > 0.0 and jitter == 0.0:
            return "level_2"
        
        # Level 3: 1 Active Sheep, Obstacle, Jitter
        elif obs_radius > 0.0 and jitter > 0.0:
            return "level_3"
        
        else:
            return "custom_level"

    # --- CASE C: ALL SOLVED OR INVALID ---
    else:
        # If active_sheep is 0 (all spawned in goal)
        return "custom_level"

def get_next_run_number(base_dir, prefix):
    if not os.path.exists(base_dir):
        return 1
    max_num = 0
    for name in os.listdir(base_dir):
        if name.startswith(prefix):
            parts = name.replace(prefix, "").replace("_", "")
            if parts.isdigit():
                num = int(parts)
                if num > max_num: max_num = num
            else:
                match = re.search(r'(\d+)', name)
                if match:
                    num = int(match.group(1))
                    if num > max_num: max_num = num
    return max_num + 1

@st.cache_resource
def launch_tensorboard(log_dir):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 6006))
    sock.close()
    if result == 0:
        return "TensorBoard is already running on port 6006."
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"]
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3) 
    return "TensorBoard started automatically on port 6006."

if not os.path.exists("logs"): os.makedirs("logs")
tb_status = launch_tensorboard("logs")

# -------------------------
# Sidebar Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("1. Agent Architecture")
    agent_type = st.selectbox("Choose Algorithm", ["PPO", "TD3", "DQN"], index=0)

    with st.expander("🌍 Environment Settings", expanded=True):
        num_sheep = st.slider("Total Number of Sheep", 1, 5, 1)
        pre_solved_sheep = st.slider("Pre-Solved Sheep (Spawn in Goal)", 0, num_sheep, 0)
        
        obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
        
        # --- NEW SLIDER: SHEEP JITTER ---
        sheep_jitter = st.slider("Sheep Jitter (Noise)", 0.0, 0.2, 0.0, 0.01)
        
        goal_radius = st.slider("Goal radius", 0.1, 2.0, 0.7, 0.1)
        max_steps = st.number_input("Max steps per episode", value=500)
        
        # --- UPDATED LEVEL DETECTION ---
        current_level = get_level_name(num_sheep, obstacle_radius, sheep_jitter, pre_solved_sheep)
        
        # UI Feedback for Levels
        active_count = num_sheep - pre_solved_sheep
        st.caption(f"Active Sheep: {active_count} (Total: {num_sheep})")

        if current_level == "level_1":
            st.success("📍 Level 1 (Basic - 1 Active)")
        elif current_level == "level_2":
            st.info("📍 Level 2 (Obstacles - 1 Active)")
        elif current_level == "level_3":
            st.warning("📍 Level 3 (Shaky - 1 Active)")
        elif current_level == "level_4":
            st.error(f"📍 Level 4 (Herd - {active_count} Active)")
        else:
            st.write(f"📍 Custom Mode: {current_level}")

    with st.expander("🧠 Hyperparameters", expanded=True):
        timesteps = st.number_input("Total Timesteps", value=100_000, step=10_000)
        curriculum = st.checkbox("Enable Curriculum", value=False)

    st.markdown("---")
    st.subheader("📂 Load Checkpoint")
    load_level = st.selectbox("Select Level to Load From", ["level_1", "level_2", "level_3", "level_4", "custom_level"])
    models_base_folder = os.path.join("models", load_level, agent_type.lower())
    
    available_models = []
    if os.path.exists(models_base_folder):
        for f in os.listdir(models_base_folder):
            full_p = os.path.join(models_base_folder, f)
            if os.path.isdir(full_p):
                available_models.append(f)
            elif f.endswith(".zip") or f.endswith(".pth"):
                available_models.append(f)
                
    available_models.sort()
    
    selected_run = st.selectbox("Select Run", ["None"] + available_models)
    
    checkpoint_path = None
    if selected_run != "None":
        full_path = os.path.join(models_base_folder, selected_run)
        if os.path.isdir(full_path):
            if agent_type in ["PPO", "TD3"]:
                if os.path.exists(os.path.join(full_path, "best_model.zip")):
                    checkpoint_path = os.path.join(full_path, "best_model.zip")
                elif os.path.exists(os.path.join(full_path, "final_model.zip")):
                    checkpoint_path = os.path.join(full_path, "final_model.zip")
            elif agent_type == "DQN":
                 if os.path.exists(os.path.join(full_path, "best_model.pth")):
                    checkpoint_path = os.path.join(full_path, "best_model.pth")
                 elif os.path.exists(os.path.join(full_path, "final_model.pth")):
                    checkpoint_path = os.path.join(full_path, "final_model.pth")
        else:
            checkpoint_path = full_path

    st.write("")
    start_training = st.button(f"▶ Start {current_level} {agent_type}", type="primary", use_container_width=True)

# -------------------------
# Main Dashboard
# -------------------------
st.markdown("### 📊 Live Training Dashboard")
st.caption(f"Backend: {tb_status}")
tb_regex = f"{current_level}/{agent_type.lower()}"
tb_url = f"http://localhost:6006/#scalars&regex={tb_regex}&time=wall_relative"
st.markdown(f'<iframe src="{tb_url}" width="100%" height="700" frameborder="0"></iframe>', unsafe_allow_html=True)
st.divider()

# -------------------------
# Training Logic
# -------------------------
if start_training:
    # --- PASS THE NEW PARAMETERS TO ENV ---
    env = ShepherdEnv(
        n_sheep=num_sheep, 
        max_steps=max_steps, 
        obstacle_radius=obstacle_radius, 
        goal_radius=goal_radius,
        n_sheep_in_goal=pre_solved_sheep,
        sheep_jitter=sheep_jitter # <--- PASSED HERE
    )
    eval_env = ShepherdEnv(
        n_sheep=num_sheep, 
        max_steps=max_steps, 
        obstacle_radius=obstacle_radius, 
        goal_radius=goal_radius,
        n_sheep_in_goal=pre_solved_sheep,
        sheep_jitter=sheep_jitter
    )

    log_root = os.path.join("logs", current_level, agent_type.lower())
    model_root = os.path.join("models", current_level, agent_type.lower())
    
    next_num = get_next_run_number(log_root, agent_type)
    run_name = f"{agent_type}_{next_num}"
    model_save_path = os.path.join(model_root, run_name)

    col_status, col_spinner = st.columns([2, 1])
    with col_status:
        st.info(f"**Run ID:** `{run_name}` | **Level:** {current_level}")
        st.write(f"📂 Logs: `{log_root}/{run_name}`")
        st.write(f"💾 Models: `{model_save_path}`")

    with st.spinner(f"🚀 Training {agent_type}..."):
        try:
            if agent_type == "PPO":
                model = train_rl_agent_ppo_mlp(
                    env=env, 
                    eval_env=eval_env, 
                    timesteps=timesteps, 
                    checkpoint_dir=checkpoint_path, 
                    criculam_learning=curriculum,
                    log_dir=log_root,
                    model_save_path=model_save_path,
                    tb_log_name=agent_type
                )
                final_path = os.path.join(model_save_path, "final_model")
                model.save(final_path)
                st.success(f"✅ Training Done! Saved to `{final_path}.zip`")

            elif agent_type == "TD3":
                model = train_rl_agent_td3_mlp(
                    env=env,
                    eval_env=eval_env,
                    timesteps=timesteps,
                    checkpoint_dir=checkpoint_path,
                    criculam_learning=curriculum,
                    log_dir=log_root,
                    model_save_path=model_save_path,
                    tb_log_name=agent_type
                )
                final_path = os.path.join(model_save_path, "final_model")
                model.save(final_path)
                st.success(f"✅ Training Done! Saved to `{final_path}.zip`")

            elif agent_type == "DQN":
                approx_episodes = max(1, int(timesteps / max_steps))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                agent = ImageDQNAgent(n_actions=N_ACTIONS, device=device)
                full_log_path = os.path.join(log_root, run_name)
                
                train_image_dqn(
                    env=env, 
                    eval_env=eval_env, 
                    agent=agent, 
                    episodes=approx_episodes, 
                    log_dir=full_log_path,
                    save_dir=model_save_path,
                    checkpoint_path=checkpoint_path
                )
                st.success(f"✅ Training Done! Saved to `{model_save_path}/final_model.pth`")

        except Exception as e:
            st.error(f"Training Error: {e}")
            raise e