import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
import re
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from envs.shepherd_env import ShepherdEnv

# Import your custom agents
try:
    from agents.rule_based_agent import RuleBasedShepherd, LazyShepherd, TipsyShepherd
    from agents.CNN_QN_agent import ImageDQNAgent, N_ACTIONS, render_env_to_rgb, ANGLES, transform
except ImportError:
    st.error("Could not import agents. Ensure `agents/rule_based_agent.py` contains RuleBasedShepherd, LazyShepherd, and TipsyShepherd.")

from stable_baselines3 import PPO, TD3

st.set_page_config(page_title="Model Benchmark", page_icon="🧪", layout="wide")
st.title("🧪 Model Performance Benchmark")
st.markdown("Evaluate and compare multiple agents side-by-side on specific levels.")

# -------------------------
# Sidebar: Configuration
# -------------------------
with st.sidebar:
    st.header("⚙️ Test Settings")
    
    # 1. Level Selection
    st.subheader("1. Scenario Level")
    selected_level = st.selectbox("Select Level", ["level_1", "level_2", "level_3", "level_4"])
    
    # 2. Env Params
    with st.expander("🌍 Environment Parameters", expanded=False):
        num_sheep = st.slider("Number of sheep", 1, 5, 1)
        pre_solved_sheep = st.slider("Pre-Solved Sheep (Spawn in Goal)", 0, num_sheep, 0)
        obstacle_radius = st.slider("Obstacle radius", 0.0, 2.0, 0.0, 0.1)
        
        # --- NEW: JITTER ---
        sheep_jitter = st.slider("Sheep Jitter (Noise)", 0.0, 0.2, 0.0, 0.01)

        goal_radius = st.slider("Goal radius", 0.1, 2.0, 0.7, 0.1)
        max_steps = st.number_input("Max steps", value=500)

    st.markdown("---")
    
    # 3. Model Selection
    st.subheader("2. Select Agents")
    
    # Baseline Agents Selection
    baseline_options = ["Rule-Based", "Lazy Shepherd", "Tipsy Shepherd"]
    selected_baselines = st.multiselect(
        "Select Baseline Agents", 
        baseline_options, 
        default=["Rule-Based"]
    )
    
    # Trained Models Selection
    available_models = [] # Format: ("Type", "Name", "Path")
    
    # Helper for natural sorting (so Run_2 comes before Run_10)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    for algo in ["ppo", "td3", "dqn"]:
        search_dir = os.path.join("models", selected_level, algo)
        
        if os.path.exists(search_dir):
            # List all entries and sort them naturally
            entries = sorted(os.listdir(search_dir), key=natural_sort_key)
            
            for entry in entries:
                full_path = os.path.join(search_dir, entry)
                
                # --- LOGIC UPDATE: CHECK FOLDERS FOR BEST MODEL ---
                if os.path.isdir(full_path):
                    # It's a run folder (e.g. "PPO_1")
                    # Check for best_model then final_model
                    model_found_path = None
                    label_suffix = ""

                    if algo == "dqn":
                        candidates = [("best_model.pth", "Best"), ("final_model.pth", "Final")]
                    else:
                        candidates = [("best_model.zip", "Best"), ("final_model.zip", "Final")]
                    
                    for filename, label in candidates:
                        check_path = os.path.join(full_path, filename)
                        if os.path.exists(check_path):
                            model_found_path = check_path
                            label_suffix = f" ({label})"
                            break # Prioritize Best, stop if found
                    
                    if model_found_path:
                        display_name = f"{entry}{label_suffix}"
                        available_models.append((algo.upper(), display_name, model_found_path))

                # --- LOGIC UPDATE: SUPPORT LEGACY FILES ---
                elif entry.endswith(".zip") or entry.endswith(".pth"):
                    clean_name = entry.replace("_model.zip", "").replace(".zip", "").replace(".pth", "")
                    available_models.append((algo.upper(), clean_name, full_path))
    
    if not available_models:
        st.info(f"No trained models found in `models/{selected_level}`")

    model_options = [f"{m[0]}: {m[1]}" for m in available_models]
    selected_model_strings = st.multiselect("Select Trained Models", model_options)
    
    # Map back selection string to path
    selected_trained_config = []
    for sel in selected_model_strings:
        match = next((m for m in available_models if f"{m[0]}: {m[1]}" == sel), None)
        if match:
            selected_trained_config.append(match)

    st.markdown("---")
    
    # 4. Evaluation Params
    st.subheader("3. Evaluation Scope")
    n_episodes = st.slider("Episodes per Model", min_value=1, max_value=500, value=50)
    
    start_eval = st.button("🚀 Start Benchmark", type="primary", use_container_width=True)


# -------------------------
# Helper Functions
# -------------------------
def load_trained_agent(algo, path, env):
    """Loads a trained RL agent."""
    try:
        if algo == "PPO":
            return PPO.load(path, env=env, device="cpu")
        elif algo == "TD3":
            return TD3.load(path, env=env, device="cpu")
        elif algo == "DQN":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ImageDQNAgent(n_actions=N_ACTIONS, lr=1e-4, gamma=0.99, device=device)
            # Handle loading specific paths safely
            agent.q_net.load_state_dict(torch.load(path, map_location=device))
            agent.q_net.eval()
            return agent
    except Exception as e:
        st.error(f"Failed to load {algo} at {path}: {e}")
        return None

def evaluate_agent(agent, agent_type, env, n_episodes):
    """Runs the evaluation loop for a single agent."""
    rewards = []
    steps = []
    successes = 0
    timeouts = 0
    
    progress_bar = st.progress(0)
    
    for i in range(n_episodes):
        if hasattr(agent, 'reset'):
            agent.reset()
            
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        while not done:
            if agent_type == "Rule-Based":
                action = agent.act(obs)
            elif agent_type == "DQN":
                state_np = render_env_to_rgb(env)
                state = torch.from_numpy(state_np).float().permute(2, 0, 1)
                action_idx = agent.select_action(state, training=False)
                action = [ANGLES[action_idx]]
            else:
                action, _ = agent.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if terminated:
                successes += 1
                done = True
            elif truncated:
                timeouts += 1
                done = True
        
        rewards.append(episode_reward)
        steps.append(episode_steps)
        progress_bar.progress((i + 1) / n_episodes)
        
    progress_bar.empty()
    
    return {
        "Avg Reward": np.mean(rewards),
        "Avg Steps (Time)": np.mean(steps),
        "Success Rate (%)": (successes / n_episodes) * 100,
        "Timeout Rate (%)": (timeouts / n_episodes) * 100
    }

# -------------------------
# Main Logic
# -------------------------
if start_eval:
    # 1. Validation
    if not selected_baselines and not selected_trained_config:
        st.error("Please select at least one agent (Baseline or Trained).")
        st.stop()

    # 2. Init Environment
    env = ShepherdEnv(
        n_sheep=num_sheep, 
        max_steps=max_steps, 
        obstacle_radius=obstacle_radius, 
        goal_radius=goal_radius,
        n_sheep_in_goal=pre_solved_sheep,
        sheep_jitter=sheep_jitter # <--- PASSED HERE
    )

    results_data = []

    # 3. Evaluation Loop
    total_agents = len(selected_baselines) + len(selected_trained_config)
    current_agent_idx = 1
    
    st.divider()
    
    # --- A. Baseline Agents ---
    for baseline_name in selected_baselines:
        with st.status(f"Testing Agent {current_agent_idx}/{total_agents}: **{baseline_name}**...", expanded=False) as status:
            
            if baseline_name == "Rule-Based":
                agent_instance = RuleBasedShepherd()
            elif baseline_name == "Lazy Shepherd":
                agent_instance = LazyShepherd()
            elif baseline_name == "Tipsy Shepherd":
                agent_instance = TipsyShepherd()
            
            metrics = evaluate_agent(agent_instance, "Rule-Based", env, n_episodes)
            
            metrics["Agent Name"] = baseline_name
            metrics["Type"] = "Baseline"
            results_data.append(metrics)
            status.update(label=f"✅ {baseline_name} Complete", state="complete")
        
        current_agent_idx += 1

    # --- B. Trained Models ---
    for algo, name, path in selected_trained_config:
        display_name = f"{algo}: {name}"
        
        with st.status(f"Testing Agent {current_agent_idx}/{total_agents}: **{display_name}**...", expanded=False) as status:
            agent = load_trained_agent(algo, path, env)
            
            if agent:
                metrics = evaluate_agent(agent, algo, env, n_episodes)
                
                metrics["Agent Name"] = name
                metrics["Type"] = algo
                results_data.append(metrics)
                status.update(label=f"✅ {display_name} Complete", state="complete")
            else:
                status.update(label=f"❌ {display_name} Failed to Load", state="error")
                
        current_agent_idx += 1

    # 4. Display Results
    st.divider()
    st.subheader("📊 Comparative Results")

    if results_data:
        df = pd.DataFrame(results_data)
        
        cols = ["Agent Name", "Type", "Success Rate (%)", "Timeout Rate (%)", "Avg Reward", "Avg Steps (Time)"]
        df = df[cols]
        
        st.dataframe(
            df.style.format({
                "Avg Reward": "{:.2f}",
                "Avg Steps (Time)": "{:.1f}",
                "Success Rate (%)": "{:.1f}%",
                "Timeout Rate (%)": "{:.1f}%"
            }).background_gradient(cmap="Greens", subset=["Success Rate (%)", "Avg Reward"])
              .background_gradient(cmap="Reds", subset=["Timeout Rate (%)", "Avg Steps (Time)"]),
            use_container_width=True
        )
        
        st.caption("Success Rate Comparison")
        st.bar_chart(df.set_index("Agent Name")["Success Rate (%)"])
    else:
        st.warning("No results generated.")