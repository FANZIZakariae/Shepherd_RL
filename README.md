# 🐑 Shepherd RL: Autonomous Sheep Herding with Reinforcement Learning

This project implements Reinforcement Learning (RL) agents to solve the "Shepherd Problem," where an autonomous agent (the dog) must herd sheep into a specific goal area while avoiding obstacles. 

It features a complete **Streamlit Dashboard** for training, visualizing, and benchmarking different RL agents (PPO, TD3, DQN) against a classic Rule-Based baseline.

---

## 📂 Project Structure

The project is organized into three difficulty levels to facilitate curriculum learning:
* **Level 1:** Single Sheep, No Obstacles.
* **Level 2:** Single Sheep, Static Obstacles.
* **Level 3:** Multiple Sheep, Static Obstacles.

The directory structure is critical for the app to function correctly:

```text
rl_shepherd-main/
├── agents/             # Agent implementations (PPO, TD3, DQN, RuleBased)
├── app/                # Streamlit Dashboard Code
│   ├── streamlit_app.py      # Main Hub Page
│   └── pages/
│       ├── 1_Training.py     # Training Interface (with embedded TensorBoard)
│       ├── 2_Demo.py         # Visual Demonstration (Pygame rendering)
│       └── 3_Test.py         # Benchmarking & Stress Testing
├── envs/               # Custom Shepherd Gymnasium Environment
├── logs/               # TensorBoard Logs (Auto-sorted by Level/Agent)
├── models/             # Saved Agent Checkpoints (Auto-sorted by Level/Agent)
└── requirements.txt    # Python dependencies
```
🚀 Installation
Clone the repository:

```Bash
git clone [https://github.com/your-username/rl_shepherd.git](https://github.com/your-username/rl_shepherd.git)
cd rl_shepherd-main
```
Create a virtual environment (Recommended):

```Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
Install dependencies:

```Bash
pip install -r requirements.txt
```
🎮 Usage
Launch the main dashboard hub using Streamlit:

```Bash
streamlit run app/streamlit_app.py
```
### 1. 🏋️ Training Deck
* **Configure:** Select your algorithm (PPO, TD3) and difficulty parameters. The app automatically detects if you are in **Level 1**, **2**, or **3** based on your settings.
* **Monitor:** TensorBoard launches automatically within the app to track reward curves in real-time.
* **Save:** Models are automatically saved to `models/level_X/agent_type/` with names synchronized to their log IDs.

### 2. 🎮 Demo Mode
* **Visualize:** Watch your trained agents in action via a Pygame-rendered video stream.
* **Control:** Use the **"View Scale"** slider to resize the simulation window to fit your screen.
* **Compare:** Switch instantly between your trained RL models and the Rule-Based heuristic agent.

### 3. 🧪 Benchmarking (Test Page)
* **Evaluate:** Run headless simulations (no rendering) to gather statistical data.
* **Metrics:** Compare **Success Rate**, **Timeout Rate**, and **Average Reward** across hundreds of episodes.
* **Side-by-Side:** Select multiple trained models to see how they stack up against the Rule-Based baseline in a comparative table.

---

## 🛠️ Agents Implemented
* **Rule-Based:** A heuristic algorithm using geometric forces (repulsion/attraction) to guide sheep.
* **PPO (Proximal Policy Optimization):** On-policy gradient method, stable and efficient.
* **TD3 (Twin Delayed DDPG):** Off-policy method, excellent for continuous control tasks.
* **DQN (Deep Q-Network):** For discrete action spaces or pixel-based observations.

## 📝 Requirements
* Python 3.8+
* Streamlit
* Stable Baselines3
* Pygame
* Gymnasium / OpenAI Gym
* TensorBoard
* Torch
* Pandas (for benchmarking tables)

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
