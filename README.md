# ShepherdRL – Progressive Robotic Shepherding with Rule-Based and RL Agents

ShepherdRL is a **custom reinforcement learning environment** for robotic shepherding inspired by biologically grounded flocking dynamics (e.g., the Strömbom model).  
The environment is structured into **four progressive problem levels**, enabling curriculum learning from simple deterministic control to multi-agent coordination.

The project follows a **Gym-style API**, includes **real-time visualization using Pygame**, and supports both **rule-based agents** and **reinforcement learning agents** (e.g., PPO via Stable-Baselines3).

---

## Installation: Create a Python environment (recommended)

```bash
conda create -n shepherding python=3.12
conda activate shepherding
pip install -r requirements.txt
```

## Project Structure

```bash
shepherd_rl/
│
├─ envs/
│  └─ shepherd_env.py        # Gym environment
│
├─ agents/
│  ├─ rule_based_agent.py   # Heuristic agent
│  ├─ rl_agent.py           # RL utilities
│  └─ CNN_QN.py           # DQN agent with pytorch
│
├─ test.py                  # Run simulation (rule-based or RL)
├─ train.py                 # Train RL agents (Levels 3 & 4)
├─ models/                  # Saved RL models (.zip)
└─ README.md
```

---

## Problem Levels

### Level 1 – Basic Shepherding (Sleepy Sheep)

**Purpose:**  
Learn fundamental shepherd–sheep interaction in a fully deterministic setting.

**Configuration:**
- Single shepherd agent
- No obstacles
- Sheep are *sleepy* (static unless influenced)
- Deterministic dynamics

**Sheep Behavior:**
- Sheep do not move unless the shepherd is close
- When close, sheep move directly away from the shepherd

**Training Variants:**
- Number of sheep: **1, 2, or 3**

---

### Level 2 – Active Sheep (Random Motion)

**Purpose:**  
Introduce stochasticity and robustness requirements.

**Configuration:**
- Single shepherd agent
- No obstacles
- Variable number of sheep

**Sheep Behavior:**
- If the shepherd is far: sheep move randomly
- If the shepherd is close: sheep move away from the shepherd

---

### Level 3 – Obstacle-Constrained Shepherding

**Purpose:**  
Introduce spatial constraints and navigation challenges.

**Configuration:**
- Single shepherd agent
- One circular obstacle in the environment

**Obstacle Rules:**
- Sheep cannot enter the obstacle area
- Shepherd cannot pass through the obstacle

**Sheep Behavior:**
- Same as Level 2

---

### Level 4 – Alternating Multi-Agent Shepherding

**Purpose:**  
Learn coordination under **turn-based multi-agent control**.

**Configuration:**
- Two trained shepherd agents
- Shared environment and shared goal
- No simultaneous actions

**Control Mechanism:**
- Only one shepherd acts at each timestep
- Agents alternate actions deterministically

---

## Sheep Behavior Summary

| Condition | Sheep Action |
|--------|-------------|
| Shepherd close | Move away from shepherd |
| Shepherd far (Level 1) | Remain static |
| Shepherd far (Levels 2–4) | Random movement |
| Inside goal | Locked, cannot exit |
| Inside obstacle | Not allowed |

---

## Rule-Based Shepherd Agents

Three **rule-based shepherd agents** are included for benchmarking, debugging, and comparison against RL policies.

---

### 1. Standard Rule-Based Shepherd

**Behavior:**
- Selects the sheep furthest from the goal
- Computes a driving point behind the sheep along the goal → sheep line
- Moves toward the driving point to push the sheep toward the goal

**Characteristics:**
- Deterministic
- Goal-driven

---

### 2. Lazy Shepherd


**Behavior:**
- Always outputs a **constant action**
- Action does not depend on sheep positions, goal location, or environment state
- No feedback or adaptation

---

### 3. Tipsy Shepherd

**Behavior:**
- Actions are sampled **entirely at random**
- No dependence on observations or goal
- No internal logic or strategy

---

## Future Extensions

- Multiple and dynamic obstacles
- Communication between shepherd agents
- Competitive multi-goal scenarios
- Domain randomization for sim-to-real transfer
