import sys
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.shepherd_env import ShepherdEnv

# --- CONFIGURATION ---
# Path to your Level 1 model (8 inputs)
OLD_MODEL_PATH = "models/level_1/ppo/PPO_5_model.zip"

# Path to save the new Level 3 compatible model (12 inputs)
NEW_MODEL_PATH = "models/level_3/ppo/PPO_5_converted.zip"

# Level 3 Settings (Target Environment)
TARGET_NUM_SHEEP = 2 
# ---------------------

def perform_surgery():
    print(f"🔪 Starting surgery on {OLD_MODEL_PATH}...")
    
    # 1. Create the Target Environment (Level 3 - 2 Sheep)
    # This tells us exactly what the new shape should be (12 inputs)
    env = ShepherdEnv(n_sheep=TARGET_NUM_SHEEP)
    
    # 2. Create a FRESH, UNTRAINED model with the new shape
    new_model = PPO("MlpPolicy", env, verbose=1)
    print(f"✨ Created fresh model with observation shape: {new_model.observation_space.shape}")

    # 3. Load the OLD, TRAINED model
    # We load it with 'custom_objects' to trick it into ignoring the env mismatch for a second
    old_model = PPO.load(OLD_MODEL_PATH, device="cpu")
    print(f"👴 Loaded old model.")

    # 4. Get the Policy Networks
    old_policy = old_model.policy.mlp_extractor.policy_net[0]
    new_policy = new_model.policy.mlp_extractor.policy_net[0]
    
    old_value = old_model.policy.mlp_extractor.value_net[0]
    new_value = new_model.policy.mlp_extractor.value_net[0]

    # Check shapes
    print(f"   Old Weight Shape: {old_policy.weight.shape} (Inputs: {old_policy.weight.shape[1]})")
    print(f"   New Weight Shape: {new_policy.weight.shape} (Inputs: {new_policy.weight.shape[1]})")

    # 5. TRANSPLANT WEIGHTS
    # We copy the weights for the first 8 inputs (original sheep & shepherd)
    # The remaining 4 inputs (for the 2nd sheep) will stay initialized to near-zero/random
    with torch.no_grad():
        # --- Policy Network Surgery ---
        # Copy existing weights
        new_policy.weight[:, :8] = old_policy.weight
        new_policy.bias[:] = old_policy.bias
        
        # --- Value Network Surgery ---
        new_value.weight[:, :8] = old_value.weight
        new_value.bias[:] = old_value.bias

    # 6. Save the Transformed Model
    os.makedirs(os.path.dirname(NEW_MODEL_PATH), exist_ok=True)
    new_model.save(NEW_MODEL_PATH)
    
    print(f"✅ Surgery Successful! Saved to: {NEW_MODEL_PATH}")
    print("🚀 You can now load this model in Level 3!")

if __name__ == "__main__":
    if not os.path.exists(OLD_MODEL_PATH):
        print(f"❌ Error: Could not find {OLD_MODEL_PATH}")
    else:
        perform_surgery()