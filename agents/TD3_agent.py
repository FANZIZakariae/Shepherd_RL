import os
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_td3_mlp(
    env, 
    eval_env, 
    timesteps=1000000, 
    checkpoint_dir=None, 
    criculam_learning=False, 
    log_dir="./logs/td3/", 
    model_save_path=None, 
    tb_log_name="TD3"
):

    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
    else:
        model_save_path = log_dir

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    if checkpoint_dir is not None:
        print(f"Loading TD3 checkpoint from {checkpoint_dir}...")
        model = TD3.load(
            checkpoint_dir, 
            env=env, 
            tensorboard_log=log_dir, 
            custom_objects={'action_noise': action_noise}
        )
        model.action_noise = action_noise
        
        # --- FIX: Removed optimizer reset that wiped weights ---
        if criculam_learning:
            print("Curriculum Mode: Fine-tuning...")
            model.learning_rate = 1e-4 
            model.set_env(env)
            
    else:
        print("Training TD3 from scratch...")
        model = TD3(
            "MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            action_noise=action_noise,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=1,
            tensorboard_log=log_dir
        )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_save_path,
        log_path=model_save_path, 
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True, 
        render=False
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=tb_log_name)
    return model