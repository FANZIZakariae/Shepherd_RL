import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

def train_rl_agent_ppo_mlp(
    env, 
    eval_env, 
    timesteps=2000000, 
    checkpoint_dir=None, 
    criculam_learning=False, 
    log_dir="./logs/ppo/", 
    model_save_path=None, 
    tb_log_name="PPO"
):
    
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
    else:
        model_save_path = log_dir

    if checkpoint_dir is not None:
        print(f"Loading checkpoint from {checkpoint_dir}...")
        model = PPO.load(
            checkpoint_dir,
            env=env,
            n_steps=2048,
            ent_coef=0.003,
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir 
        )
        
        # --- FIX: Do NOT call _setup_model() here ---
        # Only adjust parameters relevant for fine-tuning
        if criculam_learning:
            print("Curriculum Mode: Keeping weights, lowering LR for fine-tuning...")
            model.learning_rate = 1e-4  # Lower LR to prevent destroying pre-trained weights
            # We explicitly set the new environment
            model.set_env(env)
            
    else:
        print("No checkpoint provided, training from scratch.")
        model = PPO(
            "MlpPolicy",
            env=env,
            n_steps=2048,
            ent_coef=0.003, # Higher entropy for scratch training
            learning_rate=3e-4,
            gamma=0.99,
            verbose=1,
            tensorboard_log=log_dir 
        )
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_save_path, 
        log_path=model_save_path, 
        eval_freq=20000,
        n_eval_episodes=20,
        deterministic=True, 
        render=False
    )
    
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=tb_log_name)
    return model