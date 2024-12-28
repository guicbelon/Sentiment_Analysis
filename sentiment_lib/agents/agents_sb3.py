from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from sb3_contrib import RecurrentPPO

# Available DRL models to use
MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO, "rppo":RecurrentPPO}

# Default model Parameters
MODEL_KWARGS = {
    "A2C_PARAMS" : {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0005},
    "PPO_PARAMS" : {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "DDPG_PARAMS" : {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "TD3_PARAMS" : {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001},
    "SAC_PARAMS" : {
        "batch_size": 64,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "ERL_PARAMS" : {
        "learning_rate": 3e-5,
        "batch_size": 2048,
        "gamma": 0.985,
        "seed": 312,
        "net_dimension": 512,
        "target_step": 5000,
        "eval_gap": 30,
        "eval_times": 64,  
    },
    "RPPO_PARAMS" : {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64},
    "RLlib_PARAMS" : {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}}


NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

