from .agents_sb3 import *
import numpy as np
import torch
import random
from stable_baselines3.common.utils import set_random_seed
from .callbacks import *

class DRLAgent:
    """
    Provides implementations for DRL algorithms.
    """
    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed: int = 42,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name.upper()+"_PARAMS"]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        set_random_seed(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
            **model_kwargs,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000, early_stop_reward=None, early_stop_n_steps=1000):
        """
        Train the DRL model and return the trained model.
        
        Parameters:
            model : object
                The DRL model to train.
            tb_log_name : str
                The name of the Tensorboard log.
            total_timesteps : int, optional
                The total number of timesteps to train (default: 5000).
            early_stop_reward : float, optional
                If not None, training will stop if the mean reward of the last early_stop_n_steps is >= early_stop_reward (default: None).
            early_stop_n_steps : int, optional
                Number of steps to calculate the average (default: 1000).
        
        Returns:
            object: The trained DRL model.
        """
        callbacks = [TensorboardCallback()]
        if early_stop_reward is not None:
            callbacks.append(EarlyStoppingCallback(min_reward=early_stop_reward, n_steps=early_stop_n_steps, verbose=False))
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callbacks,
        )
        return model
    
    def DRL_prediction_load_from_file(self, model_name, cwd, deterministic=True):
        model = self.load_from_file(model_name, cwd)
        obs, info = self.env.reset()
        done = False
        prev_states = None
        while not done:
            action, prev_states = model.predict(obs, state=prev_states, deterministic=deterministic)
            action = float(action)
            obs, reward, done, _, info = self.env.step(action)
        return self.env

    @classmethod
    def load_from_file(cls, model_name, cwd):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except Exception as e:
            raise ValueError("Fail to load agent with error: {}".format(e))
        return model
