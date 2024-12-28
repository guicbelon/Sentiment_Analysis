from .agents_sb3 import *
import numpy as np
import torch
import random
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in Tensorboard.
    """

    def __init__(self, verbose=0):
        """
        Initializes the TensorboardCallback.

        Parameters
        ----------
        verbose : int, optional
            Verbosity level, by default 0.
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        Method called at each step of the training process to record custom metrics in Tensorboard.

        Returns
        -------
        bool
            Whether to continue training (always True).
        """
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True
    
class DRLAgent:
    """
    Provides implementations for DRL algorithms.
    """

    def __init__(self, env):
        """
        Initializes the DRLAgent with a specific environment.

        Parameters
        ----------
        env : gym.Env
            The environment where the agent will be trained.
        """
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
        """
        Setup and return a DRL model.

        Parameters
        ----------
        model_name : str
            The name of the DRL model to use.
        policy : str, optional
            The policy model to use, by default "MlpPolicy".
        policy_kwargs : dict, optional
            Additional arguments for the policy, by default None.
        model_kwargs : dict, optional
            Additional arguments for the model, by default None.
        verbose : int, optional
            Verbosity level, by default 1.
        seed : int, optional
            Random seed for reproducibility, by default 42.
        tensorboard_log : str, optional
            Path to save Tensorboard logs, by default None.

        Returns
        -------
        object
            The initialized DRL model.
        
        Raises
        ------
        NotImplementedError
            If the model name is not implemented.
        """
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
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        """
        Train the DRL model and return the trained model.

        Parameters
        ----------
        model : object
            The DRL model to train.
        tb_log_name : str
            The name of the Tensorboard log.
        total_timesteps : int, optional
            The total number of timesteps to train, by default 5000.

        Returns
        -------
        object
            The trained DRL model.
        """
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        return model
    
    def DRL_prediction_load_from_file(self, model_name, cwd, deterministic=True):
        """
        Make a prediction using a model loaded from a file.

        Parameters
        ----------
        model_name : str
            The name of the DRL model to use.
        cwd : str
            The path to the model file.
        deterministic : bool, optional
            Whether to use deterministic actions, by default True.

        Returns
        -------
        gym.Env
            The environment after executing the prediction.
        """
        model = self.load_from_file(model_name, cwd)
        obs, info = self.env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            action = float(action)
            obs, reward, done, _, info = self.env.step(action)
        return self.env

    @classmethod
    def load_from_file(cls, model_name, cwd):
        """
        Load a model from a file.

        Parameters
        ----------
        model_name : str
            The name of the DRL model to load.
        cwd : str
            The path to the model file.

        Returns
        -------
        object
            The loaded DRL model.
        
        Raises
        ------
        NotImplementedError
            If the model name is not implemented.
        ValueError
            If the model fails to load.
        """
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except Exception as e:
            raise ValueError("Fail to load agent with error: {}".format(e))
        return model
