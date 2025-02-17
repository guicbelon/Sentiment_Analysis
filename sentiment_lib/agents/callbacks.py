from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in Tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Callback that stops training if the average reward of the last n_steps is >= min_reward.
    """
    def __init__(self, min_reward: float, n_steps: int = 1000, verbose: bool = False):
        """
        Parameters:
            min_reward (float): Minimum average reward value to trigger early stopping.
            n_steps (int): Number of steps to calculate the average.
            verbose (Bool): Verbosity status.
        """
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.min_reward = min_reward
        self.n_steps = n_steps
        self.step_rewards = []

    def _on_step(self) -> bool:
        try:
            reward = self.locals["rewards"][0]
        except Exception:
            reward = self.locals["reward"][0]
        
        self.step_rewards.append(reward)
 
        if len(self.step_rewards) >= self.n_steps:
            recent_avg = np.mean(self.step_rewards[-self.n_steps:])
            if self.verbose:
                print(f"EarlyStoppingCallback: Average of the last {self.n_steps} steps = {recent_avg}")
            if recent_avg >= self.min_reward:
                if self.verbose:
                    print(f"EarlyStoppingCallback: Early stopping triggered, average {recent_avg} >= {self.min_reward}")
                return False
        return True 