import gymnasium
from gymnasium.utils import seeding
import quantstats as qs
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from .reports import *

class TradeEnv(gymnasium.Env):
    def __init__(self, 
                 data: pd.DataFrame,
                 ticker: str, 
                 initial_balance: int = 1E6, 
                 time_window: int = 60,
                 trade_cost: float = 0.0004,
                 reward_logic: str = 'position_return',
                 reward_scale:float = 1,
                 only_long: bool = False,
                 verbose: bool = False):
        self.data = data
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.time_window = time_window
        self.trade_cost = trade_cost
        self.dates = list(data.index)
        self.episode_length = len(self.dates) - 1
        self.reward_logic = reward_logic
        self.reward_scale = reward_scale
        self.only_long = only_long
        self.verbose = verbose
        self.action_space = gymnasium.spaces.Discrete(3)
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.time_window, data.shape[1] - 1), dtype=np.float64)
        self.reset()
        
    def _get_current_state_from_datetime(self, datetime):
        datetime_index = self.dates.index(datetime)
        previous_index = datetime_index + 1 - self.time_window 
        df_previous = self.data.loc[self.dates[previous_index]:self.dates[datetime_index]]
        df_previous = df_previous.drop(columns=['close_real'])
        return df_previous.values

    def _get_current_return_from_datetime(self, datetime):
        datetime_index = self.dates.index(datetime)
        curr_price = self.data.loc[datetime]['close_real']
        prev_price = self.data.loc[self.dates[datetime_index - 1]]['close_real']
        return (curr_price - prev_price) / prev_price
    
    def _get_future_return_from_datetime(self, datetime, period):
        datetime_index = self.dates.index(datetime)
        curr_price = self.data.loc[datetime]['close_real']
        try:
            future_price = self.data.loc[self.dates[datetime_index + period]]['close_real']
        except:
            future_price = self.data.loc[self.dates[-1]]['close_real']
        return (future_price - curr_price) / curr_price
    
    def _get_metrics(self, returns):
        returns = np.array(returns)
        total_return = np.prod(1 + returns)
        mean_return = np.mean(returns)
        std_dev = np.std(returns, ddof=1) 
        sharpe_ratio = mean_return / std_dev
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
        hit_ratio = len(returns[returns > 0]) / len(returns[returns != 0])
        info = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "hit_ratio": hit_ratio
        }
        return info
        
    def _print_info(self, returns, title: str = "Trading Info"):
        info = self._get_metrics(returns)
        print(f"\n==== {title} ====")
        print("Total Return : {:.2f}".format(info['total_return']))
        print("Sharpe Ratio : {:.2f}".format(info['sharpe_ratio']))
        print("Max Drawdown : {:.2f}".format(info['max_drawdown']))
        print("Hit Ratio : {:.2f}".format(info['hit_ratio']))
        return info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def _return_reward(self):
        reward = self.curr_action_return
        if self.just_opened:
            reward -= self.trade_cost
        if self.use_log:
            reward = np.log(1 + reward)
        if self.is_long_open or self.is_short_open or self.just_closed:
            return reward
        return 0

    def _position_reward(self):
        reward = self.position_return
        if self.use_log:
            reward = np.log(reward)
        if self.is_long_open or self.is_short_open or self.just_closed:
            return reward
        return 0
    
    def _time_reward(self, period):
        curr_date = self.dates[self.time_index]
        future_returns = self._get_future_return_from_datetime(curr_date, period)
        if self.is_short_open:
            future_returns = -future_returns
        if self.just_opened:
            future_returns -= self.trade_cost
        if self.use_log:
            future_returns = np.log(1 + future_returns)
        if self.is_long_open or self.is_short_open or self.just_closed:
            return future_returns
        return 0
            
    def get_reward(self):
        reward_str_split = self.reward_logic.split("_")
        self.use_log = False
        if "log" in reward_str_split:
            self.use_log = True
        if "return" in reward_str_split:
            reward = self._return_reward()
        elif "position" in reward_str_split:
            reward = self._position_reward()
        elif "time" in reward_str_split:
            period = int(reward_str_split[-1])
            reward = self._time_reward(period)
        return reward*self.reward_scale

    def step(self, action):
        self.action = action -1
        curr_date = self.dates[self.time_index]
        self.curr_return = self._get_current_return_from_datetime(curr_date)
        done = False        
        self.just_opened = False
        self.just_closed = False
        if self.time_index == self.episode_length:
            done = True
            
        if self.is_long_open:
            if self.action == -1:
                self.just_closed = True
        
        elif self.is_short_open:       
            if self.action == 1:
                self.just_closed = True
            
        else:           
            if self.action != 0:
                if self.action == 1:
                    self.just_opened = True
                    self.is_long_open = True
                elif self.action == -1 and not self.only_long:
                    self.is_short_open = True
                    self.just_opened = True
        
        if self.is_long_open or self.is_short_open:
            self.curr_action_return = self.action*self.curr_return
        else:
            self.curr_action_return = 0
            
        if self.just_opened:
            self.open_position_date = curr_date
            self.position_return = (1 + self.curr_action_return - self.trade_cost)
            self.current_balance = self.current_balance * self.position_return
        else:
            self.current_balance = self.current_balance * (1 + self.curr_action_return)
        
        self.position_return = self.position_return*(1 + self.curr_action_return)
        
        self.returns.append(self.curr_action_return)
        self.temp_returns.append(self.curr_action_return)
        self.last_returns.pop(0)
        self.last_returns.append(self.curr_action_return)
        self.last_actions.pop(0)
        self.last_actions.append(action)        
        if self.verbose:
            if self.time_index % 400 == 0:
                self._print_info(self.temp_returns, "Temporarily Metrics")
                self.temp_returns = []
                self._print_info(self.returns, "Total Metrics")     

        reward = self.get_reward()
        if self.just_closed or done:
            if self.is_long_open or self.is_short_open:
                position_info = {
                "open_time": self.open_position_date,
                "close_time": curr_date,
                "returns": self.position_return - 1,
                "type": "long" if self.is_long_open else "short"
                    }
                self.position_memory.append(position_info)
            self.is_short_open = False
            self.is_long_open = False
            self.position_return = 1

        info = {
            "reward": reward,
            "current_balance": self.current_balance,
            "action": self.action,
            "current_returns": self.curr_action_return,
        }
        self.memory[curr_date] = info
        self.time_index += 1
        curr_state = self._get_current_state_from_datetime(curr_date)
        
        return curr_state, reward, done, done, info
    
    def reset(self, *,
              seed: int = 42,
              options=None):
        self.time_index = self.time_window - 1
        self.current_balance = self.initial_balance
        self.memory = {}
        self.position_memory = []
        self.returns = []
        self.temp_returns = []
        self.is_long_open = False
        self.is_short_open = False
        self.just_opened = False
        self.position_return = 1
        curr_date = self.dates[self.time_index]
        self.last_actions = [0] * self.time_window
        self.last_returns = [0] * self.time_window
        curr_state = self._get_current_state_from_datetime(curr_date)
        self._seed(seed)
        if self.verbose:
            now = datetime.now().strftime('%Y%m%d-%Hh%M')
            print(f"Reset Environment at {now}")
        return curr_state, self.memory
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def show_final_results(self, saving_path: str = None, **kwargs):
        report_creator = ReportCreator(env = self, saving_path = saving_path)
        report_creator._create_all_time_df()
        report_creator._create_positions_df()
        print("Number of positions: ", len(self.position_memory))
        self._print_info(report_creator.df_positions.returns.values, "Final Metrics")
        report_creator._create_img_df()
        report_creator._create_strategy_performance_img()
        plt.show()
        report_creator._create_html_report()
        report_creator._add_parameters_to_html(**kwargs)
        return report_creator.df_positions