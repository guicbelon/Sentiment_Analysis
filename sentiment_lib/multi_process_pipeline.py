import threading
import warnings
from datetime import datetime
from .agents.DRL_agent import *
from .trade_env import *
import os
warnings.filterwarnings("ignore")


class MultiProcessingPipeline:
    
    def __init__(self, 
                 ticker: str,
                 train_kwargs: dict = None,
                 test_kwargs: dict = None,
                 n_process: int = 4,  
                 model: str = "rppo",):
        self.train_kwargs = train_kwargs
        self.test_kwargs = test_kwargs
        self.n_process = n_process
        self.model = model
        self.multi_process = []
        self.agents_train = []
        self.envs_train = []
        self.models_train = []
        self.tested_envs = []
        now = datetime.now()
        today = now.strftime('%Y%m%d')
        now = now.strftime('%Hh%M')
        self.base_dir = "results" + '/' + ticker + '/' + today + '/' + now + '/' + self.model
    
    def get_environment_data(self, kwargs:str=None):
        new_env = TradeEnv
        if kwargs is None:
            instanced_env = new_env()
        else:
            instanced_env = new_env(**kwargs)
        return instanced_env
        
    def create_info_directory(self, index: int) -> str:
        directory = self.base_dir + "_" + str(index)
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
        return directory
        
    def create_process_conditions(self, 
                                  model_parameters: dict,
                                  policy_parameters: dict = None, 
                                  policy: str = "MlpLstmPolicy",
                                  total_timesteps: int = 1000000):
        for process in range(self.n_process):
            print("\nCreating process ", process + 1)
            saving_dir = self.create_info_directory(process)
            train_environment = self.get_environment_data(self.train_kwargs)
            self.envs_train.append(train_environment)
            env_train, _ = train_environment.get_sb_env()
            agent = DRLAgent(env_train)
            self.agents_train.append(agent)
            model = agent.get_model(self.model, 
                                    policy=policy,
                                    model_kwargs=model_parameters, 
                                    policy_kwargs=policy_parameters,
                                    verbose=0)
            new_logger = configure(saving_dir, ["tensorboard"])
            model.set_logger(new_logger)
            process_args = (agent, model, saving_dir, total_timesteps, process)
            process = threading.Thread(target=self.train_test_individual_model, args=process_args)
            self.multi_process.append(process)
            self.models_train.append(model)  
            
    def train_test_individual_model(self, agent, model, saving_dir, total_timesteps, process_index):
        trained = agent.train_model(model, saving_dir, total_timesteps)
        model_name = saving_dir + f"/agent_{self.model}.zip"
        trained.save(model_name)        
        test_environment = self.get_environment_data(self.test_kwargs)
        agent_test = DRLAgent(test_environment)
        tested_env = agent_test.DRL_prediction_load_from_file(self.model, model_name)
        info = {
            "process_index": process_index,
            "saving_path": saving_dir,
            "tested_env": tested_env,
        }
        self.tested_envs.append(info)
        
    def multi_tasking(self): 
        for process in self.multi_process:
            process.start()
        for process in self.multi_process:
            process.join()
    
    def get_results(self):
        for info in self.tested_envs:
            try:
                index = info['process_index']
                print(f"\nResults of process {index + 1}")
                env = info['tested_env']
                saving_path = info['saving_path']
                df_balance = env.show_final_results(saving_path)
            except Exception as e:
                print(e)
                pass
