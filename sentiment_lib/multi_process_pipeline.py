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
        """
        Creates an environment instance based on the environment type and parameters.
        
        Parameters
        ----------
        kwargs : dict, optional
            Dictionary of environment-specific parameters (default is None).
        
        Returns
        -------
        object
            The environment instance.
        """
        new_env = TradeEnv
        if kwargs is None:
            instanced_env = new_env()
        else:
            instanced_env = new_env(**kwargs)
        return instanced_env
        
    def create_info_directory(self, index: int) -> str:
        """
        Creates a directory for storing information specific to a process.

        Parameters
        ----------
        index : int
            Index of the process.

        Returns
        -------
        str
            The path of the created directory.
        """
        directory = self.base_dir + "_" + str(index)
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
        return directory
        
    def create_process_conditions(self, 
                                  model_parameters: dict,
                                  policy_parameters: dict = None, 
                                  policy: str = "MlpLstmPolicy",
                                  total_timesteps: int = 1000000):
        """
        Sets up conditions for each process, including creating environments, agents, and models.

        Parameters
        ----------
        model_parameters : dict
            Dictionary of model-specific parameters.
        policy_parameters : dict, optional
            Dictionary of policy-specific parameters (default is None).
        policy : str, optional
            The policy type to use (default is "MlpLstmPolicy").
        total_timesteps : int, optional
            Total number of timesteps to train the model (default is 1000000).
        """
        self.model_parameters = model_parameters
        self.policy_parameters = policy_parameters
        self.policy = policy
        self.total_timesteps = total_timesteps
        for process in range(self.n_process):
            print("\nCreating process ", process)
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
        """
        Trains the model and tests it on a test environment, saving the results.

        Parameters
        ----------
        agent : DRLAgent
            The agent responsible for training the model.
        model : object
            The reinforcement learning model.
        saving_dir : str
            Directory to save the trained model and results.
        total_timesteps : int
            Total number of timesteps to train the model.
        process_index : int
            Index of the process.
        """
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
        """
        Starts all processes in parallel and waits for them to complete.
        """
        for process in self.multi_process:
            process.start()
        for process in self.multi_process:
            process.join()
    
    def create_report(self, env_test, env_train, saving_path, **kwargs):
        """
        Creates a report for the final results of the training and testing process.
        """
        report_creator = ReportCreator(env_test=env_test, env_train=env_train, saving_path=saving_path)
        report_creator._create_all_time_df()
        report_creator._create_positions_df()
        positions_created = len(report_creator.df_positions)
        env_test._print_info(report_creator.df_positions.returns.values, "Final Metrics")
        metrics = env_test._get_metrics(report_creator.df_positions.returns.values)
        kwargs = {**kwargs,
                  **metrics,
            "start_train_date": env_train.data.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_train_date": env_train.data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "end_test_date": env_test.data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "positions_created":positions_created,
        }
        report_creator._create_img_df()
        report_creator._create_strategy_performance_img()
        plt.show()
        report_creator._create_train_test_price_img()
        report_creator._create_html_report()
        report_creator._add_parameters_to_html(**kwargs)
    
    def get_results(self):
        """
        Retrieves and displays the final results from each tested environment.
        """
        for info in self.tested_envs:
            try:
                index = info['process_index']
                print(f"\nResults of process {index}")
                env_test = info['tested_env']
                env_train = self.envs_train[index]
                saving_path = info['saving_path']
                kwargs = {
                          "process_index": index,
                          "original_path": saving_path,
                          "test_kwargs": self.test_kwargs,
                          "model": self.model,
                          "model_parameters": self.model_parameters,
                          "policy_parameters": self.policy_parameters,
                          "policy": (self.policy if type(self.policy) == str else self.policy.__name__),  
                          "total_timesteps": self.total_timesteps}
                self.create_report(env_test = env_test, env_train = env_train,saving_path = saving_path, **kwargs)
            except Exception as e:
                print(e)
                pass

