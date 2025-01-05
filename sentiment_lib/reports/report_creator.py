import pandas as pd
from bs4 import BeautifulSoup
import quantstats as qs
from matplotlib import pyplot as plt
import logging
import os
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

class ReportCreator:
    def __init__(self, 
                 env, 
                 saving_path: str = None,):
        self.env = env
        self.save_path = saving_path
        
        
    def _create_all_time_df(self):
        df = pd.DataFrame.from_dict(self.env.memory, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        ticker_info = self.env.data.loc[self.env.data.index.isin(df.index)]
        ticker_info = ticker_info[['close_real']]
        ticker_info = ticker_info.dropna()
        ticker_info['close_real'] = ticker_info.values / ticker_info.values[0]
        self.ticker_info = ticker_info
        df['current_balance'] = df['current_balance'] / df['current_balance'].iloc[0]
        df = df.join(ticker_info)
        self.all_time_df = df
        if self.save_path is not None:
            df.to_csv(f'{self.save_path}/all_time_df.csv', index=True)
        return df
    
    def _create_positions_df(self):
        df_positions = pd.DataFrame.from_dict(self.env.position_memory)
        df_positions.open_time = pd.to_datetime(df_positions.open_time)
        df_positions.close_time = pd.to_datetime(df_positions.close_time)
        self.df_positions = df_positions
        if self.save_path is not None:
            df_positions.to_csv(f'{self.save_path}/positions_df.csv', index=True)
        return df_positions
    
    def _create_img_df(self):
        df_close = []
        df_arrow_up = []
        df_arrow_down = []
        for row_iter in self.df_positions.iterrows():
            row = row_iter[1]
            close_value = (self.ticker_info.loc[self.ticker_info.index == row["close_time"]]).close_real.values[0]
            close_date = row["close_time"]
            df_close.append({"date":close_date, "x_close":close_value})
            open_date = row["open_time"]
            open_value = (self.ticker_info.loc[self.ticker_info.index == row["open_time"]]).close_real.values[0]
            if row["type"] == "long":
                df_arrow_up.append({"date":open_date, "arrow_up":open_value})
            elif row["type"] == "short":
                df_arrow_down.append({"date":open_date, "arrow_down":open_value})
        df_close = pd.DataFrame(df_close)
        if len(df_close) > 0:
            df_close.set_index("date", inplace=True)
        df_arrow_up = pd.DataFrame(df_arrow_up)
        if len(df_arrow_up) > 0:
            df_arrow_up.set_index("date", inplace=True)
        df_arrow_down = pd.DataFrame(df_arrow_down)
        if len(df_arrow_down) > 0:
            df_arrow_down.set_index("date", inplace=True)
        df = pd.concat([self.all_time_df, df_close, df_arrow_up, df_arrow_down], axis=1)
        self.img_df = df
        return df
    
    def _create_strategy_performance_img(self):
        plt.figure(figsize=(14, 5))
        plt.title('Trading Results')
        plt.plot(self.ticker_info, label=self.env.ticker, color='orange')
        plt.plot(self.all_time_df['current_balance'], label='Balance', color='blue')        
        if self.save_path is not None:
            plt.legend()
            plt.savefig(f'{self.save_path}/trading_results.png')
        if len(self.img_df.arrow_up) > 0:
            plt.plot(self.img_df['arrow_up'], label='Buy', marker='^', markersize=10, color='g', lw=0)
        if len(self.img_df.arrow_down) > 0:
            plt.plot(self.img_df['arrow_down'], label='Sell', marker='v', markersize=10, color='r', lw=0)
        plt.plot(self.img_df['x_close'], label='Close', marker='x', markersize=10, color='black', lw=0)
        plt.legend()
        if self.save_path is not None:
            plt.savefig(f'{self.save_path}/position_results.png')
    
    def _create_html_report(self):
        df_strategy = self.all_time_df.copy()
        if len(df_strategy) == 0:
            return
        if self.save_path is None:
            return
        df_strategy.index = pd.to_datetime(df_strategy.index)
        df_strategy['strategy_returns'] = df_strategy['current_balance'].pct_change()
        df_strategy[f'{self.env.ticker}_returns'] = df_strategy['close_real'].pct_change()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        template_path = os.path.join(dir_path, 'template.html')
        qs.reports.html(returns=df_strategy['strategy_returns'], 
                    benchmark=df_strategy[f'{self.env.ticker}_returns'], 
                    output=f'{self.save_path}/strategy_report.html',
                    template_path=template_path)
        df_html = self.df_positions.to_html(index=False)
        with open(f'{self.save_path}/strategy_report.html', "r") as file:
            soup = BeautifulSoup(file, "html.parser")
        positions_div = soup.new_tag("div", id="positions")
        positions_div.append(BeautifulSoup(df_html, "html.parser"))
        right_div = soup.find(id="right")
        right_div.insert_after(positions_div)
        with open(f'{self.save_path}/strategy_report.html', "w") as file:
            file.write(str(soup))
                
    def _add_parameters_to_html(self, **kwargs):
        if self.save_path is None:
            return
        combined_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                combined_kwargs.update(value)
            else:
                combined_kwargs[key] = value
        filtered_kwargs = {
            key: value for key, value in combined_kwargs.items()
            if isinstance(value, (int, float, str, bool, list))
        }
        df_col = pd.DataFrame(list(filtered_kwargs.items()), columns=['Parameter', 'Value'])
        df_html = df_col.to_html(index=False)
        if len(df_col) == 0:
            return
        with open(f'{self.save_path}/strategy_report.html', "r") as file:
            soup = BeautifulSoup(file, "html.parser")
        parameters_div = soup.new_tag("div", id="parameters")
        parameters_div.append(BeautifulSoup(df_html, "html.parser"))
        right_div = soup.find(id="right")
        right_div.insert(0, BeautifulSoup(df_html, 'html.parser'))
        with open(f'{self.save_path}/strategy_report.html', "w") as file:
            file.write(str(soup))