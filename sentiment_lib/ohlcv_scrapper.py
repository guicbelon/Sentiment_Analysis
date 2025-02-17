import os
import pandas as pd
import requests

class AlphavantageOHLCVScrapper:
    def __init__(self) -> None:
        """
        Initializes the AlphavantageOHLCVScrapper class and loads the API key from environment variables.
        """
        self.api_key = os.getenv('ALPHAVANTAGE_KEY')
        
    def fetch_ohlcv_data(self, ticker: str, interval: str, month: str, outputsize: str = 'full'):
        """
        Fetches OHLCV (Open, High, Low, Close, Volume) data from the Alpha Vantage API.

        Args:
            ticker (str): The ticker symbol of the stock.
            interval (str): The interval between data points (e.g., '1min').
            month (str): The month for which to fetch the data.
            outputsize (str, optional): The size of the data to fetch. Defaults to 'full'.

        Returns:
            dict: The fetched OHLCV data.
        """
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={interval}&apikey={self.api_key}&month={month}&outputsize={outputsize}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'Time Series (1min)' in data:
                    return data['Time Series (1min)']
                else:
                    print(data)
        except Exception as error:
            print(f"Error in fetching data: {error}")
            return None
        
    def process_data_as_df(self, data: dict, ticker: str):
        """
        Processes the raw OHLCV data and converts it into a DataFrame.

        Args:
            data (dict): The raw OHLCV data.
            ticker (str): The ticker symbol of the stock.

        Returns:
            pd.DataFrame: The processed OHLCV data as a DataFrame.
        """
        df = pd.DataFrame.from_dict(data, orient='index')
        df.columns = [col + f"_{ticker}" for col in ['open', 'high', 'low', 'close', 'volume']]
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1]
        return df
    
    def create_ohlcv_df(self, ticker: str, open_time: str, close_time: str):
        """
        Creates a DataFrame containing OHLCV data for the given ticker and time range.

        Args:
            ticker (str): The ticker symbol of the stock.
            open_time (str): The start time for fetching OHLCV data.
            close_time (str): The end time for fetching OHLCV data.

        Returns:
            pd.DataFrame: The DataFrame containing the OHLCV data.
        """
        months = pd.date_range(start=open_time, end=close_time, freq='MS').strftime('%Y-%m').tolist()
        dfs_to_concat = []
        for month in months:
            data = self.fetch_ohlcv_data(ticker, '1min', month)
            if data:
                df = self.process_data_as_df(data, ticker)
                dfs_to_concat.append(df)
        df_ohlcv = pd.concat(dfs_to_concat)
        return df_ohlcv