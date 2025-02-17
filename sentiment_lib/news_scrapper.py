from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
import requests

class AlphavantageScrapper:
    def __init__(self) -> None:
        """
        Initializes the AlphavantageScrapper class and loads the API key from environment variables.
        """
        load_dotenv()
        self.api_key = os.getenv('ALPHAVANTAGE_KEY')
        
    def process_data_as_df(self, data, ticker: str):
        """
        Processes the raw news data and converts it into a DataFrame.

        Args:
            data (list): The raw news data.
            ticker (str): The ticker symbol of the stock.

        Returns:
            pd.DataFrame: The processed news data as a DataFrame.
        """
        new_data = []
        for info in data:
            ticker_info = [item for item in info['ticker_sentiment'] if item['ticker'] == ticker][0]
            filtered_info = {
                'time': info['time_published'],
                'title': info['title'],
                'url': info['url'],
                'summary': info['summary'],
                'source': info['source_domain'],
                'overall_sentiment_score': float(info['overall_sentiment_score']),
                'overall_sentiment_label': info['overall_sentiment_label'],
                'ticker': ticker,
                'ticker_relevance_score': float(ticker_info['relevance_score']),
                'ticker_sentiment_score': float(ticker_info['ticker_sentiment_score']),
                'ticker_sentiment_label': ticker_info['ticker_sentiment_label']
            }
            new_data.append(filtered_info)
        df = pd.DataFrame(new_data)
        df.time = pd.to_datetime(df.time, format='%Y%m%dT%H%M%S')
        df.set_index('time', inplace=True)
        return df
    
    def fetch_news_data(self, ticker: str, open_time: datetime, close_time: datetime, limit: int = 1000):
        """
        Fetches news data from the Alpha Vantage API.

        Args:
            ticker (str): The ticker symbol of the stock.
            open_time (datetime): The start time for fetching news data.
            close_time (datetime): The end time for fetching news data.
            limit (int, optional): The maximum number of news articles to fetch. Defaults to 1000.

        Returns:
            list: The fetched news data.
        """
        open_time = open_time.strftime('%Y%m%dT%H%M')
        close_time = close_time.strftime('%Y%m%dT%H%M')
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&time_from={open_time}&time_to={close_time}&sort=EARLIEST&limit={limit}&apikey={self.api_key}'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'feed' in data:
                    return data['feed']
                else:
                    print(data)
                    raise Exception("API limit reached")
        except Exception as error:
            print(f"Error in fetching data: {error}")
            return None
        
    def create_news_df(self, ticker: str, open_time: datetime, close_time: datetime, limit: int = 1000):
        """
        Creates a DataFrame containing news data for the given ticker and time range.

        Args:
            ticker (str): The ticker symbol of the stock.
            open_time (datetime): The start time for fetching news data.
            close_time (datetime): The end time for fetching news data.
            limit (int, optional): The maximum number of news articles to fetch. Defaults to 1000.

        Returns:
            pd.DataFrame: The DataFrame containing the news data.
        """
        news_df = pd.DataFrame()
        last_date = open_time
        previous_date = open_time
        while True:
            news_data = self.fetch_news_data(ticker, last_date, close_time, limit)
            if news_data is not None:
                news_data = self.process_data_as_df(news_data, ticker)
                news_df = pd.concat([news_df, news_data])
                last_date = news_data.index[-1]
                if previous_date == last_date:
                    break
                previous_date = last_date
            else:
                break
        news_df = news_df[~news_df.index.duplicated(keep='first')]
        return news_df