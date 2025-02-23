import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ast
import warnings
warnings.filterwarnings("ignore")
import pandas_ta as pdta
from .params import *


def returns_indicator(close, length, *args, **kwargs):
    """
    Calculates a returns indicator based on the ratio between the current returns and
    its rolling average.
    
    Parameters:
        close (pd.Series): Historical series of closing prices.
        length (int): Window size for calculating the rolling average of the returns.
        *args: Additional positional arguments (not used).
        **kwargs: Additional keyword arguments (not used).
    
    Returns:
        pd.Series: Returns ratio values.
    """
    returns = close.pct_change()
    returns = abs(returns)
    rolling_returns = returns.rolling(window=length).mean()
    returns_ratio = returns / rolling_returns
    return returns_ratio


def bollinger_bands_indicator(close, length, *args, **kwargs): 
    """
    Calculates a Bollinger Bands score based on the z-score of the closing prices.

    Parameters:
        close (pd.Series): Historical series of closing prices.
        length (int): Window size for calculating the rolling mean and standard deviation.
        *args: Additional positional arguments (not used).
        **kwargs: Additional keyword arguments (not used).

    Returns:
        pd.Series: Bollinger Bands scores (z-scores of the closing prices).
    """
    rolling_mean = close.rolling(window=length).mean()
    rolling_std = close.rolling(window=length).std()
    bb_score = (close - rolling_mean) / rolling_std
    return bb_score

def rsi_func(close, length=14, *args, **kwargs):
    """
    Calculates the Relative Strength Index (RSI) using the Moving Average Wilder (RMA).
    
    Parameters:
        close (pd.Series): Historical series of closing prices.
        length (int): length for RSI calculation (default: 14).
        
    Returns:
        pd.Series: RSI values.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(window=length, min_periods=length).mean()
    avg_down = down.rolling(window=length, min_periods=length).mean()
        
    for i in range(length + 1, len(avg_up)):
        avg_up.iloc[i] = (avg_up.iloc[i-1] * (length - 1) + up.iloc[i]) / length
        avg_down.iloc[i] = (avg_down.iloc[i-1] * (length - 1) + down.iloc[i]) / length
    m_div = avg_up / avg_down
    rsi = 100 - (100 / (1 + m_div))
    return rsi


def stochastic_rsi_func(close, rsi_length=14, length=14, *args, **kwargs):
    """
    Calculates the Stochastic RSI based on the RSI.
    
    Parameters:
        close (pd.Series): Historical series of closing prices.
        rsi_length (int): RSI length (default: 14).
        length (int): Stochastic RSI length (default: 14).
        
    Returns:
        pd.Series: Stochastic RSI values.
    """
    rsi = rsi_func(close, length=rsi_length)
    min_rsi = rsi.rolling(window=length).min()
    max_rsi = rsi.rolling(window=length).max()
    stoch_rsi = 100*(rsi - min_rsi) / (max_rsi - min_rsi)    
    return stoch_rsi

TA_INFO = {
        'stochrsi':{
            "method": stochastic_rsi_func,
            "period": 14,
            "secondary_period": 14},
        'rsi':{
            "method": rsi_func,
            "period": 14},
        'atr':{
            "method": pdta.atr,
            "period": 14},
        'mom':{
            "method": pdta.mom,
            "period": 14},
        'cci':{
            "method": pdta.cci,
            "period": 14},
        'stoch':{
            "method": pdta.stoch,
            "period": 14},
        
        'macd':{
            "method": pdta.macd,
            "period": 12,
            "secondary_period": 26,
            "signal_period": 9},
        'bollinger_bands':{
            "method": bollinger_bands_indicator,
            "period": 20},
        'returns':{
            "method": returns_indicator,
            "period": 14}
    }

class RLDatabase:
    def __init__(self,
                 start_train: str,
                 end_train: str,
                 end_test: str,
                 ticker: str = 'AAPL',
                 time_window: int = 60,
                 include_sentiment_info:bool = False,
                 normalize_data: bool = True):
        """
        Initializes the RLDatabase class with the given parameters.

        Args:
            start_train (str): The start date for the training period.
            end_train (str): The end date for the training period.
            end_test (str): The end date for the testing period.
            ticker (str, optional): The ticker symbol of the stock. Defaults to 'AAPL'.
            time_window (int, optional): The time window for the data. Defaults to 60.
            include_sentiment_info (bool, optional): Whether to include sentiment information. Defaults to False.
            normalize_data (bool, optional): Whether to normalize the data. Defaults to True.
        """
        self.start_train = pd.to_datetime(start_train)
        self.end_train = pd.to_datetime(end_train)
        self.end_test = pd.to_datetime(end_test)
        self.ticker = ticker
        self.time_window = time_window
        self.include_sentiment_info = include_sentiment_info
        self.normalize_data = normalize_data
        self.ohlcv = None
        self.ta_info = {}
        self.ta_df = None
        self.sentiment_df = None
        self.train_test_info = None
        
    def _create_ohlcv_df(self):
        """
        Creates the OHLCV DataFrame by reading data from a CSV file.

        Returns:
            pd.DataFrame: The OHLCV DataFrame.
        """
        if self.ohlcv is not None:
            return self.ohlcv
        try:
            ohlcv = pd.read_csv(f"{OHLCV_PATH}/{self.ticker}_ohlcv.csv", index_col=0)
            ohlcv.index = pd.to_datetime(ohlcv.index)
            ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
            self.ohlcv = ohlcv
        except Exception as e:
            print("Error in creating OHLCV df: {e}")
        return self.ohlcv
    
    def _create_ta_df(self):
        """
        Creates the technical analysis DataFrame by calculating various technical indicators.

        Returns:
            pd.DataFrame: The technical analysis DataFrame.
        """
        if self.ta_df is not None:
            return self.ta_df
        try:
            ohlcv = self._create_ohlcv_df()
            for indicator in TA_INFO.keys():
                technical_indicator = TA_INFO[indicator]["method"]
                ta_period = TA_INFO[indicator]['period']
                if indicator == 'stochrsi':
                        secondary_ta_period = TA_INFO[indicator]['secondary_period']
                        complete_ta_data = technical_indicator(close=ohlcv['close'], length=ta_period,
                                                        rsi_length = secondary_ta_period, k = 1, d = 1)
                elif indicator == 'macd':
                    secondary_ta_period = TA_INFO[indicator]['secondary_period']
                    signal_period = TA_INFO[indicator]['signal_period']
                    complete_ta_data = technical_indicator(close = ohlcv['close'], fast = ta_period,
                                                    slow = secondary_ta_period, signal = signal_period)
                    complete_ta_data = complete_ta_data[f"MACDh_{ta_period}_{secondary_ta_period}_{signal_period}"]
                elif indicator == 'stoch':
                    complete_ta_data = technical_indicator(high = ohlcv['high'], 
                                                        low = ohlcv['low'], 
                                                        close = ohlcv['close'], 
                                                        k = ta_period, d = 1, smooth_k = 1)
                    complete_ta_data = complete_ta_data[f"STOCHk_{ta_period}_1_1"]
                else:
                    indicator_kwargs = {
                        'close': ohlcv['close'],
                        'high': ohlcv['high'],
                        'low': ohlcv['low'],
                        'volume': ohlcv['volume'],
                        'length': ta_period,
                        'k':1, 'd':1,
                    }
                    complete_ta_data = technical_indicator(**indicator_kwargs)   
                self.ta_info[indicator] = complete_ta_data    
        except Exception as e:
            print(f"Error in creating TA df: {e}")
        self.ta_df = pd.DataFrame(self.ta_info)
        return self.ta_df
    
    def _create_finbert_score(self, df):
        """
        Creates the FinBERT sentiment score for each row in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the sentiment data.

        Returns:
            pd.DataFrame: The DataFrame with the FinBERT sentiment score added.
        """
        finbert_score = []
        for _, row in df.iterrows():
            finbert_info = ast.literal_eval(row["finbert"]) # neutral; positive; negative
            positive_info = finbert_info[1]
            negative_info = finbert_info[2]
            sentiment_score = positive_info - negative_info
            finbert_score.append(sentiment_score)
        df['finbert_score'] = finbert_score
        return df
    
    def _create_sentiment_by_news(self, df):
        """
        Creates the sentiment score for each news article.

        Args:
            df (pd.DataFrame): The DataFrame containing the news data.

        Returns:
            pd.DataFrame: The DataFrame with the sentiment score for each news article.
        """
        df_news = {}
        grouped_df = df.groupby("date")
        for group in grouped_df:
            index_date = group[0]
            sentiment_info = group[1]
            weights = 1 / sentiment_info['embedding_distance'] 
            sentiment_score = (sentiment_info['finbert_score'] * weights).sum() / weights.sum()
            df_news[index_date] = sentiment_score
        return pd.DataFrame.from_dict(df_news, orient='index', columns=['sentiment_score'])

    def _create_sentiment_index(self, news_df, start_time, end_time, lookback_window='1D'):
        """
        Creates the sentiment index based on the news data.

        Args:
            news_df (pd.DataFrame): The DataFrame containing the news data.
            start_time (str): The start time for the sentiment index.
            end_time (str): The end time for the sentiment index.
            lookback_window (str, optional): The lookback window for the sentiment index. Defaults to '1D'.

        Returns:
            pd.DataFrame: The sentiment index DataFrame.
        """
        news_df.index = pd.to_datetime(news_df.index)
        max_score = news_df['sentiment_score'].abs().max()
        news_df['normalized_sentiment'] = news_df['sentiment_score'] / max_score
        
        time_index = pd.date_range(start=start_time, end=end_time, freq="1T")
        sentiment_index = pd.Series(0.0, index=time_index)

        lookback_window_timedelta = pd.Timedelta(lookback_window)

        for current_time in time_index:
            window_start = current_time - lookback_window_timedelta
            relevant_news = news_df[(news_df.index > window_start) & (news_df.index <= current_time)]

            if not relevant_news.empty:
                time_deltas = (current_time - relevant_news.index).total_seconds()
                time_deltas = np.array(time_deltas) 
                weights = np.exp(-time_deltas / lookback_window_timedelta.total_seconds())
                weighted_sentiment = (relevant_news['normalized_sentiment'].values * weights).sum()
                sentiment_index[current_time] = weighted_sentiment / weights.sum()
            else:
                previous_value = sentiment_index.get(current_time - pd.Timedelta("1T"), 0.0)
                sentiment_index[current_time] = previous_value * 0.9
        sentiment_index = pd.DataFrame(sentiment_index, columns=["sentiment_score"])
        return sentiment_index
    
    def _create_sentiment_df(self):
        """
        Creates the sentiment DataFrame by reading data from a CSV file and processing it.

        Returns:
            pd.DataFrame: The sentiment DataFrame.
        """
        if self.sentiment_df is not None:
            return self.sentiment_df
        try:
            ebd_fbrt = pd.read_csv(f"{SENTIMENT_PATH}/{self.ticker}_ebd_fbrt.csv", index_col=0)
            ebd_fbrt.index = pd.to_datetime(ebd_fbrt.index)
            ebd_fbrt = self._create_finbert_score(ebd_fbrt)
            ebd_fbrt = self._create_sentiment_by_news(ebd_fbrt)
            ebd_fbrt = self._create_sentiment_index(ebd_fbrt, start_time=OPEN_DATE, end_time=CLOSE_DATE)
            self.sentiment_df = ebd_fbrt
            return self.sentiment_df
      
        except Exception as e:
            print(f"Error in creating sentiment df: {e}")
            
    def create_train_test_info(self):
        """
        Creates the training and testing data by combining OHLCV, technical analysis, and sentiment data.

        Returns:
            dict: A dictionary containing the training and testing data.
        """
        if self.train_test_info is not None:
            return self.train_test_info
        ohlcv = self._create_ohlcv_df()
        ta_df = self._create_ta_df()
        base_info_df = pd.concat([ohlcv, ta_df], axis = 1)
        base_info_df.index = pd.to_datetime(base_info_df.index)
        close_info = base_info_df['close']
        close_info.name = "close_real"
        end_train_index = close_info.index.searchsorted(self.end_train)
        
        start_of_test = base_info_df.index[end_train_index - self.time_window]
        
        base_train_df = base_info_df.loc[self.start_train:self.end_train].dropna()
        base_test_df = base_info_df.loc[start_of_test:self.end_test].dropna()
        if self.normalize_data:
            scaler = StandardScaler()
            base_train_df = pd.DataFrame(scaler.fit_transform(base_train_df), columns=base_train_df.columns, index=base_train_df.index)
            base_test_df = pd.DataFrame(scaler.transform(base_test_df), columns=base_test_df.columns, index=base_test_df.index)
        train_df = pd.concat([close_info, base_train_df], axis=1)
        test_df = pd.concat([close_info, base_test_df], axis=1)
        if self.include_sentiment_info:
            sentiment_df = self._create_sentiment_df()
            train_df = pd.concat([train_df, sentiment_df], axis=1)
            train_df.sentiment_score.fillna(0, inplace=True)
            test_df = pd.concat([test_df, sentiment_df], axis=1)
            test_df.sentiment_score.fillna(0, inplace=True)
        train_df = train_df.dropna()
        test_df = test_df.dropna()
        self.train_test_info = {"train": train_df, "test": test_df}
        return self.train_test_info