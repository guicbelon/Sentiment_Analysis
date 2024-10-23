import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

REQUESTS_HEADER = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com',
        'DNT': '1', 
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

class FullTextScrapper:
    def __init__(self) -> None:
       pass

    
    def filter_text(self, text):
        return text
    
    def get_full_text_by_url(self, url):  
        response = requests.get(url, headers=REQUESTS_HEADER, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ''
        for parag in soup.find_all('p'):
            text += parag.get_text() + '\n'
        return text
    
    def create_df_from_full_text(self, df, ticker:str, temp_df=None, save_count:int=100):
        dates = []
        errors = []
        full_texts = []
        if temp_df is not None:
            errors = temp_df.errors.tolist()
            dates = temp_df.time.tolist()
            full_texts = temp_df.news.tolist()    
            df = df.loc[df.index > temp_df.index[-1]]    
            print("Tamanho do dataframe temporário: ",len(temp_df))
        print("Tamanho do dataframe faltante: ",len(df))
        count = 0
        for news_index in range(len(df)):
            url = df.url[news_index]
            try:
                full_text = self.get_full_text_by_url(url)
                errors.append(None)
            except Exception as e:
                full_text = ''
                errors.append(e)
            full_texts.append(full_text)
            dates.append(df.index[news_index])
            count +=1
            if count >= save_count:
                temp_df = pd.DataFrame({"time":dates, "news":full_texts, "errors":errors})
                temp_df.to_csv(f"temps_texts/{ticker}.csv")
                print(f"Last updated for {ticker}: {datetime.now()}")
                count = 0
        temp_df = pd.DataFrame({"time":dates, "news":full_texts, "errors":errors})
        temp_df.to_csv(f"temps_texts/{ticker}.csv")
        temp_df.index = pd.to_datetime(temp_df.time)
        return temp_df
            

