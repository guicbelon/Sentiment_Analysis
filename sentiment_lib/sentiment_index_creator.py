import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch
from datetime import datetime
import os
import json


TICKERS_TEXTS = {
    'BA': ['Boeing', 'Dreamliner', 'Boeing Defense'],
    'MCD': ['McDonald\'s', 'McFlurry', 'Big Mac', 'McCafé'],
    'JNJ': ['Johnson & Johnson', 'Tylenol', 'Neutrogena', 'Janssen Pharmaceuticals'],
    'INTC': ['Intel', 'Pentium', 'Xeon', 'Intel Inside'],
    'BAC': ['Bank of America', 'Merrill Lynch', 'Better Money Habits', 'BankAmeriDeals'],
    'AMZN': ['Amazon', 'Prime Video', 'AWS', 'Alexa'],
    'NVDA': ['NVIDIA', 'GeForce', 'CUDA', 'Tegra'],
    'ORCL': ['Oracle', 'Oracle Database', 'Exadata', 'Java Platform'],
    'DIS': ['Disney', 'Disneyland', 'ESPN', 'Pixar'],
    'GS': ['Goldman Sachs', 'Marcus', '10,000 Small Businesses', 'Goldman Sachs Research'],
    'GOOG': ['Google', 'AdSense', 'YouTube', 'PageRank'],
    'MSFT': ['Microsoft', 'Windows', 'Xbox', 'Azure'],
    'TSLA': ['Tesla', 'Autopilot', 'Cybertruck', 'Gigafactory'],
    'META': ['Meta', 'WhatsApp', 'Instagram', 'Oculus'],
    'AAPL': ['Apple', 'iOS', 'Apple Watch', 'MacBook'],
    'V': ['Visa', 'VisaNet', 'Visa Direct', 'Cybersource'],
    'PFE': ['Pfizer', 'Comirnaty', 'Prevnar', 'Pfizer Oncology'],
    'KO': ['Coca-Cola', 'Minute Maid', 'Powerade'],
    'GE': ['General Electric', 'Predix', 'GE Renewable Energy', 'Additive Manufacturing'],
    'CAT': ['Caterpillar', 'Cat®', 'Track-Type Tractor', 'Cat Financial'],
    'XOM': ['Exxon Mobil', 'Mobil 1', 'Esso', 'ExxonMobil Chemical'],
    'CVX': ['Chevron', 'Chevron Phillips', 'Chevron Texaco', 'Delo'],
    'DE': ['Deere', 'GreenStar', 'S-Series Combine', 'John Deere Financial'],
    'F': ['Ford', 'F-150', 'Lincoln', 'EcoBoost'],
    'WMT': ['Walmart', 'Great Value', 'Sam\'s Club', 'Walmart+']
}

class NLPInfo():
    def __init__(self, 
                 bert_model:str = 'bert-large-uncased',
                 finbert_model:str = 'yiyanghkust/finbert-tone'):
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.finbert_model = BertForSequenceClassification.from_pretrained(finbert_model)
        self.finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.finbert_model.to(self.device)
        self.bert_model.eval()
        self.finbert_model.eval()
        
    def create_reference_embeddings(self, ticker:str):
        texts = TICKERS_TEXTS[ticker] + [ticker]
        self.reference_embeddings = []
        for text in texts:
            self.reference_embeddings.append(self.create_bert_embedding(text))
        
    def create_bert_embedding(self, text:str):
        with torch.no_grad():
            input_ids = self.bert_tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt").to(self.device)
            output = self.bert_model(input_ids)
            return output[0].mean(dim=1).cpu().numpy()
        
    def create_finbert_tensor(self, text:str):
        with torch.no_grad():
            input_ids = self.finbert_tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt").to(self.device)
            output = self.finbert_model(input_ids)
            return output.logits.cpu().numpy()
    
    def calculate_best_euclidean_distance(self, new_embedding:np.array):
        best_distance = np.inf
        for ref_embedding in self.reference_embeddings:
            distance = np.linalg.norm(new_embedding - ref_embedding)
            if distance < best_distance:
                best_distance = distance
        return best_distance
        
    def create_mlp_df_from_news(self, news_df:pd.DataFrame, temp_df=None, save_count:int=35):
        dfs_to_concat = []        
        ticker = news_df['ticker'].iloc[0]
        self.create_reference_embeddings(ticker)
        if temp_df is not None:
            news_df = news_df.loc[news_df.index > temp_df.index[-1]]    
            dfs_to_concat.append(temp_df)
            print("Tamanho do dataframe temporário: ",len(temp_df))
        count = 0
        for date in news_df.index:
            new_info = news_df.loc[date]
            texts_embedding_distance = []
            finbert_tensors = []
            new_text = new_info['news']
            if not pd.isna(new_text):
                text_split = new_text.split("\n")
                for text in text_split:
                    bert_embedding = self.create_bert_embedding(text)
                    euclidean_distance = self.calculate_best_euclidean_distance(bert_embedding)
                    finbert_tensor = self.create_finbert_tensor(text)
                    finbert_tensor_str = json.dumps((finbert_tensor[0]).tolist()) 
                    texts_embedding_distance.append(euclidean_distance)
                    finbert_tensors.append(finbert_tensor_str)
            df_to_add = pd.DataFrame({'date': date,
                               'ticker': ticker,                                   
                               'embedding_distance': texts_embedding_distance,
                               'finbert': finbert_tensors})
            df_to_add.set_index('date', inplace=True)
            dfs_to_concat.append(df_to_add)
            count +=1
            if count >= save_count:
                temp_df = pd.concat(dfs_to_concat)
                temp_df.to_csv(f"temps_nlp/{ticker}.csv")
                print(f"Last updated for {ticker}: {datetime.now()}")
                count = 0            
        return pd.concat(dfs_to_concat)

        