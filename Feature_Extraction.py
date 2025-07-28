# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 01:44:20 2025

@author: Yash Singhal
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 00:56:26 2025

@author: Yash Singhal
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
#_____________________________________Configuration____________________________
ticker_name = 'OIL'
price_data_path = f'{ticker_name}_prices.csv'
news_data_path = f'{ticker_name}_news_oilprice_scrapped.csv'
processed_data_path = f'{ticker_name}_processed_features.pkl'


#Key Hyperparameters:
look_forward_period = 4 #hours
price_change_threshold = 0.007 #1.5% - Oil is more volatile than equities

fin_bert_model = "ProsusAI/finbert"

def create_ground_truth_labels(news_df,price_df):
    """
    Parameters
    ----------
    news_df : DataFrame
        DataFrame containing scrapped news
    price_df : DataFrame
        DataFrame containing OIL prices

    Returns
    -------
    DataFrame

    """
    print('Creating Ground Truth Labels')
    
    # Ensure both dataframes have similar timestamp order
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'],utc = True)
    price_df.sort_values('Datetime',inplace = True)
    price_df['returns'] = price_df['Close'].pct_change()
    price_df['volatility'] = price_df['returns'].rolling(window=24).std() # 24-hour rolling volatility
    
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'],utc = True)
    news_df.sort_values('timestamp',inplace = True)
    
    #Align news timestamp with price bar that contains it
    aligned_df = pd.merge_asof(
        left = news_df,
        right = price_df,
        left_on = 'timestamp',
        right_on = 'Datetime',
        direction = 'forward' # Take the first price bar which is available to us after the news 
        )
    
    #Find the price in the future    
    future_label  = []
    for idx,row in tqdm(aligned_df.iterrows(),total = aligned_df.shape[0],desc = 'Labelling'):
        future_time = row['Datetime'] + pd.Timedelta(hours = look_forward_period)
        
        # Find 1st available price after future_time
        future_price_rows = price_df[price_df['Datetime'] >= future_time]
        
        if not future_price_rows.empty:
            future_price = future_price_rows.iloc[0]['Close']
            current_price = row['Close']
            price_change = (future_price - current_price)/current_price
            
            if price_change > price_change_threshold:
                future_label.append(1) # Up
            elif price_change < -price_change_threshold:
                future_label.append(-1) # Down
            else:
                future_label.append(0) # Neutral
        else:
            future_label.append(np.nan) # Not enough future data to look at
            
    aligned_df['label'] = future_label
    aligned_df = aligned_df.dropna(subset = ['label']).reset_index(drop = True)
    aligned_df['label'] = aligned_df['label'].astype(int)
    aligned_df['hour_of_day'] = aligned_df['timestamp'].dt.hour
    aligned_df = aligned_df.dropna(subset=['label', 'volatility']).reset_index(drop=True)
    
    print(f"Label Distribution :\n{aligned_df['label'].value_counts()}")
    return aligned_df


def get_finbert_embeddings(headlines):
    """
    Parameters
    ----------
    headlines : list of strings
        Headline to get Finbert Embeddings
    Returns
    -------
    Embeddings
    """
    print('Generating Finbert Embeddings...')
    tokenizer = AutoTokenizer.from_pretrained(fin_bert_model)
    model = AutoModel.from_pretrained(fin_bert_model)
    
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    inputs = tokenizer(headlines,padding = True,truncation = True,return_tensors='pt',max_length = 512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # We use the embedding of the [CLS] token as the sentence representation
    cls_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
    return cls_embeddings




if __name__ == '__main__':
    news_df = pd.read_csv(news_data_path)
    price_df = pd.read_csv(price_data_path)
        
    labeled_df = create_ground_truth_labels(news_df,price_df)
    headlines = labeled_df['headline'].tolist()
    embeddings = get_finbert_embeddings(headlines)
    
    feature_df = pd.DataFrame(embeddings,index = labeled_df.index)
    feature_df.columns = [f'emb_{i}' for i in range(embeddings.shape[1])]
    
    final_df = pd.concat([
        labeled_df[['timestamp', 'headline', 'label', 'volatility', 'hour_of_day','Close']], # Add your features here
        feature_df
    ], axis=1)
    
    final_df.to_pickle(processed_data_path)
    
    print('\nProcessed Data with features saved to {processed_data_path}')
    print('Sample of Final DF')
    print(final_df.head())
                                 