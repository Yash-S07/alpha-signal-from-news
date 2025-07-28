# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 01:37:07 2025

@author: Yash Singhal
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 23:29:54 2025

@author: Yash Singhal
"""

# Oil News Data Scraping

'''
An attempt to improve:
Downloading 2 years worth of data
'''

import pandas as pd
import yfinance as yf
import requests 
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from dateutil import parser


#_________________________________Configuration________________________________
ticker = 'CL=F'
ticker_name = 'OIL'
start_date = '2023-07-29'
end_date = '2025-07-25'
price_data_path = f'{ticker_name}_prices.csv'
news_data_path = f'{ticker_name}_news_oilprice_scrapped.csv'

#________________________________Get Price Data________________________________

def get_price_data(ticker,start,end):
    '''
    Parameters
    ----------
    ticker : string
        The symbol for the product to be scrapped
    start : string
            Start Date for data scrapping
    end : string
        End Date for data scrapping.

    Returns
    -------
    A datarframe containing oil prices' historical data

    '''
    print(f'Fetching {ticker} data from {start} to {end}')
    df = yf.download(ticker,start,end,interval = '1h',auto_adjust = False,multi_level_index=False)
    df.to_csv(price_data_path)
    print(f'Price Data saved to {price_data_path}')
    return df

#________________________Scrape for Real-Time News_____________________________

def scrap_oilprice_news(pages = 295):
    """
    Parameters
    ----------
    pages : int
        The number of pages to scrap. The default is 144 because it matches 1 year data

    Returns
    -------
    DataFrame containing all the scraped news.
    """
    print('Scrapping Oilprices.com for Latest News')
    base_url = 'https://oilprice.com/Latest-Energy-News/World-News/'
    news_list = []
    for page in tqdm(range(1,pages+1),desc = 'Scrapping Pages'):
        url = f'{base_url}/Page-{page}.html' if page>1 else base_url
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f'error fetching page {page}:{e}')
            continue
        
        
        soup = BeautifulSoup(response.text,'html.parser')
        article_containers = soup.find_all('div',class_ = 'categoryArticle')
         
        if not article_containers:
            print(f'No articles found on page {page}. Scrapping might be blocked or structure changed')
            continue
        
        for article in article_containers:
            title_tag = article.find('h2')
            byline_tag = article.find('p',class_ = 'categoryArticle__meta')
            
            
            if title_tag and byline_tag:
                headline = title_tag.get_text(strip = True)
                byline_text = byline_tag.get_text(strip = True)
                
                # --- CRITICAL TIMESTAMP PARSING ---
                try:
                    timestamp_str = byline_text.split(' | ')[0]
                    timestamp = parser.parse(timestamp_str)
                    news_list.append({'timestamp':timestamp,'headline':headline})
                except (IndexError, parser.ParserError) as e:
                    # This handles cases where the byline format is unexpected
                    print(f"Could not parse timestamp from byline: '{byline_text}'. Error: {e}")
        time.sleep(1)
    if not news_list:
        print("Fatal: Could not scrape any news items. Please check the website structure and your connection.")
        return pd.DataFrame()
          
    df = pd.DataFrame(news_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'],utc = True)
    df = df.sort_values('timestamp').reset_index(drop = True)
    df.to_csv(news_data_path,index = False)
    print(f'Scrapped {len(df)} news items and saved to {news_data_path}')
    return df

    
if __name__ == '__main__':
    #Step1: Get the price data
    price_data = get_price_data(ticker, start_date, end_date)
    
    #Step2: Get News Data
    news_data = scrap_oilprice_news()
    
    print('\nData sourcing Complete')
    if not news_data.empty:
        print('Sample of Scrapped news from Oilprices.com')
        print(news_data.head())

