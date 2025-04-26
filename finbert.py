import torch
from transformers import pipeline

import requests
from datetime import datetime, timedelta
from collections import Counter

import yfinance

api_key = '4867727998754629ad10c19521b7015e'

def fetch_news(ticker):
    today = datetime.today().date()
    one_month_ago = today - timedelta(days=30)

    url = 'https://newsapi.org/v2/everything'

    params = {
        'q': 'ticker',               
        'from': one_month_ago.isoformat(),        
        'to': today.isoformat(),   
        'language': 'en',           
        'sortBy': 'relevancy',   
        'pageSize': 100,              
        'apiKey': api_key            
    }

    response = requests.get(url, params=params)

    data = response.json()

    if response.status_code == 200:
        titles = [article['title'] for article in data['articles']]
        return titles
    else:
        print("Error:", data)
        return None


def stock_ticker_information(stock_ticker):
    stock_information = yfinance.Ticker(stock_ticker)

    stock_news_articles = stock_information.get_news(count=100)
    if not stock_news_articles:
        print("There was no news found on this stock")
        return []

    stock_article_titles = []
    stock_article_summaries = []
    stock_article_info = []
    for article in stock_news_articles:
        title = ""
        summary = ""
        try:
            title = article['content']['title']
        except KeyError:
            title = "" # in the case where a title cant be found
    
        try:
            summary = article['content']['summary']
        except KeyError:
            summary = "" # in the case where a summary cant be found
        
        stock_article_titles.append(title)
        stock_article_summaries.append(summary)
        stock_article_info.append("TITLE: " + title + ", SUMMARY: " + summary)

    return stock_article_info

def sentiment(text):

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)

    result = pipe(text)[0]['label']

    return result

if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ")

    titles = fetch_news(ticker)
    information = stock_ticker_information(ticker)

    all_sentiments = []
    for info in information:
        all_sentiments.append(sentiment(info)) 

    for title in titles:
        all_sentiments.append(sentiment(title))

    sentiment_counts = Counter(all_sentiments)
    majority_sentiment, count = sentiment_counts.most_common(1)[0]
    print(majority_sentiment)

