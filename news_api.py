import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('API_KEY')

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


if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ")
    titles = fetch_news(ticker)
    
    if titles:
        for title in titles:
            print(title)