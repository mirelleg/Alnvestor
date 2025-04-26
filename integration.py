import yfinance
import plotly.express as px
import torch
from transformers import pipeline
import requests
from datetime import datetime, timedelta
from collections import Counter
import yfinance
import praw
import google.generativeai as genai
import json
import re

def extract_json_from_response(response_text):
    """
    Robust extraction: handles ```json ... ``` wrappers, leading/trailing spaces, line breaks.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        json_text = match.group(0)
        return json.loads(json_text)
    else:
        raise ValueError("No valid JSON object found in response.")

gemini_api_key = "AIzaSyADBstQ9m7ZX-JOCSJBKy82S_24FIy0StQ"

def generate_investment_suggestion(stock_ticker, sentiment_result, recent_trading, fundamentals):
    """
    Generate investment suggestion using Gemini model based on sentiment and financial data.
    """
    prompt = f"""
You are a financial investment assistant.

Input:
- Stock Ticker: {stock_ticker}
- Sentiment Summary from media: {sentiment_result}
- Recent Trading Summary: {recent_trading}
- Fundamentals: {fundamentals}

Task:
Based on the information you are given, make an investment suggestion for the given stock. 
Choose one action from ["Buy", "Sell", "Hold"] and provide a professional, well-reasoned explanation supporting your recommendation.
Write like a professional financial analyst. Keep the explanation concise but substantial, using concrete observations from the data provided. Avoid overly brief conclusions.

Format your output strictly in JSON as:
{{
  "investment_action": "Buy" or "Sell" or "Hold",
  "reason": your explanation here
}}
    """
    genai.configure(api_key = gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    try:
        result = extract_json_from_response(response.text)
        return result
    except Exception as e:
        print("Failed to parse output as JSON:", e)
        print("Raw response:", response.text)
        return None


finbert_api_key = '4867727998754629ad10c19521b7015e'

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
        'pageSize': 20,              
        'apiKey': finbert_api_key            
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

    stock_news_articles = stock_information.get_news(count=15)
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

# Function to fetch Reddit posts related to a stock keyword
def fetch_reddit_posts(keyword):
    reddit = praw.Reddit(
        client_id="hXa0VNF87XwdKlJEcwCuLA",
        client_secret="hJaEUA7bJnYw7FgY-Nr1gWzKuWDFMg",
        user_agent="AInvestor Hackathon project for LAHack"
    )

    subreddits = [
        "wallstreetbets", "Superstonk", "options", "stocks", "StockMarket"
    ]

    posts = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(query=keyword, sort="new", time_filter="month", limit=20):
                if len(post.selftext) > 50 and post.score > 10:
                    posts.append(f"{post.title} {post.selftext}")

                if len(posts) >= 10:
                    break
        except Exception as e:
            print(f"Error accessing subreddit {sub_name}: {e}")

        if len(posts) >= 100:
            break

    return posts

def sentiment(text):
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)

    chunk_size = 450

    if len(text) <= chunk_size:
        result = pipe(text, truncation=True)[0]['label']
        return result
    
    sentiments = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        sentiment_result = pipe(chunk, truncation=True)[0]['label']
        sentiments.append(sentiment_result)
    
    sentiment_counts = Counter(sentiments)
    majority_sentiment, _ = sentiment_counts.most_common(1)[0]

    return majority_sentiment


def stock_ticker_history_for_visualization(stock_ticker):
    stock_ticker = yfinance.Ticker(stock_ticker)

    stock_history = stock_ticker.history(period="3mo")
    stock_history = stock_history[['Open', 'High', 'Low', 'Close']].reset_index()
    dividends = stock_ticker.dividends
    info = stock_ticker.info
    market_cap = info.get('marketCap', None)
    p_e_ratio = info.get('trailingPE', None)

    stock_info_dict = {
        "history": stock_history,
        "dividends": dividends,
        "market cap": market_cap,
        "P/E ratio": p_e_ratio
    }

    stock_history['Market Cap'] = market_cap
    stock_history['P/E Ratio'] = p_e_ratio

    stock_history['Dividends'] = stock_history['Date'].map(dividends) # NaN for non existing values

    stock_history.to_csv(f'{stock_ticker}_visualization_data.csv')

    return stock_info_dict

def visualize_stock_history(history_dict, stock_ticker):
    df = history_dict['history']

    fig = px.line(
        df,
        x="Date",
        y=["High", "Low"],
        title=f"{stock_ticker} High and Low Prices Over Last 3 Months",
        labels={"value": "Price (USD)", "Date": "Date", "variable": "Price Type"}
    )

    fig.update_layout(
        title_font_size=24,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend_title="Price Type",
        template="plotly_white",
        hovermode="x unified"    
    )

    fig.show()


def stock_ticker_history_for_llm(stock_ticker):
    stock_ticker = yfinance.Ticker(stock_ticker)

    stock_history = stock_ticker.history(period="5d")
    stock_history = stock_history[['Open', 'High', 'Low', 'Close']]
    dividends = stock_ticker.dividends
    info = stock_ticker.info
    market_cap = info.get('marketCap', None)
    p_e_ratio = info.get('trailingPE', None)

    # stock_info_dict = {
    #     "history": stock_history,
    #     "dividends": dividends,
    #     "market cap": market_cap,
    #     "P/E ratio": p_e_ratio
    # }

    stock_history['Market Cap'] = market_cap
    stock_history['P/E Ratio'] = p_e_ratio

    stock_history['Dividends'] = stock_history.index.to_series().map(dividends) # NaN for non existing values

    stock_history.to_csv(f'{stock_ticker}_for_llm.csv')

    # stock_history_table = []
    stock_history_table = ""
    stock_history_table += "Here are the prices across the past 5 days: " + "\n"
    open_prices = "Open: "
    for o in stock_history['Open']:
        open_prices += str(o) + " "
    high_prices = "High: "
    for h in stock_history['High']:
        high_prices += str(h) + " "
    low_prices = "Low: "
    for l in stock_history['Low']:
        low_prices += str(l) + " "
    close_prices = "Close: "
    for c in stock_history['Close']:
        close_prices += str(c) + " "
    
    stock_history_table += open_prices + "\n" + high_prices + "\n" + low_prices + "\n" + close_prices

    return stock_history_table

def sentiment_for_llm(ticker):
    
    titles = fetch_news(ticker)
    information = stock_ticker_information(ticker)
    posts = fetch_reddit_posts(ticker)

    all_sentiments = []
    for info in information:
        all_sentiments.append(sentiment(info)) 

    for title in titles:
        all_sentiments.append(sentiment(title))

    for post in posts:
        all_sentiments.append(sentiment(post))

    sentiment_counts = Counter(all_sentiments)
    majority_sentiment, count = sentiment_counts.most_common(1)[0]
    return majority_sentiment


if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ").strip().upper()
    recent_trading = stock_ticker_history_for_llm(ticker)
    sentiment_result = sentiment_for_llm(ticker)
    fundamentals = ""
    suggestion = generate_investment_suggestion(ticker, sentiment_result, recent_trading, fundamentals)
    
    if suggestion:
        print(json.dumps(suggestion, indent=4))
    else:
        print("❌ Failed to generate investment suggestion.")
    