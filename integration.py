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
- Fundamenytals: {fundamentals}

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
def fetch_stock_data_for_llm_and_visualization(stock_ticker, period="90d"):
    stock = yfinance.Ticker(stock_ticker)
    stock_history = stock.history(period=period)

    if stock_history.empty or len(stock_history) < 5:
        return None, None, None

    stock_history = stock_history[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()

    
    info = stock.info

    # Calculate indicators
    stock_history['Range (%)'] = ((stock_history['High'] - stock_history['Low']) / stock_history['Low']) * 100
    stock_history['Volume Change (%)'] = stock_history['Volume'].pct_change() * 100
    stock_history['RSI'] = compute_rsi(stock_history['Close'])
    stock_history['MA5'] = stock_history['Close'].rolling(window=5).mean()
    stock_history['MA30'] = stock_history['Close'].rolling(window=30).mean()


    stock_history.to_csv(f'{stock_ticker}_visualization_data.csv', index=False)

    # --- Prepare summary ---
    recent = stock_history.tail(5)

    open_prices = "Open: " + " ".join(f"{p:.2f}" for p in recent['Open'])
    high_prices = "High: " + " ".join(f"{p:.2f}" for p in recent['High'])
    low_prices = "Low: " + " ".join(f"{p:.2f}" for p in recent['Low'])
    close_prices = "Close: " + " ".join(f"{p:.2f}" for p in recent['Close'])
    volume = "Volume: " + " ".join(f"{int(v)}" for v in recent['Volume'])

    stock_history_table = (
        "Here are the prices and volumes over the past 5 consecutive trading days:\n" +
        f"{open_prices}\n{high_prices}\n{low_prices}\n{close_prices}\n{volume}\n"
    )

    latest = recent.iloc[-1]

    summary = stock_history_table + (
        f"\nSummary for {stock_ticker}:\n"
        f"- Average daily High/Low range over past 5 days: {recent['Range (%)'].mean():.2f}%\n"
        f"- Latest trading day's volume: {int(latest['Volume'])}, "
        f"change from previous day: {latest['Volume Change (%)']:.2f}%\n"
        f"- Current MA5 (5-day moving average close): {latest['MA5']:.2f}\n"
        f"- Current MA30 (30-day moving average close): {latest['MA30']:.2f}\n"
        f"- Highest price over last 90 days: {stock_history['High'].max():.2f}\n"
        f"- Lowest price over last 90 days: {stock_history['Low'].min():.2f}\n"
        f"- Average volume over last 90 days: {int(stock_history['Volume'].mean())}\n"
        f"- Current RSI: {latest['RSI']:.2f}, indicating "
        f"{'overbought' if latest['RSI'] > 70 else 'oversold' if latest['RSI'] < 30 else 'neutral'} market conditions."
    )

    return stock_history, summary


def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

    

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

def fetch_yhf_fundamentals_summary(stock_ticker):
    """
    Fetch key financial metrics from Yahoo Finance for a given stock ticker.
    
    Returns:
    - fundamentals: dict containing key metrics
    - fundamentals_summary: str describing financial highlights
    """
    stock = yfinance.Ticker(stock_ticker)
    try:
        info = stock.info  # Try fetching stock info
    except Exception as e:
        print(f"Error fetching info for {stock_ticker}: {e}")
        return {}, "Failed to retrieve stock information."

    # Safe extraction with fallback
    pe_ratio = safe_get(info, 'trailingPE')
    eps_growth = safe_get(info, 'earningsQuarterlyGrowth')
    revenue_growth = safe_get(info, 'revenueGrowth')
    debt_to_equity = safe_get(info, 'debtToEquity')
    cash = safe_get(info, 'totalCash')
    market_cap = safe_get(info, 'marketCap')
    industry = safe_get(info, 'industry')

    dividend_yield = safe_get(info, 'dividendYield')
    return_on_equity = safe_get(info, 'returnOnEquity')
    profit_margin = safe_get(info, 'profitMargins')
    beta = safe_get(info, 'beta')

    fundamentals = {
        "P/E Ratio": pe_ratio,
        "EPS Growth (YoY)": eps_growth,
        "Revenue Growth (YoY)": revenue_growth,
        "Debt-to-Equity Ratio": debt_to_equity,
        "Cash Position ($)": cash,
        "Market Cap ($)": market_cap,
        "Industry": industry,
        "Dividend Yield": format_percentage(dividend_yield),
        "Return on Equity (ROE)": format_percentage(return_on_equity),
        "Profit Margin": format_percentage(profit_margin),
        "Beta (5Y Monthly)": beta
    }

    fundamentals_summary = (
        f"{stock_ticker} currently trades at a P/E ratio of {pe_ratio}. "
        f"EPS growth YoY is {format_percentage(eps_growth)}, and revenue growth YoY is {format_percentage(revenue_growth)}. "
        f"The company maintains a cash position of ${format_large_number(cash)} and has a debt-to-equity ratio of {debt_to_equity}. "
        f"It operates in the {industry} industry. "
        f" Dividend yield is {format_percentage(dividend_yield)} and profit margin is {format_percentage(profit_margin)}. "
        f"ROE stands at {format_percentage(return_on_equity)}, and the stock has a beta of {beta}."
    )

    return fundamentals, fundamentals_summary

def safe_get(dictionary, key):
    """
    Safely get a value from a dictionary. Return 'N/A' if not found.
    """
    value = dictionary.get(key, 'N/A')
    if value is None:
        return 'N/A'
    return value

def format_percentage(value):
    """
    Format a float into a percentage string. Return 'N/A' if invalid.
    """
    try:
        return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "N/A"
    except Exception:
        return "N/A"

def format_large_number(value):
    """
    Format large numbers into readable strings. Return 'N/A' if invalid.
    """
    try:
        if isinstance(value, (int, float)):
            if value >= 1e9:
                return f"{value / 1e9:.2f}B"
            elif value >= 1e6:
                return f"{value / 1e6:.2f}M"
            else:
                return f"{value:,.0f}"
        return "N/A"
    except Exception:
        return "N/A" 

if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ").strip().upper()
    stock_history, recent_trading = fetch_stock_data_for_llm_and_visualization(ticker)
    fundamentals, fundamentals_summary = fetch_yhf_fundamentals_summary(ticker)
    sentiment_result = sentiment_for_llm(ticker)
    suggestion = generate_investment_suggestion(ticker, sentiment_result, recent_trading, fundamentals_summary)
    
    if suggestion:
        print(json.dumps(suggestion, indent=4))
    else:
        print("❌ Failed to generate investment suggestion.")
    