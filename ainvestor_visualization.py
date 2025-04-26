import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance

# Streamlit UI for visualizing the stock history
stock_ticker = "AAPL"  # Example stock ticker

# Adding a title to the main page
st.image("ainvestor_logo.jpeg", width=150)
st.markdown("<h1 style='font-size: 36px; color: white; text-align: center;'>Welcome to AInvestor!</h1>", unsafe_allow_html=True)
#st.subtitle("Here's your detailed breakdown of " + str(stock_ticker))
st.markdown(f"<h2 style='font-size: 24px; color: white; text-align: center;'>Here's your detailed breakdown of {stock_ticker}</h2>", unsafe_allow_html=True)

# Creating a sidebar
st.sidebar.title("Welcome to AI Investor!")
st.sidebar.write("AI-driven stock market investment manager")

# Streamlit UI for visualizing the stock history
stock_ticker = "AAPL"  # Example stock ticker

# You can also add other widgets to the sidebar, for example:
stock_ticker = st.sidebar.text_input("Please enter the stock ticker you would like to learn more about:", placeholder="AAPL")
#st.write(f"Text input from sidebar: {sidebar_text}")

if stock_ticker == "":
    stock_ticker = "AAPL"

if stock_ticker:  # Check if the input is not empty
    stock = yfinance.Ticker(stock_ticker)  # Fetch the stock data for the entered ticker
    stock_data = stock.history(period="1mo")  # Get the stock data for the last month
    st.write(stock_data.tail())  # Display the stock data on the page

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


# # Sample history data for demonstration purposes (replace with actual data)
# history_dict = {
#     'history': pd.DataFrame({
#         'Date': pd.date_range(start="2023-01-01", periods=60, freq='D'),
#         'High': [100 + i for i in range(60)],
#         'Low': [90 + i for i in range(60)]
#     })
# }

history_dict = stock_ticker_history_for_visualization(stock_ticker)

# Function to visualize the stock history within a card
def visualize_stock_history(history_dict, stock_ticker):
    df = history_dict['history']

    # Create a line plot with Plotly
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

    return fig



# Create a card-like container with a collapsible section (expandable)

with st.expander("", expanded=True):
    # Custom large title inside the expander
    st.markdown(f"""
        <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <h2 style="text-align: center; color: black;">📈 View {stock_ticker} Stock History</h2>
            <p style="text-align: center; color: black;">Visualizing the high and low prices over the last 3 months</p>
        </div>
    """, unsafe_allow_html=True)

    # Call the function to generate the plot and display the figure
    fig = visualize_stock_history(history_dict, stock_ticker)
    
    # Display Plotly chart in Streamlit
    st.plotly_chart(fig)

# Function to get 5 urls from Yahoo Finance API
def top_urls(stock_ticker):
    stock_information = yfinance.Ticker(stock_ticker)
    stock_news_articles = stock_information.get_news(count=5)
    url_info = []
    for article in stock_news_articles:
        url = ""
        try:
            url = article['content']['clickThroughUrl']['url']
        except Exception as e:
            url = "" 
        url_info.append(url)
    return url_info

# A second card for the top 5 urls


# Function to get 5 urls from Yahoo Finance API
def top_urls_info(stock_ticker):
    stock_information = yfinance.Ticker(stock_ticker)
    stock_news_articles = stock_information.get_news(count=5)
    url_info = []
    url_titles = []
    url_summaries = []
    url_thumbnails = []
    for article in stock_news_articles:
        url = ""
        title = ""
        summary = ""
        thumbnail = ""
        try:
            url = article['content']['clickThroughUrl']['url']
        except Exception as e:
            url = "" 
        try:
            title = article['content']['title']
        except KeyError:
            title = "" # in the case where a title cant be found

        try:
            summary = article['content']['summary']
        except KeyError:
            summary = "" # in the case where a summary cant be found

        try:
            thumbnail = article['content']['thumbnail']  # Assuming 'thumbnail' is available
        except KeyError:
            thumbnail = ""  # in case the thumbnail is missing

        url_info.append(url)
        url_titles.append(title)
        url_summaries.append(summary)
        url_thumbnails.append(thumbnail)
    return url_info, url_titles, url_summaries, url_thumbnails

# A second card for the top 5 urls
top_5_urls, url_titles, url_summaries, url_thumbnails = top_urls_info(stock_ticker)

with st.expander("",expanded=True):
    # You can use st.markdown inside the expander to add a custom header with larger font
    st.markdown(
        f"<h1 style='font-size: 30px; font-weight: bold; color: #4CAF50;'>📰 Top 5 News Links for {stock_ticker}</h1>",
        unsafe_allow_html=True
    )
    st.markdown("""
        <style>
            .card-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
                padding: 20px;
            }
            .article-card {
                width: 100%;
                max-width: 700px;
                border-radius: 15px;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
                margin-bottom: 20px;
                overflow: hidden;
                background-color: white;
            }
            .card-image {
                width: 100%;
                height: 200px;
                background-size: cover;
                background-position: center;
            }
            .card-content {
                padding: 20px;
                background-color: white;
            }
            .card-title {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
                color: black;
            }
            .card-description {
                font-size: 16px;
                margin-bottom: 15px;
                color: black;
            }
            .card-button {
                display: inline-block;
                background-color: #1E88E5;  
                color: white;  
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
            }
        </style>
        <div class="card-container">
        """, unsafe_allow_html=True)

    # Create each card individually
    for i in range(5):
        url = top_5_urls[i]
        title = url_titles[i]
        description = url_summaries[i]
        thumbnail = url_thumbnails[i]['originalUrl']

        if not thumbnail:
            thumbnail = "https://via.placeholder.com/700x200.png?text=No+Image+Available"
        
        # Create HTML for a single card
        card_html = f"""
        <div class="article-card">
            <div class="card-image" style="background-image: url('{thumbnail}');"></div>
            <div class="card-content">
                <h3 class="card-title">{title}</h3>
                <p class="card-description">{description}</p>
                <a href="{url}" target="_blank" class="card-button">Go to Article</a>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)

    # Close the container
    st.markdown("</div>", unsafe_allow_html=True)



# Gemini AI Integration file recommendation code here

import torch
from transformers import pipeline
import requests
from datetime import datetime, timedelta
from collections import Counter
import praw
import google.generativeai as genai
import json
import re
import time

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

    fundamentals = {
        "P/E Ratio": pe_ratio,
        "EPS Growth (YoY)": eps_growth,
        "Revenue Growth (YoY)": revenue_growth,
        "Debt-to-Equity Ratio": debt_to_equity,
        "Cash Position ($)": cash,
        "Market Cap ($)": market_cap,
        "Industry": industry
    }

    fundamentals_summary = (
        f"{stock_ticker} currently trades at a P/E ratio of {pe_ratio}. "
        f"EPS growth YoY is {format_percentage(eps_growth)}, and revenue growth YoY is {format_percentage(revenue_growth)}. "
        f"The company maintains a cash position of ${format_large_number(cash)} and has a debt-to-equity ratio of {debt_to_equity}. "
        f"It operates in the {industry} industry."
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

def get_ai_recommendation(stock_ticker, progress_bar):
    
    def smooth_progress(start, end, duration=1.5):
        steps = int((end - start))
        sleep_time = duration / steps
        for p in range(start, end):
            progress_bar.progress(p)
            time.sleep(sleep_time)
        
    smooth_progress(0, 20, duration=1.5)
    recent_trading = stock_ticker_history_for_llm(stock_ticker)

    smooth_progress(20, 70, duration=5)
    sentiment_result = sentiment_for_llm(stock_ticker)
    
    fundamentals, fundamentals_summary = fetch_yhf_fundamentals_summary(stock_ticker)
    
    smooth_progress(70, 95, duration=2)
    suggestion = generate_investment_suggestion(stock_ticker, sentiment_result, recent_trading, fundamentals_summary)

    smooth_progress(95, 100, duration=0.5)

    if suggestion:
        investment_action = suggestion['investment_action']
        reason = suggestion['reason']
        return investment_action, reason
    else:
        return "❌ Failed to generate investment suggestion."

with st.expander("", expanded=True):
    st.markdown(f"""
         <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg" alt="Gemini Logo" style="width:60px;height:auto;margin-bottom:10px;">
            <h2 style="color: black;"> Gemini AI Recommendation for {stock_ticker}</h2>
        </div>
    """, unsafe_allow_html=True)

    recommendation_placeholder = st.empty()
    progress_bar = st.progress(0)

    recommendation_placeholder.markdown("""
        <div style="background-color: #f4f4f4; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: black;">Loading recommendation...</h4>
        </div>
    """, unsafe_allow_html=True)

    # Pass the progress bar into get_ai_recommendation
    investment_action, reason = get_ai_recommendation(stock_ticker, progress_bar)

    # Replace "Loading..." with formatted final output
    recommendation_placeholder.markdown(f"""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; text-align: center;">
            <h4 style="color: black;">Recommended Action: {investment_action}</h4>
            <p style="color: black;">Reason: {reason}</p>
        </div>
    """, unsafe_allow_html=True)

    progress_bar.empty()


