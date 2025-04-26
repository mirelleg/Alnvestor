import time
import yfinance


def stock_ticker_information(stock_ticker):
    # stock_information = yfinance.Ticker(stock_ticker)

    # return stock_information.get_history_metadata()
    
    stock_information = yfinance.Ticker(stock_ticker)

    stock_news_articles = stock_information.news
    for article in stock_news_articles:
        article_title = article['title']
    return stock_information.news


print(stock_ticker_information('AAPL'))