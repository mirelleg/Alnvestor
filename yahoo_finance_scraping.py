import time
import yfinance


def stock_ticker_information(stock_ticker):
    # stock_information = yfinance.Ticker(stock_ticker)

    # return stock_information.get_history_metadata()
    
    stock_information = yfinance.Ticker(stock_ticker)

    stock_news_articles = stock_information.news(count=100)
    stock_article_titles = []
    for article in stock_news_articles:
        article_title = article['title']
        stock_article_titles.append(article_title)

    return stock_article_titles

if __name__ == "__main__":
    ticker = input("Please enter a stock ticker: ")
    titles = stock_ticker_information(ticker)
    
    if titles:
        for title in titles:
            print(title)
    else:
        print("We could not find information on that stock. Please make sure it is a valid stock")