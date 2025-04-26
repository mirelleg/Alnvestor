import time
import yfinance


def stock_ticker_information(stock_ticker):
    stock_information = yfinance.Ticker(stock_ticker)

    stock_news_articles = stock_information.get_news(count=100)
    stock_article_titles = []
    stock_article_summaries = []
    stock_article_info = []
    for article in stock_news_articles:
        title = article['content']['title']
        summary = article['content']['summary']
        
        stock_article_titles.append(title)
        stock_article_summaries.append(summary)
        stock_article_info.append({"TITLE: " + title + " ; SUMMARY: " + summary})

    return stock_article_info
    

if __name__ == "__main__":
    ticker = input("Please enter a stock ticker: ")
    information = stock_ticker_information(ticker)
    
    if information:
        for info in information:
            print(info, "\n")
    else:
        print("We could not find information on that stock. Please make sure it is a valid stock")