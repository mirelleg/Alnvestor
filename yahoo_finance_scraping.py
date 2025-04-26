import time
import yfinance


def stock_ticker_information(stock_ticker):
    stock_information = yfinance.Ticker(stock_ticker)

    stock_news_articles = stock_information.get_news(count=100)
    
    if not stock_news_articles:
        print("There was no news found on this stock")
        return []

    stock_article_titles = []
    stock_article_summaries = []
    stock_article_info = []
    url_info = []
    i=0
    for article in stock_news_articles:
        title = ""
        summary = ""
        url = ""
        try:
            url = article['content']['clickThroughUrl']['url']
        except Exception as e:
            url = "" # in the case where a url cant be found
         
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
        url_info.append(url)
        
        stock_article_info.append("URL: " + url + " ; TITLE: " + title + " ; SUMMARY: " + summary)

    first_5_url = url_info[:5]
    return stock_article_info, first_5_url
    

if __name__ == "__main__":
    ticker = input("Please enter a stock ticker: ")
    information, first_5_url = stock_ticker_information(ticker)
    print(first_5_url)
    
    
    if information:
        for info in information:
            information = info.split(' ; ')
            print(information[0].strip())
            print(information[1].strip())
            print(information[2])
            print("\n")
            #print(info, "\n")
    # else:
    #     print("We could not find information on that stock. Please make sure it is a valid stock")