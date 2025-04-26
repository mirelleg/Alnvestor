import yfinance
import plotly.express as px


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
    

if __name__ == "__main__":

    ticker = input("Please enter a stock ticker: ").upper()
    history = stock_ticker_history_for_llm(ticker)
    if history:
        print(history)
    else:
        print("We could not find information on that stock. Please make sure it is a valid stock")

    # history = stock_ticker_history_for_visualization(ticker)
    # visualize_stock_history(history, ticker)
