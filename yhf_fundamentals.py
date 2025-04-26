import yfinance as yf

def fetch_yhf_fundamentals_summary(stock_ticker):
    """
    Fetch key financial metrics from Yahoo Finance for a given stock ticker.
    
    Returns:
    - fundamentals: dict containing key metrics
    - fundamentals_summary: str describing financial highlights
    """
    stock = yf.Ticker(stock_ticker)
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
    ticker = input("Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA): ").upper()
    fundamentals, fundamentals_summary = fetch_yhf_fundamentals_summary(ticker)
    #print(fundamentals)
    #print(fundamentals_summary)