import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def visualize_stock_history(stock_history, stock_ticker):
    df = stock_history

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        specs=[[{"type": "xy"}],
               [{"type": "bar"}],
               [{"type": "xy"}]]
    )

    # --- Row 1: Candlestick + Close + MA5 ---
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='midnightblue')   # 深蓝
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA5'],
        mode='lines',
        name='MA5 (5-day Avg)',
        line=dict(color='mediumpurple', dash='dash')   # 紫色
    ), row=1, col=1)

    # --- Row 2: Volume ---
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name="Volume",
        marker_color='midnightblue'  # 浅蓝
    ), row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['RSI'],
        name="RSI",
        line=dict(color='midnightblue')
    ), row=3, col=1)

    # --- RSI Reference Lines ---
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=[70]*len(df),
        name="Overbought (70)",
        line=dict(color='red', dash='dash')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=[30]*len(df),
        name="Oversold (30)",
        line=dict(color='green', dash='dash')
    ), row=3, col=1)

    # --- Layout + Buttons ---
    fig.update_layout(
        height=950,
        title_text=f"{stock_ticker} Stock Overview (Last 90 Days)",
        template="plotly_white",
        font=dict(color="midnightblue"),   # 整体字体也变蓝
        legend_tracegroupgap=10,
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.2,
                buttons=list([
                    dict(
                        label="Show All",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, True, True]},
                              {"title": {"text": f"{stock_ticker} Stock Overview (Last 90 Days)"}}]
                    ),
                    dict(
                        label="Candle Only",
                        method="update",
                        args=[{"visible": [True, False, False, True, True, True, True]},
                              {"title": {"text": f"{stock_ticker} Stock Overview (Last 90 Days)"}}]
                    )
                ]),
                showactive=True
            )
        ]
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        tickformat="%Y-%m-%d",
        tickangle=45
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)

    fig.add_annotation(
        text="📌 Tip: Use 'Show All' or 'Candle Only' buttons above.",
        xref="paper", yref="paper",
        x=0, y=1.3,
        showarrow=False,
        font=dict(size=12, color="mediumpurple")  # 小提示也改紫色
    )

    return fig

def fetch_stock_data_for_llm_and_visualization(stock_ticker, period="90d"):
    stock = yf.Ticker(stock_ticker)
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


if __name__ == "__main__":

    ticker = input("Please enter a stock ticker: ").upper()
    stock_history, summary = fetch_stock_data_for_llm_and_visualization(ticker)
    visualize_stock_history(stock_history, ticker)
    if summary:
        print(summary)
    else:
        print("We could not find information on that stock. Please make sure it is a valid stock")

    # history = stock_ticker_history_for_visualization(ticker)
    # visualize_stock_history(history, ticker)
