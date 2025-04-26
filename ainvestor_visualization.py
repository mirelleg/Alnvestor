# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import matplotlib.pyplot as plt

# # Function to fetch stock data
# def get_stock_data(ticker):
#     stock = yf.Ticker(ticker)
#     return stock.history(period="1mo")  # Get 1 month of data

# # AI Model Placeholder for stock suggestion
# def ai_investment_suggestion(stock_data):
#     # For simplicity, let's assume the model suggests based on price trend
#     last_price = stock_data['Close'].iloc[-1]
#     trend = "Buy" if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[0] else "Sell"
#     return trend, last_price

# # Streamlit UI
# def main():
#     st.title("AI Stock Investment Suggestions")

#     # User input for stock ticker
#     ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

#     # Fetch stock data
#     if ticker:
#         stock_data = get_stock_data(ticker)
#         st.write(f"Showing data for {ticker}")

#         # Show stock data table
#         st.write(stock_data.tail())

#         # Plot the closing prices
#         st.subheader(f"{ticker} - Stock Price Chart")
#         plt.figure(figsize=(10, 5))
#         plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
#         plt.title(f"{ticker} Stock Price")
#         plt.xlabel("Date")
#         plt.ylabel("Price (USD)")
#         plt.legend()
#         st.pyplot(plt)

#         # Get AI investment suggestion
#         suggestion, last_price = ai_investment_suggestion(stock_data)
#         st.subheader(f"AI Investment Suggestion for {ticker}")
#         st.write(f"Suggested Action: {suggestion}")
#         st.write(f"Current Price: ${last_price:.2f}")

    

#     # Sample data representing articles
#     articles = [
#         {"title": "Stock Market Analysis: April 2025", "url": "https://example.com/article1", "summary": "A detailed analysis of the stock market trends for April 2025."},
#         {"title": "AI in Finance: Revolutionizing Investment", "url": "https://example.com/article2", "summary": "How artificial intelligence is changing the world of investment and trading."},
#         {"title": "Top Tech Stocks to Watch", "url": "https://example.com/article3", "summary": "A list of top technology stocks that are expected to perform well in 2025."},
#     ]

#     # Streamlit UI
#     st.title("Relevant Articles")

#     # Display cards using columns
#     for article in articles:
#         col1, col2 = st.columns([1, 3])  # Split layout with two columns
#         with col1:
#             # Optionally, you can add an image here for each article (just as an example)
#             st.image("https://via.placeholder.com/150", width=150)  # Placeholder image (replace with actual image)
#         with col2:
#             # Create a "card" style content block
#             st.markdown(f"""
#             <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; box-shadow: 0px 0px 5px rgba(0,0,0,0.1);">
#                 <h4 style="color: #0056b3;">{article['title']}</h4>
#                 <p>{article['summary']}</p>
#                 <a href="{article['url']}" target="_blank" style="color: #007bff; text-decoration: none;">Read more...</a>
#             </div>
#             """, unsafe_allow_html=True)



# if __name__ == "__main__":
#     main()

# import streamlit as st
# import time

# # Set up session state for page navigation
# if 'page' not in st.session_state:
#     st.session_state.page = 'welcome'

# # Welcome Page with Animated Background
# if st.session_state.page == 'welcome':
#     st.markdown("""
#         <style>
#         body {
#             background: linear-gradient(45deg, #ff6b6b, #ffcb6b, #6bffcb, #6b6bff);
#             background-size: 400% 400%;
#             animation: gradientAnimation 10s ease infinite;
#         }
        
#         @keyframes gradientAnimation {
#             0% { background-position: 0% 50%; }
#             50% { background-position: 100% 50%; }
#             100% { background-position: 0% 50%; }
#         }
        
#         .title {
#             text-align: center;
#             font-size: 50px;
#             color: white;
#             font-weight: bold;
#             margin-top: 200px;
#         }
        
#         .description {
#             text-align: center;
#             font-size: 20px;
#             color: white;
#             margin-top: 20px;
#         }

#         .button-container {
#             display: flex;
#             justify-content: center;
#             margin-top: 30px;
#         }

#         .start-button {
#             background-color: #4CAF50;
#             color: white;
#             font-size: 20px;
#             padding: 10px 20px;
#             border-radius: 10px;
#             border: none;
#             cursor: pointer;
#             transition: background-color 0.3s;
#         }

#         .start-button:hover {
#             background-color: #45a049;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown('<div class="title">Welcome to AI Stock Investor!</div>', unsafe_allow_html=True)
#     st.markdown('<div class="description">Get personalized stock recommendations and track your portfolio performance with AI-driven insights.</div>', unsafe_allow_html=True)

#     # Add the "Let's Get Started" button
#     if st.button("Let's Get Started"):
#         st.session_state.page = 'dashboard'

# # Dashboard Page
# if st.session_state.page == 'dashboard':
#     st.title("AI Stock Investor Dashboard")
#     st.write("Welcome to the dashboard. Here's where you can track your stocks, portfolio, and get AI-based recommendations.")
    
#     # Simulate some content for the dashboard
#     st.write("More content goes here!")
#     st.write("Add charts, stock analysis, AI predictions, and more!")

#     # Example of a simple loading spinner
#     with st.spinner("Loading dashboard..."):
#         time.sleep(2)  # Simulate loading delay
#     st.success("Dashboard loaded!")


# import streamlit as st
# import plotly.graph_objs as go
# import yfinance as yf
# import time

# # Set up session state for page navigation
# if 'page' not in st.session_state:
#     st.session_state.page = 'welcome'

# # Get live stock data (example: S&P 500 index)
# stock_ticker = yf.Ticker('^GSPC')
# stock_history = stock_ticker.history(period="1mo")

# # Create a stock chart using Plotly
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=stock_history.index, y=stock_history['Close'],
#                          mode='lines', name='S&P 500',
#                          line=dict(color='white', width=2)))

# fig.update_layout(title='S&P 500 Stock Price (Last 1 Month)',
#                   xaxis_title='Date', yaxis_title='Price',
#                   plot_bgcolor='rgba(0,0,0,0)',
#                   paper_bgcolor='rgba(0,0,0,0)',
#                   font=dict(color='white'))

# # Welcome Page with Finance Chart as Background
# if st.session_state.page == 'welcome':
#     st.markdown("""
#         <style>
#         body {
#             background-color: #2e3b4e;
#             color: white;
#             height: 100vh;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#         }

#         .title {
#             text-align: center;
#             font-size: 50px;
#             color: white;
#             font-weight: bold;
#             margin-bottom: 20px;
#         }

#         .description {
#             text-align: center;
#             font-size: 20px;
#             color: white;
#             margin-bottom: 30px;
#         }

#         .button-container {
#             display: flex;
#             justify-content: center;
#             margin-top: 30px;
#         }

#         .start-button {
#             background-color: #4CAF50;
#             color: white;
#             font-size: 20px;
#             padding: 10px 20px;
#             border-radius: 10px;
#             border: none;
#             cursor: pointer;
#             transition: background-color 0.3s;
#         }

#         .start-button:hover {
#             background-color: #45a049;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     st.markdown('<div class="title">Welcome to AI Stock Investor!</div>', unsafe_allow_html=True)
#     st.markdown('<div class="description">Get personalized stock recommendations and track your portfolio performance with AI-driven insights.</div>', unsafe_allow_html=True)

#     # Add the "Let's Get Started" button
#     if st.button("Let's Get Started"):
#         st.session_state.page = 'dashboard'

# # Dashboard Page
# if st.session_state.page == 'dashboard':
#     st.title("AI Stock Investor Dashboard")
#     st.write("Welcome to the dashboard. Here's where you can track your stocks, portfolio, and get AI-based recommendations.")
    
#     # Display the animated finance chart
#     st.plotly_chart(fig)

#     # Simulate some content for the dashboard
#     st.write("More content goes here!")
#     st.write("Add charts, stock analysis, AI predictions, and more!")

#     # Example of a simple loading spinner
#     with st.spinner("Loading dashboard..."):
#         time.sleep(2)  # Simulate loading delay
#     st.success("Dashboard loaded!")


import streamlit as st
import time

# Set up session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Welcome Page with Animated Finance Graph Background
if st.session_state.page == 'welcome':
    st.markdown("""
        <style>
        /* Style the page and make the background fill the screen */
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #111;
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        /* Create an animated background with a moving finance graph */
        .animated-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }

        /* Create a moving line that simulates a graph */
        .graph-line {
            position: absolute;
            top: 50%;
            left: -100%;
            width: 200%;
            height: 2px;
            background: linear-gradient(to right, #4CAF50, #FF5733);
            animation: moveLine 5s ease-in-out infinite;
        }

        /* Create the movement of the line graph */
        @keyframes moveLine {
            0% { left: -100%; }
            50% { left: 50%; }
            100% { left: 100%; }
        }

        /* Style for the welcome text */
        .title {
            font-size: 50px;
            font-weight: bold;
            color: white;
            text-align: center;
            z-index: 1;
        }

        .description {
            font-size: 20px;
            color: white;
            text-align: center;
            z-index: 1;
            margin-top: 20px;
        }

        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }

        .start-button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .start-button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add the animated graph line to the background
    st.markdown('<div class="animated-background"><div class="graph-line"></div></div>', unsafe_allow_html=True)

    # Add title and description
    st.markdown('<div class="title">Welcome to AI Stock Investor!</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Get personalized stock recommendations and track your portfolio performance with AI-driven insights.</div>', unsafe_allow_html=True)

    # Add "Let's Get Started" button
    if st.button("Let's Get Started"):
        st.session_state.page = 'dashboard'

# Dashboard Page
if st.session_state.page == 'dashboard':
    st.title("AI Stock Investor Dashboard")
    st.write("Welcome to the dashboard. Here's where you can track your stocks, portfolio, and get AI-based recommendations.")
    
    # Simulate some content for the dashboard
    st.write("More content goes here!")
    st.write("Add charts, stock analysis, AI predictions, and more!")

    # Example of a simple loading spinner
    with st.spinner("Loading dashboard..."):
        time.sleep(2)  # Simulate loading delay
    st.success("Dashboard loaded!")
