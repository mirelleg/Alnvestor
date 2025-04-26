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
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    try:
        result = extract_json_from_response(response.text)
        return result
    except Exception as e:
        print("Failed to parse output as JSON:", e)
        print("Raw response:", response.text)
        return None

def main():
    # Configure Gemini API
    genai.configure(api_key="AIzaSyADBstQ9m7ZX-JOCSJBKy82S_24FIy0StQ")

    # Example input
    stock_ticker = "AAPL"
    sentiment_result = "Overall positive sentiment from Reddit, Twitter, and financial news."
    recent_trading = "Over the past 5 trading days, AAPL’s stock price increased from $175.50 to $177.00 (+0.85%). Average daily trading volume remained stable around 80M shares."
    fundamentals = "Current P/E ratio is 28x. Revenue growth has slowed to 2% year-over-year. Strong balance sheet with $50B in cash reserves."

    suggestion = generate_investment_suggestion(stock_ticker, sentiment_result, recent_trading, fundamentals)
    
    if suggestion:
        print(json.dumps(suggestion, indent=4))
    else:
        print("❌ Failed to generate investment suggestion.")

if __name__ == "__main__":
    main()