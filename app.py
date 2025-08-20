# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from forex_python.converter import CurrencyRates, RatesNotAvailableError
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set up the page config
st.set_page_config(page_title="AI Investment Predictor & SIP Calculator", layout="wide")

# App Title
st.title("Investment Predictor & SIP Calculator")

# Function: suggest some stocks if user doesn't enter any
def suggest_stocks():
    return {
        "AAPL": "Tech giant - Moderate Risk",
        "MSFT": "Stable growth - Low Risk",
        "TSLA": "High growth - High Risk",
        "INFY.NS": "Indian IT - Moderate Risk",
        "RELIANCE.NS": "Indian Conglomerate - Moderate Risk",
    }

# Input fields for users
currency = st.selectbox("Select your currency", ["USD", "INR", "EUR", "GBP", "JPY"])
sip_amount = st.number_input("Monthly SIP Amount", min_value=100, step=100, value=1000)
lump_sum = st.number_input("Lump Sum Investment Amount", min_value=0, step=100, value=0)
duration_years = st.number_input("Investment Duration (Years)", min_value=1, max_value=30, value=5)
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, INFY.NS)")

# If no stock ticker entered, show suggestions
if not stock_ticker:
    st.info("You did not enter any stock ticker. Here are some popular ones:")
    suggestions = suggest_stocks()
    for ticker, description in suggestions.items():
        st.markdown(f"- **{ticker}**: {description}")

# If stock ticker entered, fetch and process data
if stock_ticker:
    try:
        # Download historical price data for the ticker
        data = yf.download(stock_ticker, period=f"{duration_years}y", interval="1mo")
        if data.empty:
            st.error("Sorry, couldn't fetch data for this stock ticker. Please check the symbol.")
        else:
            prices = data['Adj Close'].dropna()

            st.subheader(f"Historical Adjusted Close Prices of {stock_ticker}")
            st.line_chart(prices)

            # Calculate monthly returns and annualized statistics
            returns = prices.pct_change().dropna()
            avg_return = returns.mean() * 12  # annualized average return
            risk = returns.std() * np.sqrt(12)  # annualized volatility (risk)

            # Simple price trend prediction using linear regression
            X = np.arange(len(prices)).reshape(-1, 1)
            y = prices.values
            model = LinearRegression().fit(X, y)
            predicted_price = model.predict(np.array([[len(prices) + duration_years * 12]]))[0]

            # Future value calculation for SIP and lump sum
            fv_sip = sip_amount * (((1 + avg_return)**duration_years - 1) / avg_return) * (1 + avg_return)
            fv_lump = lump_sum * ((1 + avg_return)**duration_years)
            total_fv = fv_sip + fv_lump

            # Convert USD amounts to selected currency
            c = CurrencyRates()
            if currency != 'USD':
                try:
                    conversion_rate = c.get_rate('USD', currency)
                    total_fv *= conversion_rate
                    predicted_price *= conversion_rate
                    sip_amount_converted = sip_amount * conversion_rate
                except RatesNotAvailableError:
                    st.warning("Currency conversion rate not available now. Showing values in USD.")
                    sip_amount_converted = sip_amount
            else:
                sip_amount_converted = sip_amount

            # Display the results
            st.markdown(f"### Predicted Results for {duration_years} Years")
            st.write(f"- Predicted Stock Price: {predicted_price:.2f} {currency}")
            st.write(f"- Estimated SIP Investment Value: {total_fv:.2f} {currency}")
            st.write(f"- Estimated Risk (Annual Volatility): {risk:.2%}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer note
st.markdown("---")
st.write("This is a basic demo using historical stock data and simple predictions. Future updates will add AI models, stock suggestions, and multi-currency support.")
