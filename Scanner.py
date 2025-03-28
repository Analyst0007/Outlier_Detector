# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 08:09:37 2025

@author: Hemal
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("Outlier Detector")
st.markdown("""
Welcome to the Outlier Detector application! This app allows you to analyze stock returns, check for normality, and identify outliers.
""")

# User inputs
st.sidebar.header("User Inputs")
symbols_input = st.sidebar.text_input("Enter Stock Symbols (comma-separated)", "^NSEI,HDFCBANK.NS")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
std_dev_multiplier = st.sidebar.slider("Select Standard Deviation", 1, 2, 3)

# Convert symbols input to list
symbols = [symbol.strip() for symbol in symbols_input.split(",")]

# Fetch stock data
@st.cache_data
def fetch_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data

data = fetch_data(symbols, start_date, end_date)

# Calculate returns
daily_returns = data.pct_change().dropna() * 100
weekly_returns = data.resample('W').ffill().pct_change().dropna() * 100
monthly_returns = data.resample('M').ffill().pct_change().dropna() * 100
yearly_returns = data.resample('Y').ffill().pct_change().dropna() * 100

# Function to check normality and find outliers
def check_normality_and_outliers(returns, period, std_dev_multiplier):
    normality_test = {}
    outliers = {}
    for symbol in returns.columns:
        # Normality test using Anderson-Darling
        result = stats.anderson(returns[symbol].dropna(), dist='norm')
        normality_test[symbol] = result.statistic < result.critical_values[2]  # Null hypothesis: data is normally distributed

        # Identify outliers
        mean = returns[symbol].mean()
        std_dev = returns[symbol].std()
        outliers[symbol] = returns[symbol][(returns[symbol] > mean + std_dev_multiplier*std_dev) | (returns[symbol] < mean - std_dev_multiplier*std_dev)]

    # Create DataFrame for results
    normality_df = pd.DataFrame.from_dict(normality_test, orient='index', columns=['Normality Test'])
    outliers_df = pd.DataFrame.from_dict(outliers, orient='index').transpose()

    return normality_df, outliers_df

# Check normality and find outliers for each period
daily_normality, daily_outliers = check_normality_and_outliers(daily_returns, 'daily', std_dev_multiplier)
weekly_normality, weekly_outliers = check_normality_and_outliers(weekly_returns, 'weekly', std_dev_multiplier)
monthly_normality, monthly_outliers = check_normality_and_outliers(monthly_returns, 'monthly', std_dev_multiplier)
yearly_normality, yearly_outliers = check_normality_and_outliers(yearly_returns, 'yearly', std_dev_multiplier)

# Display results
st.header("Normality Test Results")
st.subheader("Daily")
st.dataframe(daily_normality)
st.subheader("Weekly")
st.dataframe(weekly_normality)
st.subheader("Monthly")
st.dataframe(monthly_normality)
st.subheader("Yearly")
st.dataframe(yearly_normality)

st.header("Outliers")
st.subheader("Daily")
st.dataframe(daily_outliers)
st.subheader("Weekly")
st.dataframe(weekly_outliers)
st.subheader("Monthly")
st.dataframe(monthly_outliers)
st.subheader("Yearly")
st.dataframe(yearly_outliers)

# Visualization
st.header("Outlier Returns vs Mean Returns")
for symbol in symbols:
    st.subheader(f"{symbol} - Daily Returns")
    plt.figure(figsize=(10, 6))
    plt.scatter(daily_returns.index, daily_returns[symbol], label='Returns')
    plt.axhline(y=daily_returns[symbol].mean(), color='r', linestyle='--', label='Mean Return')
    plt.axhline(y=daily_returns[symbol].mean() + std_dev_multiplier*daily_returns[symbol].std(), color='g', linestyle='--', label=f'{std_dev_multiplier} Std Dev')
    plt.axhline(y=daily_returns[symbol].mean() - std_dev_multiplier*daily_returns[symbol].std(), color='g', linestyle='--')
    plt.title(f"{symbol} Daily Returns with Outliers")
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.legend()
    st.pyplot(plt)
