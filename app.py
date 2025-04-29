import time
import streamlit as st
import yfinance as yf
from functools import wraps
import praw
import plotly.graph_objects as go
import pandas as pd

# Rate limit manager
class RateLimiter:
    def __init__(self, calls_per_hour=1800):
        self.calls_per_hour = calls_per_hour
        self.last_call = 0
        self.interval = 3600 / calls_per_hour

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()
            return func(*args, **kwargs)
        return wrapper

rate_limiter = RateLimiter()

@st.cache_data(ttl=3600)
@rate_limiter
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    for attempt in range(6):
        try:
            info = stock.info
            if info and isinstance(info, dict) and "symbol" in info:
                return info
            time.sleep(10)
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                delay = 10 * (2 ** attempt)
                st.warning(f"Rate limit hit for {ticker}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                st.warning(f"Error fetching {ticker} info: {e}")
                return None
    st.warning(f"Failed to fetch {ticker} info after 6 attempts.")
    return None

@st.cache_data(ttl=3600)
@rate_limiter
def get_stock_history(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    for attempt in range(6):
        try:
            hist = stock.history(period=period)
            if not hist.empty:
                return hist
            time.sleep(10)
        except Exception as e:
            if "429" in str(e) or "Expecting value" in str(e):
                delay = 10 * (2 ** attempt)
                st.warning(f"Rate limit hit for {ticker} history, retrying in {delay}s...")
                time.sleep(delay)
            else:
                st.warning(f"Error fetching {ticker} history: {e}")
                return None
    st.warning(f"Failed to fetch {ticker} history after 6 attempts.")
    return None

@st.cache_data(ttl=3600)
@rate_limiter
def get_portfolio_info(tickers):
    results = {}
    for ticker in tickers.split():
        info = get_stock_info(ticker)
        results[ticker] = info if info else {}
    return results

# Reddit setup (use your credentials)
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="StockApp/1.0"
)
# Streamlit app setup
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer")

# CSS for styling
st.markdown("""
<style>
    .company-info { font-size: 18px; margin-bottom: 10px; }
    .description { font-size: 16px; color: #555; }
    .metric-value { font-size: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Company info
st.subheader("🔍 Company Information")
stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL").strip().upper()

def market_cap_display(market_cap):
    if isinstance(market_cap, (int, float)):
        if market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        else:
            return f"${market_cap / 1_000_000:.2f}M"
    return "N/A"

if stock_ticker:
    with st.spinner("Fetching company info..."):
        stock_info = get_stock_info(stock_ticker)
        if stock_info:
            company_name = stock_info.get("longName", "N/A")
            sector = stock_info.get("sector", "N/A")
            market_cap = stock_info.get("marketCap")
            summary = stock_info.get("longBusinessSummary", "No description available.")
            st.markdown(
                f'<p class="company-info"><strong>Company:</strong> {company_name} | '
                f'<strong>Sector:</strong> {sector} | '
                f'<strong>Market Cap:</strong> {market_cap_display(market_cap)}</p>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p class="description">{summary}</p>', unsafe_allow_html=True)
        else:
            st.warning(f"Could not fetch company info for {stock_ticker}. Please try again later.")

# Key metrics
st.subheader("📊 Key Metrics")
if stock_ticker:
    with st.spinner("Fetching key metrics..."):
        stock_info = get_stock_info(stock_ticker)  # Reuses cached data
        if stock_info:
            pe_ratio = stock_info.get("trailingPE", "N/A")
            pb_ratio = stock_info.get("priceToBook", "N/A")
            pe_display = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
            pb_display = f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio
            st.markdown(
                f'<p class="metric-value">P/E Ratio: {pe_display} | P/B Ratio: {pb_display}</p>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Could not fetch key metrics.")

# Price history
st.subheader("📈 Price History")
if stock_ticker:
    with st.spinner("Fetching price history..."):
        hist = get_stock_history(stock_ticker, period="1y")
        if hist is not None and not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price"))
            fig.update_layout(title=f"{stock_ticker} Price (1 Year)", xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price history for {stock_ticker}.")

# Technical indicators (RSI)
st.subheader("📉 Technical Indicators")
if stock_ticker:
    with st.spinner("Calculating RSI..."):
        hist = get_stock_history(stock_ticker, period="1y")  # Reuses cached data
        if hist is not None and not hist.empty:
            # Calculate 14-day RSI
            delta = hist["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1] if not rsi.empty else None

            if latest_rsi is not None and not pd.isna(latest_rsi):
                rsi_display = f"{latest_rsi:.2f}"
                if latest_rsi > 70:
                    rsi_status = "Overbought (potential sell signal)"
                elif latest_rsi < 30:
                    rsi_status = "Oversold (potential buy signal)"
                else:
                    rsi_status = "Neutral"
                st.markdown(
                    f'<p class="metric-value">14-Day RSI: {rsi_display} ({rsi_status})</p>',
                    unsafe_allow_html=True
                )
                st.markdown("**RSI Explanation**: Relative Strength Index (0–100) measures momentum. Above 70 indicates overbought; below 30 indicates oversold.")
            else:
                st.warning("Could not calculate RSI (insufficient data).")
        else:
            st.warning("Could not fetch price history for RSI calculation.")
# Portfolio Section
with st.container():
    st.subheader("💼 Portfolio Analysis")
    portfolio_input = st.text_input(
        "Enter up to 2 tickers (comma-separated, e.g., AAPL,MSFT):",
        value="AAPL",
        key="portfolio_tickers"
    )
    shares_input = st.text_input(
        "Enter shares for each ticker (comma-separated, same order):",
        value="100",
        key="portfolio_shares"
    )
    portfolio_tickers = [ticker.strip().upper() for ticker in portfolio_input.split(",")][:2]
    if len(portfolio_tickers) > 2:
        st.warning("Limited to 2 tickers to ensure performance. Please enter fewer tickers.")
    shares_list = [float(x.strip()) for x in shares_input.split(",")] if shares_input else [100] * len(portfolio_tickers)

    st.markdown("### What-If Scenarios")
    with st.expander("Adjust Portfolio"):
        what_if_shares = {}
        for i, ticker in enumerate(portfolio_tickers):
            default_shares = shares_list[i] if i < len(shares_list) else 100
            what_if_shares[ticker] = st.number_input(
                f"Shares for {ticker}", min_value=0.0, value=default_shares, step=1.0, key=f"what_if_{ticker}"
            )

    portfolio_data = {}
    sector_data = {}
    total_value = 0
    what_if_total_value = 0
    undervalued_count = 0

    if portfolio_tickers:
        with st.spinner("Fetching portfolio data..."):
            tickers_str = " ".join(portfolio_tickers)
            portfolio_info = get_portfolio_info(tickers_str)
            for i, ticker in enumerate(portfolio_tickers):
                info = portfolio_info.get(ticker, {})
                if not info:
                    st.warning(f"Could not fetch data for {ticker}.")
                    continue

                price = info.get("regularMarketPrice", info.get("regularMarketPreviousClose", 0))
                shares = shares_list[i] if i < len(shares_list) else 100
                what_if_share = what_if_shares.get(ticker, shares)
                value = price * shares
                what_if_value = price * what_if_share
                total_value += value
                what_if_total_value += what_if_value

                eps = info.get("trailingEps")
                intrinsic_value = eps * 15 if eps else None
                is_undervalued = intrinsic_value and price < intrinsic_value * 0.9
                if is_undervalued:
                    undervalued_count += 1

                sector = info.get("sector", "Unknown")
                sector_data[sector] = sector_data.get(sector, 0) + value

                portfolio_data[ticker] = {
                    "Price": price,
                    "Shares": shares,
                    "Value": value,
                    "WhatIfShares": what_if_share,
                    "WhatIfValue": what_if_value,
                    "Undervalued": is_undervalued,
                    "Intrinsic": intrinsic_value
                }

            if portfolio_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Total Portfolio Value**: ${total_value:.2f}")
                with col2:
                    st.markdown(f"**What-If Total Value**: ${what_if_total_value:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    fig_pie = go.Figure(data=[go.Pie(labels=list(portfolio_data.keys()), values=[d["Value"] for d in portfolio_data.values()])])
                    fig_pie.update_layout(title="Portfolio Allocation")
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    fig_what_if_pie = go.Figure(data=[go.Pie(labels=list(portfolio_data.keys()), values=[d["WhatIfValue"] for d in portfolio_data.values()])])
                    fig_what_if_pie.update_layout(title="What-If Portfolio Allocation")
                    st.plotly_chart(fig_what_if_pie, use_container_width=True)

                st.markdown("### Undervaluation Flags")
                for ticker, data in portfolio_data.items():
                    if data["Undervalued"]:
                        st.markdown(f"🔼 {ticker}: Undervalued (Price: ${data['Price']:.2f} vs. Intrinsic: ${data['Intrinsic']:.2f})")
                    else:
                        st.markdown(f"🔄 {ticker}: Fairly valued or overvalued")

                st.markdown("### Sector Breakdown")
                fig_sector_pie = go.Figure(data=[go.Pie(labels=list(sector_data.keys()), values=list(sector_data.values()))])
                fig_sector_pie.update_layout(title="Portfolio by Sector")
                st.plotly_chart(fig_sector_pie, use_container_width=True)
# Reddit Sentiment Section
with st.container():
    st.subheader("🗣️ Social Sentiment Analysis")
    if stock_ticker:
        with st.spinner("Fetching Reddit sentiment..."):
            try:
                subreddit = reddit.subreddit("stocks")
                posts = subreddit.search(stock_ticker, sort="new", limit=5)
                positive_words = ["bullish", "buy", "great", "strong", "up", "profit", "growth"]
                negative_words = ["bearish", "sell", "bad", "weak", "down", "loss", "crash"]
                positive_count, negative_count = 0, 0
                for post in posts:
                    text = post.title.lower() + " " + post.selftext.lower()
                    for word in positive_words:
                        positive_count += text.count(word)
                    for word in negative_words:
                        negative_count += text.count(word)
                sentiment_score = positive_count - negative_count
                sentiment_label = "Bullish" if sentiment_score > 0 else "Bearish" if sentiment_score < 0 else "Neutral"
                st.markdown(f"**Sentiment for {stock_ticker}**: {sentiment_score} ({sentiment_label})")
            except Exception as e:
                st.warning(f"Could not fetch Reddit sentiment: {e}")
