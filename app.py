import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import time

# Initialize session state for caching and theme
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "theme_styles" not in st.session_state:
    st.session_state.theme_styles = {
        "bg": "#FFFFFF",
        "text": "#1F2937",
        "plot_bg": "rgba(240, 240, 240, 0.5)",
        "grid": "rgba(200, 200, 200, 0.5)",
        "line": "#00A8E8",
        "sma20": "#FF6F61",
        "sma50": "#6B7280",
        "sma200": "#34D399",
        "bar_colors": ["#00C4B4", "#FF6F61", "#F4A261", "#34D399", "#6B7280", "#A78BFA", "#EC4899", "#EF4444"],
        "calc_header": "#00A8E8"
    }

# Reddit API setup (replace with your credentials)
reddit = praw.Reddit(
    client_id="V3rxmA_qYIzBNTNW79LWIg",
    client_secret="xKBA3Nx7f7VQS0fXnOgmJhYZOmGasA",
    user_agent="python:EquityScopeBot:v1.0"
)

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Cache stock info
def get_stock_info(ticker):
    if ticker not in st.session_state.cache:
        try:
            stock = yf.Ticker(ticker)
            st.session_state.cache[ticker] = stock.info
        except:
            return None
    return st.session_state.cache[ticker]

# Cache stock history
def get_stock_history(ticker, period):
    cache_key = f"{ticker}_{period}"
    if cache_key not in st.session_state.cache:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            st.session_state.cache[cache_key] = hist
        except:
            return None
    return st.session_state.cache[cache_key]

# Company Info and Metrics
st.set_page_config(
    page_title="EquityScope: Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)
st.title("📈 EquityScope: Stock Analyzer")

# Theme toggle
st.sidebar.title("Settings")
theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = theme.lower()

# Define theme styles
light_theme = {
    "bg": "#FFFFFF",
    "text": "#111827",
    "plot_bg": "rgba(240, 240, 240, 0.5)",
    "grid": "rgba(150, 150, 150, 0.5)",
    "line": "#00A8E8",
    "sma20": "#FF6F61",
    "sma50": "#6B7280",
    "sma200": "#34D399",
    "bar_colors": ["#00C4B4", "#FF6F61", "#F4A261", "#34D399", "#6B7280", "#A78BFA", "#EC4899", "#EF4444"],
    "calc_header": "#00A8E8"
}
dark_theme = {
    "bg": "#1F2937",
    "text": "#F3F4F6",
    "plot_bg": "rgba(31, 41, 55, 0.8)",
    "grid": "rgba(107, 114, 128, 0.5)",
    "line": "#60A5FA",
    "sma20": "#F87171",
    "sma50": "#9CA3AF",
    "sma200": "#34D399",
    "bar_colors": ["#2DD4BF", "#F87171", "#FBBF24", "#34D399", "#9CA3AF", "#C4B5FD", "#F472B6", "#F87171"],
    "calc_header": "#60A5FA"
}
st.session_state.theme_styles = light_theme if st.session_state.theme == "light" else dark_theme

# Apply CSS with updated styles for sidebar background and tick labels
st.markdown(f"""
<style>
    .stApp {{
        background-color: {st.session_state.theme_styles['bg']} !important;
        color: {st.session_state.theme_styles['text']} !important;
    }}
    .stMarkdown, .stRadio > label, .stAlert, .company-info, .description, .metric-value, .calc-header, .st-expander {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    .calc-header {{
        color: {st.session_state.theme_styles['calc_header']} !important;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }}
    [data-testid="stTextInput"] label, [data-testid="stTextInput"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: { '#F9FAFB' if st.session_state.theme == 'light' else '#374151' } !important;
    }}
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1 {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] .stRadio > label p, [data-testid="stSidebar"] .stRadio > div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stDataFrame"] a {{
        color: {st.session_state.theme_styles['text']} !important;
        text-decoration: underline !important;
    }}
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .ticktext {{
        fill: #000000 !important;
        color: #000000 !important;
    }}
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-xtitle,
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-ytitle {{
        fill: #000000 !important;
        color: #000000 !important;
    }}
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
            st.warning(f"Could not fetch company info for {stock_ticker}.")

# Key metrics
st.subheader("📊 Key Metrics")
if stock_ticker:
    with st.spinner("Fetching key metrics..."):
        stock_info = get_stock_info(stock_ticker)
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
time_frame_options = ["1D", "1W", "1M", "6M", "1Y", "5Y", "10Y", "All"]
time_frame_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "10Y": "10y", "All": "max"}
selected_time_frame = st.selectbox("Select time frame:", time_frame_options, index=4)
selected_period = time_frame_map[selected_time_frame]

if stock_ticker:
    with st.spinner(f"Fetching price history for {selected_time_frame}..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])))
            fig.update_layout(
                title=dict(text=f"{stock_ticker} Price ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                legend=dict(font=dict(color=axis_text_color)),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price history for {stock_ticker}.")

# Moving averages
st.subheader("📈 Moving Averages")
if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: 50-day and 200-day SMAs may be less reliable for short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating moving averages..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            sma_20 = hist["Close"].rolling(window=20).mean()
            sma_50 = hist["Close"].rolling(window=50).mean()
            sma_200 = hist["Close"].rolling(window=200).mean()
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_20, mode="lines", name="20-day SMA", line=dict(color=st.session_state.theme_styles["sma20"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_50, mode="lines", name="50-day SMA", line=dict(color=st.session_state.theme_styles["sma50"])))
            fig.add_trace(go.Scatter(x=hist.index, y=sma_200, mode="lines", name="200-day SMA", line=dict(color=st.session_state.theme_styles["sma200"])))
            fig.update_layout(
                title=dict(text=f"{stock_ticker} Moving Averages ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                legend=dict(font=dict(color=axis_text_color)),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Moving Averages Explanation**: SMAs smooth price data to identify trends. The 20-day SMA reflects short-term trends, 50-day medium-term, and 200-day long-term.
            """)
        else:
            st.warning(f"Could not fetch price history for moving averages.")

# Technical indicators graph
st.subheader("📉 Technical Indicators Graph")
if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: Bollinger Bands and MACD may be less reliable for short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating technical indicators..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            window = 20
            sma = hist["Close"].rolling(window=window).mean()
            std = hist["Close"].rolling(window=window).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            exp12 = hist["Close"].ewm(span=12, adjust=False).mean()
            exp26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f"{stock_ticker} Price with Bollinger Bands", "MACD"), row_heights=[0.7, 0.3])
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close Price", line=dict(color=st.session_state.theme_styles["line"])), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=upper_band, mode="lines", name="Upper Band", line=dict(color=st.session_state.theme_styles["sma50"], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=lower_band, mode="lines", name="Lower Band", line=dict(color=st.session_state.theme_styles["sma50"], dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=macd, mode="lines", name="MACD", line=dict(color=st.session_state.theme_styles["line"])), row=2, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=signal, mode="lines", name="Signal Line", line=dict(color=st.session_state.theme_styles["sma20"])), row=2, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=histogram, name="Histogram", marker_color=st.session_state.theme_styles["sma50"]), row=2, col=1)
            fig.update_layout(
                height=600,
                title=dict(text=f"{stock_ticker} Technical Indicators ({selected_time_frame})", font=dict(color=st.session_state.theme_styles["text"])),
                showlegend=True,
                legend=dict(font=dict(color=axis_text_color)),
                xaxis2_title="Date",
                yaxis_title="Price (USD)",
                yaxis2_title="MACD",
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor=st.session_state.theme_styles["bg"],
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                xaxis2=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                ),
                yaxis2=dict(
                    title=dict(font=dict(color=axis_text_color)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"]
                )
            )
            fig.update_xaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            fig.update_yaxes(tickfont=dict(family="Arial", size=12, color=axis_text_color))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            **Bollinger Bands**: Prices near the upper band may indicate overbought conditions; near the lower band, oversold.
            **MACD**: MACD line crossing above the Signal line is bullish; below is bearish.
            """)
        else:
            st.warning(f"Could not fetch price history for technical indicators.")
# Technical indicators (RSI)
st.subheader("📉 Technical Indicators")
if selected_time_frame in ["1D", "1W"]:
    st.warning("Note: RSI may be less reliable for very short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating RSI..."):
        hist = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
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
                rsi_status = "Overbought (sell signal)" if latest_rsi > 70 else "Oversold (buy signal)" if latest_rsi < 30 else "Neutral"
                st.markdown(f'<p class="metric-value">14-Day RSI: {rsi_display} ({rsi_status})</p>', unsafe_allow_html=True)
                st.markdown("**RSI**: Above 70 indicates overbought; below 30 indicates oversold.")
            else:
                st.warning("Could not calculate RSI (insufficient data).")
        else:
            st.warning(f"Could not fetch price history for RSI.")

# Valuation Section
st.subheader("💰 Valuation")
st.markdown("""
Estimate the fair value of the stock using multiple valuation methods. Each method calculates an **Intrinsic Value per Share**, which you can compare to the current market price to assess whether the stock is undervalued or overvalued.
""")
if stock_ticker:
    with st.spinner("Calculating valuations..."):
        stock = yf.Ticker(stock_ticker)
        stock_info = get_stock_info(stock_ticker)
        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
        except:
            balance_sheet = None
            cash_flow = None
            st.warning(f"Could not fetch financial statements for {stock_ticker}.")
        if stock_info:
            current_price = stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))
            eps = stock_info.get("trailingEps")
            forward_eps = stock_info.get("forwardEps", eps)
            shares_outstanding = stock_info.get("sharesOutstanding")
            book_value = stock_info.get("bookValue")
            dividend_rate = stock_info.get("dividendRate")
            five_year_avg_dividend = stock_info.get("fiveYearAvgDividendYield", 0) / 100 if stock_info.get("fiveYearAvgDividendYield") else 0
            pe_ratio = stock_info.get("trailingPE")
            growth_rate = min(stock_info.get("earningsGrowth", 0.20), 0.25)
            beta = stock_info.get("beta", 1.0)
            discount_rate = 0.05 + beta * 0.02
            perpetual_growth = 0.03
            total_debt = stock_info.get("totalDebt", 0)
            total_cash = stock_info.get("totalCash", 0)
            valuations = []
            calculation_details = []
            # DCF
            try:
                if cash_flow is not None and not cash_flow.empty and forward_eps and shares_outstanding:
                    fcf = forward_eps * shares_outstanding * 0.6
                    growth_rates = [growth_rate * (1 - 0.05 * t) for t in range(5)]
                    cash_flows = [fcf * (1 + g) ** t for t, g in enumerate(growth_rates, 1)]
                    terminal_value = cash_flows[-1] * (1 + perpetual_growth) / (discount_rate - perpetual_growth)
                    cash_flows_with_terminal = cash_flows + [terminal_value]
                    pv_cash_flows = [cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows_with_terminal, 1)]
                    enterprise_value = sum(pv_cash_flows)
                    net_debt = total_debt - total_cash
                    equity_value = max(enterprise_value - net_debt, 0)
                    dcf_value = equity_value / shares_outstanding
                    dcf_value = f"${dcf_value:.2f}" if isinstance(dcf_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We estimate future cash flows and discount them to today’s value.
                    - **Starting Point**: Free Cash Flow = ${fcf:,.2f}
                    - **Growth**: {growth_rate*100:.1f}% initially, tapering over 5 years
                    - **Yearly Cash Flows**: {', '.join([f'${cf:,.2f}' for cf in cash_flows])}
                    - **Terminal Value**: ${terminal_value:,.2f}
                    - **Discounted Value**: ${sum(pv_cash_flows):,.2f}
                    - **Equity Value**: ${enterprise_value:,.2f} - ${net_debt:,.2f} = ${equity_value:,.2f}
                    - **Per Share**: ${equity_value:,.2f} ÷ {shares_outstanding:,} = {dcf_value}
                    """
                else:
                    dcf_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nWe need cash flow or earnings data."
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": dcf_value,
                    "Description": "Discounts future cash flows.",
                 
                    "Pros and Cons": "✅ Growth firms (e.g., AAPL).\n❌ Volatile cash flows (e.g., startups)."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Discounts future cash flows.",
                    "Pros and Cons": "✅ Growth firms (e.g., AAPL).\n❌ Volatile cash flows (e.g., startups)."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # DDM
            try:
                if dividend_rate and dividend_rate > 0 and forward_eps:
                    expected_dividend = dividend_rate * (1 + 0.10)
                    ddm_value = expected_dividend / (discount_rate - perpetual_growth)
                    ddm_value = f"${ddm_value:.2f}" if isinstance(ddm_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We value the stock based on future dividends.
                    - **Starting Point**: Dividend = ${dividend_rate:.2f}
                    - **Growth**: 10% annually
                    - **Expected Dividend**: ${dividend_rate:.2f} × (1 + 10%) = ${expected_dividend:.2f}
                    - **Discount Rate**: {discount_rate*100:.1f}%, Long-Term Growth: 3%
                    - **Per Share**: ${expected_dividend:.2f} ÷ ({discount_rate*100:.1f}% - 3%) = {ddm_value}
                    """
                else:
                    ddm_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo dividends or data."
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": ddm_value,
                    "Description": "Values stock via dividends.",
                    "Pros and Cons": "✅ Dividend stocks (e.g., PG).\n❌ Non-dividend stocks (e.g., TSLA)."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Values stock via dividends.",
                    "Pros and Cons": "✅ Dividend stocks (e.g., PG).\n❌ Non-dividend stocks (e.g., TSLA)."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # RIM
            try:
                if book_value and forward_eps and shares_outstanding:
                    roe = forward_eps / book_value if book_value != 0 else 0
                    retention_ratio = 1 - (dividend_rate / forward_eps if dividend_rate and forward_eps else 0)
                    residual_income = forward_eps - (discount_rate * book_value)
                    rim_value = book_value + (residual_income * retention_ratio * (1 + growth_rate) / (discount_rate - perpetual_growth))
                    rim_value = f"${rim_value:.2f}" if isinstance(rim_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We combine book value with excess earnings.
                    - **Starting Point**: Book Value = ${book_value:.2f}, EPS = ${forward_eps:.2f}
                    - **Retention**: {retention_ratio*100:.1f}%
                    - **Residual Income**: ${forward_eps:.2f} - ({discount_rate*100:.1f}% × ${book_value:.2f}) = ${residual_income:.2f}
                    - **Growth**: {growth_rate*100:.1f}%
                    - **Per Share**: ${book_value:.2f} + ${(residual_income * retention_ratio * (1 + growth_rate) / (discount_rate - perpetual_growth)):.2f} = {rim_value}
                    """
                else:
                    rim_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo book value or EPS."
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": rim_value,
                    "Description": "Uses book value and income.",
                    "Pros and Cons": "✅ Strong book value (e.g., JPM).\n❌ Low book value (e.g., tech startups)."
                })
                calculation_details.append(("Residual Income Model (RIM)", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Uses book value and income.",
                    "Pros and Cons": "✅ Strong book value (e.g., JPM).\n❌ Low book value (e.g., tech startups)."
                })
                calculation_details.append(("Residual Income Model (RIM)", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # Asset-Based
            try:
                if balance_sheet is not None and not balance_sheet.empty and shares_outstanding:
                    total_assets = balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 0
                    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in balance_sheet.index else 0
                    net_assets = total_assets - total_liabilities
                    asset_value = (net_assets / shares_outstanding) * 1.5
                    asset_value = f"${asset_value:.2f}" if isinstance(asset_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We calculate net assets with a premium.
                    - **Starting Point**: Assets = ${total_assets:,.2f}, Liabilities = ${total_liabilities:,.2f}
                    - **Net Assets**: ${total_assets:,.2f} - ${total_liabilities:,.2f} = ${net_assets:,.2f}
                    - **Premium**: 1.5x
                    - **Per Share**: ${net_assets:,.2f} × 1.5 ÷ {shares_outstanding:,} = {asset_value}
                    """
                else:
                    asset_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo balance sheet data."
                valuations.append({
                    "Method": "Asset-Based Valuation",
                    "Intrinsic Value per Share": asset_value,
                    "Description": "Net assets with premium.",
                    "Pros and Cons": "✅ Asset-heavy firms (e.g., REITs).\n❌ High-debt firms (e.g., tech)."
                })
                calculation_details.append(("Asset-Based Valuation", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Asset-Based Valuation",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Net assets with premium.",
                    "Pros and Cons": "✅ Asset-heavy firms (e.g., REITs).\n❌ High-debt firms (e.g., tech)."
                })
                calculation_details.append(("Asset-Based Valuation", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # EPV
            try:
                if forward_eps:
                    normalized_earnings = forward_eps * (1 - five_year_avg_dividend)
                    epv_value = normalized_earnings / (discount_rate - 0.015)
                    epv_value = f"${epv_value:.2f}" if isinstance(epv_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We estimate sustainable earnings.
                    - **Starting Point**: EPS = ${forward_eps:.2f}, Dividend Yield = {five_year_avg_dividend*100:.1f}%
                    - **Earnings**: ${forward_eps:.2f} × (1 - {five_year_avg_dividend:.2f}) = ${normalized_earnings:.2f}
                    - **Discount Rate**: {discount_rate*100:.1f}% - 1.5%
                    - **Per Share**: ${normalized_earnings:.2f} ÷ {(discount_rate - 0.015)*100:.1f}% = {epv_value}
                    """
                else:
                    epv_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo EPS data."
                valuations.append({
                    "Method": "Earnings Power Value (EPV)",
                    "Intrinsic Value per Share": epv_value,
                    "Description": "Sustainable earnings value.",
                    "Pros and Cons": "✅ Stable earnings (e.g., staples).\n❌ Cyclical firms (e.g., airlines)."
                })
                calculation_details.append(("Earnings Power Value (EPV)", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Earnings Power Value (EPV)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Sustainable earnings value.",
                    "Pros and Cons": "✅ Stable earnings (e.g., staples).\n❌ Cyclical firms (e.g., airlines)."
                })
                calculation_details.append(("Earnings Power Value (EPV)", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # Graham
            try:
                if forward_eps and growth_rate:
                    graham_value = forward_eps * (10 + 2.5 * growth_rate * 100)
                    graham_value = f"${graham_value:.2f}" if isinstance(graham_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We use a formula based on earnings and growth.
                    - **Starting Point**: EPS = ${forward_eps:.2f}, Growth = {growth_rate*100:.1f}%
                    - **Multiplier**: 10 + 2.5 × {growth_rate*100:.1f} = {(10 + 2.5 * growth_rate * 100):.1f}
                    - **Per Share**: ${forward_eps:.2f} × {(10 + 2.5 * growth_rate * 100):.1f} = {graham_value}
                    """
                else:
                    graham_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo EPS or growth data."
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": graham_value,
                    "Description": "EPS with growth multiplier.",
                    "Pros and Cons": "✅ Value stocks (e.g., KO).\n❌ High-growth stocks (e.g., NVDA)."
                })
                calculation_details.append(("Graham Method", calc))
            except Exception as e:
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "EPS with growth multiplier.",
                    "Pros and Cons": "✅ Value stocks (e.g., KO).\n❌ High-growth stocks (e.g., NVDA)."
                })
                calculation_details.append(("Graham Method", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            # PEG
            try:
                if pe_ratio and growth_rate and growth_rate != 0:
                    peg = pe_ratio / (growth_rate * 100)
                    if peg <= 1:
                        peg_value = current_price * 1.5
                    else:
                        peg_value = current_price / (peg * 0.7)
                    peg_value = f"${peg_value:.2f}" if isinstance(peg_value, (int, float)) else "N/A"
                    calc = f"""
                    **What We Did** 📊
                    We adjust P/E based on growth.
                    - **Starting Point**: P/E = {pe_ratio:.2f}, Growth = {growth_rate*100:.1f}%, Price = ${current_price:.2f}
                    - **PEG**: {pe_ratio:.2f} ÷ {growth_rate*100:.1f} = {peg:.2f}
                    - **Per Share**: {'${:.2f} × 1.5 = {}'.format(current_price, peg_value) if peg <= 1 else '${:.2f} ÷ ({:.2f} × 0.7) = {}'.format(current_price, peg, peg_value)}
                    """
                else:
                    peg_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo P/E or growth data."
                valuations.append({
                    "Method": "PEG Ratio",
                    "Intrinsic Value per Share": peg_value,
                    "Description": "P/E adjusted for growth.",
                    "Pros and Cons": "✅ Growth stocks (e.g., AMZN).\n❌ Low-growth firms (e.g., utilities)."
                })
                calculation_details.append(("PEG Ratio", calc))
            except Exception as e:
                valuations.append({
                    "Method": "PEG Ratio",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "P/E adjusted for growth.",
                    "Pros and Cons": "✅ Growth stocks (e.g., AMZN).\n❌ Low-growth firms (e.g., utilities)."
                })
                calculation_details.append(("PEG Ratio", f"**Why It’s Missing** 🚫\nError: {str(e)}."))
            valuation_df = pd.DataFrame(valuations)
            st.markdown(f"**Current Market Price**: ${current_price:.2f}")
            st.markdown("### Valuation Estimates")
            st.dataframe(
                valuation_df[["Method", "Intrinsic Value per Share", "Description", "Pros and Cons"]],
                width=1200,
                column_config={
                    "Intrinsic Value per Share": st.column_config.TextColumn(help="Estimated fair price per share."),
                    "Pros and Cons": st.column_config.TextColumn(help="When to use this method.")
                }
            )
            
            st.markdown("### Valuation Comparison Chart")
            chart_data = valuation_df[valuation_df["Intrinsic Value per Share"] != "N/A"]
            if not chart_data.empty:
                methods = chart_data["Method"].tolist()
                intrinsic_values = [float(val.replace("$", "")) for val in chart_data["Intrinsic Value per Share"]]
                chart_methods = methods + ["Current Price"]
                chart_values = intrinsic_values + [current_price]
                axis_text_color = "#000000" if st.session_state.theme == "light" else st.session_state.theme_styles['text']
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=chart_methods,
                        y=chart_values,
                        marker_color=st.session_state.theme_styles["bar_colors"][:len(chart_methods)],
                        text=[f"${v:.2f}" for v in chart_values],
                        textposition="auto",
                        hovertemplate="%{x}: $%{y:.2f}<extra></extra>"
                    )
                )
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color=st.session_state.theme_styles["bar_colors"][-1],
                    annotation_text="Current Price",
                    annotation_position="top right",
                    annotation_font=dict(size=12, color=st.session_state.theme_styles["text"])
                )
                fig.update_layout(
                    title=dict(text=f"{stock_ticker} Intrinsic Value vs. Current Price", font=dict(color=st.session_state.theme_styles["text"])),
                    xaxis_title="Valuation Method",
                    yaxis_title="Price per Share (USD)",
                    showlegend=False,
                    legend=dict(font=dict(color=axis_text_color)),
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor=st.session_state.theme_styles["bg"],
                    font=dict(family="Arial", size=14, color=axis_text_color),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        title=dict(font=dict(color=axis_text_color)),
                        tickfont=dict(family="Arial", size=14, color=axis_text_color),
                        tickangle=45,
                        gridcolor=st.session_state.theme_styles["grid"]
                    ),
                    yaxis=dict(
                        title=dict(font=dict(color=axis_text_color)),
                        tickfont=dict(family="Arial", size=14, color=axis_text_color),
                        gridcolor=st.session_state.theme_styles["grid"]
                    )
                )
                fig.update_xaxes(tickfont=dict(family="Arial", size=14, color=axis_text_color))
                fig.update_yaxes(tickfont=dict(family="Arial", size=14, color=axis_text_color))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for valuation chart.")
            st.markdown("### How Intrinsic Values Are Calculated")
            st.markdown("Click to expand each method for a step-by-step breakdown.")
            for method, calc in calculation_details:
                with st.expander(f"{method} Calculation"):
                    st.markdown(f'<span class="calc-header">{method}</span>', unsafe_allow_html=True)
                    st.markdown(calc)
        else:
            st.warning("Could not fetch data for valuations.")

# Learning Section
st.subheader("📚 Learn")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Test your investing knowledge with interactive quizzes.</p>', unsafe_allow_html=True)
quiz_level = st.selectbox("Select Quiz Level:", ["Beginner", "Intermediate", "Expert"])

# Enhanced CSS to fix gray font and style buttons
st.markdown(f"""
<style>
    .stRadio [role="radio"] label, .stRadio [role="radio"] label p, .stRadio [role="radio"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    .stRadio > label, .stRadio > label p, .stRadio > div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    .stAlert > div, .stAlert > div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    .stMarkdown, .stMarkdown p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    button[kind="secondary"] {{
        background-color: {st.session_state.theme_styles['bar_colors'][3]} !important;
        color: {st.session_state.theme_styles['text']} !important;
        border: 1px solid {st.session_state.theme_styles['text']} !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        opacity: 1 !important;
    }}
    button[kind="secondary"]:hover {{
        background-color: {st.session_state.theme_styles['bar_colors'][0]} !important;
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
</style>
""", unsafe_allow_html=True)

quizzes = {
    "Beginner": [
        {"question": "What is a stock?", "options": ["A loan to a company", "Ownership in a company", "A type of bond"], "answer": "Ownership in a company"},
        {"question": "What does P/E ratio measure?", "options": ["Profit margin", "Price per earnings", "Portfolio value"], "answer": "Price per earnings"},
        {"question": "What is a dividend?", "options": ["A loan repayment", "A share of profits paid to shareholders", "A stock split"], "answer": "A share of profits paid to shareholders"},
        {"question": "What is a stock market?", "options": ["A place to buy bonds", "A marketplace for trading company shares", "A savings account"], "answer": "A marketplace for trading company shares"},
        {"question": "What is a bull market?", "options": ["Falling prices", "Rising prices", "Stable prices"], "answer": "Rising prices"},
        {"question": "What is a bear market?", "options": ["Rising prices", "Falling prices", "Neutral market"], "answer": "Falling prices"},
        {"question": "What does diversification mean?", "options": ["Investing in one stock", "Spreading investments across assets", "Selling all stocks"], "answer": "Spreading investments across assets"},
        {"question": "What is a blue-chip stock?", "options": ["A new company stock", "A stable, large company stock", "A risky stock"], "answer": "A stable, large company stock"},
        {"question": "What is the S&P 500?", "options": ["A stock price", "An index of 500 large companies", "A type of bond"], "answer": "An index of 500 large companies"},
        {"question": "What is a brokerage account?", "options": ["A savings account", "An account to buy/sell securities", "A retirement fund"], "answer": "An account to buy/sell securities"}
    ],
    "Intermediate": [
        {"question": "What is a golden cross?", "options": ["50-day SMA crossing above 200-day SMA", "A sharp price drop", "A dividend increase"], "answer": "50-day SMA crossing above 200-day SMA"},
        {"question": "What does RSI above 70 indicate?", "options": ["Oversold", "Overbought", "Neutral"], "answer": "Overbought"},
        {"question": "What is beta?", "options": ["A measure of debt", "A measure of stock volatility", "A type of option"], "answer": "A measure of stock volatility"},
        {"question": "What is a death cross?", "options": ["50-day SMA crossing below 200-day SMA", "A sharp price rise", "A stock split"], "answer": "50-day SMA crossing below 200-day SMA"},
        {"question": "What does MACD stand for?", "options": ["Market Average Daily Change", "Moving Average Convergence Divergence", "Momentum Adjusted Capital Divergence"], "answer": "Moving Average Convergence Divergence"},
        {"question": "What is a stop-loss order?", "options": ["An order to buy at a higher price", "An order to sell if price drops to a level", "An order to hold a stock"], "answer": "An order to sell if price drops to a level"},
        {"question": "What is short selling?", "options": ["Buying low, selling high", "Selling borrowed shares to buy back later", "Holding stocks long-term"], "answer": "Selling borrowed shares to buy back later"},
        {"question": "What does a high P/B ratio indicate?", "options": ["Undervalued stock", "Overvalued stock", "Stable stock"], "answer": "Overvalued stock"},
        {"question": "What is a market cap?", "options": ["Total debt of a company", "Total value of a company's shares", "Annual revenue"], "answer": "Total value of a company's shares"},
        {"question": "What is a dividend yield?", "options": ["Annual dividend per share divided by stock price", "Total company profits", "Stock price increase"], "answer": "Annual dividend per share divided by stock price"}
    ],
    "Expert": [
        {"question": "What is the DCF valuation method?", "options": ["Dividend discount model", "Discounted cash flow", "Debt-to-equity calculation"], "answer": "Discounted cash flow"},
        {"question": "What does a PEG ratio below 1 suggest?", "options": ["Overvalued stock", "Undervalued stock", "High debt"], "answer": "Undervalued stock"},
        {"question": "What is the purpose of Bollinger Bands?", "options": ["Measure earnings", "Identify overbought/oversold conditions", "Calculate dividends"], "answer": "Identify overbought/oversold conditions"},
        {"question": "What is a WACC?", "options": ["Weighted Average Cost of Capital", "Weighted Annual Cash Conversion", "Weighted Asset Capital Cost"], "answer": "Weighted Average Cost of Capital"},
        {"question": "What does a high Sharpe ratio indicate?", "options": ["High risk", "Better risk-adjusted return", "Low volatility"], "answer": "Better risk-adjusted return"},
        {"question": "What is a leveraged buyout (LBO)?", "options": ["Buying a company with borrowed funds", "Selling company assets", "Merging two companies"], "answer": "Buying a company with borrowed funds"},
        {"question": "What is the Efficient Market Hypothesis?", "options": ["Markets are always inefficient", "Stock prices reflect all information", "Markets are predictable"], "answer": "Stock prices reflect all information"},
        {"question": "What is a Black-Scholes model used for?", "options": ["Valuing bonds", "Pricing options", "Calculating dividends"], "answer": "Pricing options"},
        {"question": "What does a high alpha indicate?", "options": ["Underperformance", "Outperformance relative to benchmark", "Low risk"], "answer": "Outperformance relative to benchmark"},
        {"question": "What is a poison pill strategy?", "options": ["A merger tactic", "A defense against hostile takeovers", "A dividend policy"], "answer": "A defense against hostile takeovers"}
    ]
}

if quiz_level:
    st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>{quiz_level} Quiz</b></p>', unsafe_allow_html=True)
    for i, q in enumerate(quizzes[quiz_level], 1):
        st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>Question {i}: {q["question"]}</b></p>', unsafe_allow_html=True)
        answer = st.radio(f"Select an answer for question {i}:", q["options"], key=f"quiz_{quiz_level}_{i}")
        if st.button(f"Check Answer {i}"):
            if answer == q["answer"]:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The answer is: {q['answer']}.")
# Portfolio
st.subheader("💼 Portfolio")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Track your investments and analyze sentiment.</p>', unsafe_allow_html=True)

# Fetch sentiment for a ticker
def get_sentiment(ticker):
    cache_key = f"sentiment_{ticker}"
    if cache_key not in st.session_state.cache:
        try:
            subreddits = ["wallstreetbets", "stocks"]
            posts = []
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(ticker, limit=5):
                    text = submission.title + " " + (submission.selftext[:200] if submission.selftext else "")
                    score = analyzer.polarity_scores(text)["compound"]
                    posts.append(score)
            if posts:
                avg_score = np.mean(posts)
                sentiment = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
                st.session_state.cache[cache_key] = f"{sentiment} ({avg_score:.2f})"
            else:
                st.session_state.cache[cache_key] = "N/A"
        except:
            st.session_state.cache[cache_key] = "N/A"
    return st.session_state.cache[cache_key]

# Portfolio input
portfolio_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL,MSFT):", value="AAPL,MSFT").strip().upper()
shares_input = st.text_input("Enter shares for each ticker (comma-separated, e.g., 100,50):", value="100,50").strip()

# Add CSS to fix Portfolio Simulator headers
st.markdown(f"""
<style>
    [data-testid="stNumberInput"] label, [data-testid="stNumberInput"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    [data-testid="stSlider"] label, [data-testid="stSlider"] div p {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
</style>
""", unsafe_allow_html=True)

if portfolio_input and shares_input:
    tickers = [t.strip() for t in portfolio_input.split(",")]
    try:
        shares = [int(s.strip()) for s in shares_input.split(",")]
        if len(tickers) != len(shares):
            st.error("Number of tickers must match number of shares.")
        else:
            portfolio_data = []
            with st.spinner("Fetching portfolio data..."):
                for ticker, share in zip(tickers, shares):
                    stock_info = get_stock_info(ticker)
                    if stock_info:
                        price = stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))
                        pe_ratio = stock_info.get("trailingPE", "N/A")
                        pb_ratio = stock_info.get("priceToBook", "N/A")
                        sentiment = get_sentiment(ticker)
                        portfolio_data.append({
                            "Ticker": ticker,
                            "Shares": share,
                            "Price": f"${price:.2f}",
                            "Value": f"${price * share:.2f}",
                            "P/E Ratio": f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio,
                            "P/B Ratio": f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio,
                            "Sentiment": sentiment
                        })
                    else:
                        st.warning(f"Could not fetch data for {ticker}.")
            if portfolio_data:
                st.markdown("### Portfolio Summary")
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(
                    portfolio_df,
                    use_container_width=True,
                    column_config={
                        "Sentiment": st.column_config.TextColumn(help="AI-driven sentiment from Reddit posts.")
                    }
                )

                # Export Functionality for Portfolio Summary
                csv = portfolio_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Summary as CSV",
                    data=csv,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

                # Portfolio simulator
                st.markdown("### Portfolio Simulator")
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Estimate your portfolio’s growth over time.</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">**Note**: The S&P 500 average annual return over the past 100 years is approximately 10.3% (including dividends). Use this as a reference for growth rate.</p>', unsafe_allow_html=True)
                current_age = st.number_input("Enter your current age:", min_value=18, max_value=100, value=30, step=1)
                retirement_age = 65
                years_to_retirement = max(retirement_age - current_age, 1)
                growth_rate = st.slider("Expected Annual Growth Rate (%):", 0.0, 20.0, 10.3, 1.0)
                years = st.slider("Investment Horizon (Years):", 1, 50, years_to_retirement, 1)
                total_value = sum(float(item["Value"].replace("$", "")) for item in portfolio_data)
                future_value = total_value * (1 + growth_rate / 100) ** years
                retirement_value = total_value * (1 + growth_rate / 100) ** years_to_retirement
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Current Portfolio Value: ${total_value:,.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Future Value ({years} years, {growth_rate}% growth): ${future_value:,.2f}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Value at Retirement (Age {retirement_age}, {years_to_retirement} years, {growth_rate}% growth): ${retirement_value:,.2f}</p>', unsafe_allow_html=True)
    except ValueError:
        st.error("Invalid shares input. Please enter numbers (e.g., 100,50).")

# Reddit Sentiment
st.subheader("🗣️ Reddit Sentiment")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Analyze sentiment from Reddit posts.</p>', unsafe_allow_html=True)
sentiment_ticker = st.text_input("Enter a ticker for sentiment analysis:", value="AAPL").strip().upper()

if sentiment_ticker:
    with st.spinner(f"Fetching Reddit posts for {sentiment_ticker}..."):
        try:
            subreddits = ["wallstreetbets", "stocks"]
            posts = []
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(sentiment_ticker, limit=5):
                    text = submission.title + " " + (submission.selftext[:200] if submission.selftext else "")
                    score = analyzer.polarity_scores(text)["compound"]
                    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                    posts.append({
                        "Title": submission.title,
                        "URL": submission.url,
                        "Snippet": submission.selftext[:100] + "..." if submission.selftext else "No text",
                        "Sentiment": f"{sentiment} ({score:.2f})",
                        "Reddit Score": submission.score,
                        "Date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d")
                    })
            if posts:
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Overall Sentiment: {get_sentiment(sentiment_ticker)}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><strong>Sentiment Score Explanation:</strong> The score ranges from -1 (most negative) to +1 (most positive). Here’s what it means:<br>'
                            f'• <strong>Excellent</strong>: > 0.5 (highly positive sentiment)<br>'
                            f'• <strong>Good</strong>: 0.05 to 0.5 (positive sentiment)<br>'
                            f'• <strong>Neutral</strong>: -0.05 to 0.05 (neutral sentiment)<br>'
                            f'• <strong>Bad</strong>: -0.5 to -0.05 (negative sentiment)<br>'
                            f'• <strong>Very Bad</strong>: < -0.5 (highly negative sentiment)<br>'
                            f'For example, a score of 0.09 indicates a slightly positive ("Good") sentiment.</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><strong>Reddit Score Explanation:</strong> The Reddit Score is the net upvotes (upvotes minus downvotes) a post has received. It reflects the post’s popularity or community agreement:<br>'
                            f'• <strong>High Score</strong> (e.g., > 500): Indicates strong community engagement or agreement.<br>'
                            f'• <strong>Moderate Score</strong> (50–500): Moderate attention from the community.<br>'
                            f'• <strong>Low Score</strong> (< 50): Limited community impact.<br>'
                            f'A high score on a positive sentiment post may suggest strong community optimism about the stock.</p>', unsafe_allow_html=True)
                st.markdown("### Recent Reddit Posts")
                reddit_df = pd.DataFrame(posts)
                st.dataframe(
                    reddit_df,
                    use_container_width=True,
                    column_config={
                        "Title": st.column_config.LinkColumn(label="Title", help="Click to read the full post on Reddit"),
                        "URL": st.column_config.LinkColumn(label="Reddit Post URL", help="Click to read the full post on Reddit"),
                        "Sentiment": st.column_config.TextColumn(help="AI-driven sentiment score for this post."),
                        "Reddit Score": st.column_config.NumberColumn(help="Net upvotes (upvotes minus downvotes) received on Reddit.")
                    }
                )

                # Export Functionality for Reddit Sentiment
                csv = reddit_df.to_csv(index=False)
                st.download_button(
                    label="Download Reddit Sentiment as CSV",
                    data=csv,
                    file_name=f"reddit_sentiment_{sentiment_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            else:
                st.warning(f"No Reddit posts found for {sentiment_ticker}.")
        except Exception as e:
            st.error(f"Error fetching Reddit posts: {str(e)}. Please check API credentials.")

# Stock News
st.subheader("📰 Stock News")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Latest news articles related to the stock.</p>', unsafe_allow_html=True)

if stock_ticker:
    with st.spinner(f"Fetching news for {stock_ticker}..."):
        try:
            stock = yf.Ticker(stock_ticker)
            news_items = stock.news
            if news_items and any("title" in item or "headline" in item for item in news_items):
                news_data = []
                for item in news_items[:5]:
                    title = item.get("title") or item.get("headline") or "N/A"
                    link = item.get("link") or item.get("url") or "#"
                    publisher = item.get("publisher") or item.get("source") or "Unknown"
                    timestamp = item.get("providerPublishTime") or item.get("publishedAt") or 0
                    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d") if timestamp else "N/A"
                    summary = item.get("summary") or item.get("description") or "No summary available."
                    if len(summary) > 100:
                        summary = summary[:100] + "..."
                    news_data.append({
                        "Title": title,
                        "URL": link,
                        "Publisher": publisher,
                        "Date": date,
                        "Summary": summary
                    })
            else:
                news_data = None
        except Exception as e:
            st.warning(f"Failed to fetch news from Yahoo Finance: {str(e)}. Falling back to NewsAPI.")
            news_data = None

        if not news_data:
            try:
                import requests
                newsapi_key = "57d660f019654d129b06bba77f1014d4"
                ticker_to_company = {
                    "AAPL": "Apple",
                    "MSFT": "Microsoft",
                    "GOOGL": "Google",
                    "AMZN": "Amazon",
                    "TSLA": "Tesla"
                }
                company_name = ticker_to_company.get(stock_ticker, stock_ticker)
                query = f"{company_name} ({stock_ticker} OR stock OR earnings OR finance OR shares OR market)"
                url = f"https://newsapi.org/v2/everything?q={query}&apiKey={newsapi_key}&language=en&sortBy=publishedAt"
                response = requests.get(url)
                response.raise_for_status()
                news_items = response.json().get("articles", [])
                if news_items:
                    news_data = []
                    seen_urls = set()
                    stock_keywords = ["stock", "earnings", "finance", "shares", "market", stock_ticker.lower()]
                    for item in news_items[:15]:
                        title = item.get("title", "N/A").lower()
                        summary = item.get("description", "No summary available.").lower()
                        link = item.get("url", "#")
                        if link in seen_urls:
                            continue
                        is_relevant = any(keyword in title or keyword in summary for keyword in stock_keywords)
                        if not is_relevant:
                            continue
                        seen_urls.add(link)
                        publisher = item.get("source", {}).get("name", "Unknown")
                        date = item.get("publishedAt", "N/A")
                        if date != "N/A":
                            date = date.split("T")[0]
                        summary = item.get("description", "No summary available.")
                        if len(summary) > 100:
                            summary = summary[:100] + "..."
                        news_data.append({
                            "Title": item.get("title", "N/A"),
                            "URL": link,
                            "Publisher": publisher,
                            "Date": date,
                            "Summary": summary
                        })
                        if len(news_data) >= 5:
                            break
                    if not news_data:
                        st.warning(f"No relevant news articles found for {stock_ticker} using NewsAPI.")
                else:
                    st.warning(f"No news found for {stock_ticker} using NewsAPI.")
                    news_data = None
            except Exception as e:
                st.error(f"Error fetching news from NewsAPI: {str(e)}. Please check your API key or internet connection.")
                news_data = None

        if news_data and any(row["Title"] != "N/A" for row in news_data):
            st.markdown("### Recent News Articles")
            news_df = pd.DataFrame(news_data)
            st.dataframe(
                news_df,
                use_container_width=True,
                column_config={
                    "Title": st.column_config.LinkColumn(label="Title", help="Click to read the full article"),
                    "URL": st.column_config.LinkColumn(label="Article URL", help="Click to read the full article"),
                    "Summary": st.column_config.TextColumn(help="A brief summary of the article.")
                }
            )

            # Export functionality
            csv = news_df.to_csv(index=False)
            st.download_button(
                label="Download Stock News as CSV",
                data=csv,
                file_name=f"stock_news_{stock_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        elif not news_data:
            st.warning("Unable to fetch news articles from any source.")

# Stock Screener
st.subheader("🔎 Stock Screener")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Find stocks based on multiple financial metrics.</p>', unsafe_allow_html=True)

# Expanded list of stocks with additional metrics
stock_list = [
    {"Ticker": "AAPL", "Company Name": "Apple Inc.", "Sector": "Technology", "Market Cap": 2.5e12, "P/E Ratio": 28.5, "P/B Ratio": 36.0, "Dividend Yield": 0.5, "Beta": 1.2, "Revenue Growth": 5.0, "Debt/Equity": 1.8, "ROE": 150.0, "P/S Ratio": 7.0},
    {"Ticker": "MSFT", "Company Name": "Microsoft Corporation", "Sector": "Technology", "Market Cap": 2.1e12, "P/E Ratio": 35.0, "P/B Ratio": 12.0, "Dividend Yield": 0.8, "Beta": 0.9, "Revenue Growth": 15.0, "Debt/Equity": 0.5, "ROE": 40.0, "P/S Ratio": 10.0},
    {"Ticker": "GOOGL", "Company Name": "Alphabet Inc.", "Sector": "Technology", "Market Cap": 1.8e12, "P/E Ratio": 25.0, "P/B Ratio": 6.5, "Dividend Yield": 0.0, "Beta": 1.0, "Revenue Growth": 10.0, "Debt/Equity": 0.1, "ROE": 25.0, "P/S Ratio": 6.0},
    {"Ticker": "JPM", "Company Name": "JPMorgan Chase & Co.", "Sector": "Financials", "Market Cap": 5.0e11, "P/E Ratio": 12.0, "P/B Ratio": 1.5, "Dividend Yield": 2.5, "Beta": 1.1, "Revenue Growth": 3.0, "Debt/Equity": 1.2, "ROE": 15.0, "P/S Ratio": 3.0},
    {"Ticker": "BAC", "Company Name": "Bank of America", "Sector": "Financials", "Market Cap": 3.5e11, "P/E Ratio": 14.5, "P/B Ratio": 1.2, "Dividend Yield": 2.0, "Beta": 1.3, "Revenue Growth": 2.0, "Debt/Equity": 1.5, "ROE": 10.0, "P/S Ratio": 2.5},
    {"Ticker": "XOM", "Company Name": "Exxon Mobil Corporation", "Sector": "Energy", "Market Cap": 4.0e11, "P/E Ratio": 15.0, "P/B Ratio": 2.0, "Dividend Yield": 5.0, "Beta": 0.9, "Revenue Growth": -5.0, "Debt/Equity": 0.8, "ROE": 12.0, "P/S Ratio": 1.5},
    {"Ticker": "CVX", "Company Name": "Chevron Corporation", "Sector": "Energy", "Market Cap": 3.0e11, "P/E Ratio": 13.5, "P/B Ratio": 1.8, "Dividend Yield": 4.5, "Beta": 1.0, "Revenue Growth": -3.0, "Debt/Equity": 0.7, "ROE": 10.0, "P/S Ratio": 1.8},
    {"Ticker": "PFE", "Company Name": "Pfizer Inc.", "Sector": "Healthcare", "Market Cap": 2.5e11, "P/E Ratio": 10.0, "P/B Ratio": 2.5, "Dividend Yield": 4.0, "Beta": 0.6, "Revenue Growth": 8.0, "Debt/Equity": 0.9, "ROE": 20.0, "P/S Ratio": 2.0},
    {"Ticker": "JNJ", "Company Name": "Johnson & Johnson", "Sector": "Healthcare", "Market Cap": 4.5e11, "P/E Ratio": 16.0, "P/B Ratio": 5.0, "Dividend Yield": 2.8, "Beta": 0.5, "Revenue Growth": 4.0, "Debt/Equity": 0.4, "ROE": 30.0, "P/S Ratio": 4.5},
    {"Ticker": "WMT", "Company Name": "Walmart Inc.", "Sector": "Consumer Staples", "Market Cap": 4.0e11, "P/E Ratio": 20.0, "P/B Ratio": 4.0, "Dividend Yield": 1.5, "Beta": 0.4, "Revenue Growth": 3.0, "Debt/Equity": 0.6, "ROE": 20.0, "P/S Ratio": 0.8},
    {"Ticker": "PG", "Company Name": "Procter & Gamble", "Sector": "Consumer Staples", "Market Cap": 3.5e11, "P/E Ratio": 22.0, "P/B Ratio": 6.0, "Dividend Yield": 2.4, "Beta": 0.3, "Revenue Growth": 2.0, "Debt/Equity": 0.5, "ROE": 25.0, "P/S Ratio": 4.0},
    {"Ticker": "TSLA", "Company Name": "Tesla, Inc.", "Sector": "Consumer Discretionary", "Market Cap": 8.0e11, "P/E Ratio": 50.0, "P/B Ratio": 15.0, "Dividend Yield": 0.0, "Beta": 2.0, "Revenue Growth": 30.0, "Debt/Equity": 0.3, "ROE": 25.0, "P/S Ratio": 8.0},
    {"Ticker": "AMZN", "Company Name": "Amazon.com, Inc.", "Sector": "Consumer Discretionary", "Market Cap": 1.7e12, "P/E Ratio": 45.0, "P/B Ratio": 8.0, "Dividend Yield": 0.0, "Beta": 1.2, "Revenue Growth": 20.0, "Debt/Equity": 0.4, "ROE": 20.0, "P/S Ratio": 4.0}
]

# Convert stock list to DataFrame
stocks_df = pd.DataFrame(stock_list)

# Function to format Market Cap for display
def format_market_cap(market_cap):
    if market_cap >= 1_000_000_000_000:
        return f"${market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.2f}B"
    else:
        return f"${market_cap / 1_000_000:.2f}M"

stocks_df["Market Cap"] = stocks_df["Market Cap"].apply(format_market_cap)
stocks_df["Dividend Yield"] = stocks_df["Dividend Yield"].apply(lambda x: f"{x:.1f}%")
stocks_df["Revenue Growth"] = stocks_df["Revenue Growth"].apply(lambda x: f"{x:.1f}%")
stocks_df["Debt/Equity"] = stocks_df["Debt/Equity"].apply(lambda x: f"{x:.2f}")
stocks_df["ROE"] = stocks_df["ROE"].apply(lambda x: f"{x:.1f}%")
stocks_df["P/S Ratio"] = stocks_df["P/S Ratio"].apply(lambda x: f"{x:.2f}")

# Filters
st.markdown("### Filter Stocks")
sectors = sorted(stocks_df["Sector"].unique())
selected_sectors = st.multiselect("Select Sectors:", sectors, default=sectors)

# Market Cap filter
market_cap_min, market_cap_max = st.slider(
    "Market Cap Range ($B):",
    min_value=0.0,
    max_value=3000.0,
    value=(0.0, 3000.0),
    step=10.0
)

# P/E Ratio filter
pe_min, pe_max = st.slider(
    "P/E Ratio Range:",
    min_value=0.0,
    max_value=60.0,
    value=(0.0, 60.0),
    step=1.0
)

# P/B Ratio filter
pb_min, pb_max = st.slider(
    "P/B Ratio Range:",
    min_value=0.0,
    max_value=40.0,
    value=(0.0, 40.0),
    step=1.0
)

# Dividend Yield filter
div_min, div_max = st.slider(
    "Dividend Yield Range (%):",
    min_value=0.0,
    max_value=6.0,
    value=(0.0, 6.0),
    step=0.1
)

# Beta filter
beta_min, beta_max = st.slider(
    "Beta Range:",
    min_value=0.0,
    max_value=2.5,
    value=(0.0, 2.5),
    step=0.1
)

# Revenue Growth filter
rev_growth_min, rev_growth_max = st.slider(
    "Revenue Growth Range (%):",
    min_value=-10.0,
    max_value=35.0,
    value=(-10.0, 35.0),
    step=1.0
)

# Debt/Equity filter
debt_equity_min, debt_equity_max = st.slider(
    "Debt/Equity Range:",
    min_value=0.0,
    max_value=2.0,
    value=(0.0, 2.0),
    step=0.1
)

# ROE filter
roe_min, roe_max = st.slider(
    "ROE Range (%):",
    min_value=0.0,
    max_value=160.0,
    value=(0.0, 160.0),
    step=1.0
)

# P/S Ratio filter
ps_min, ps_max = st.slider(
    "P/S Ratio Range:",
    min_value=0.0,
    max_value=12.0,
    value=(0.0, 12.0),
    step=0.5
)

# Apply filters
filtered_df = stocks_df.copy()

# Filter by sector
filtered_df = filtered_df[filtered_df["Sector"].isin(selected_sectors)]

# Convert Market Cap back to numeric for filtering
filtered_df["Market Cap Numeric"] = filtered_df["Market Cap"].apply(
    lambda x: float(x.replace("$", "").replace("T", "e12").replace("B", "e9").replace("M", "e6"))
)
filtered_df = filtered_df[
    (filtered_df["Market Cap Numeric"] >= market_cap_min * 1e9) &
    (filtered_df["Market Cap Numeric"] <= market_cap_max * 1e9)
]

# Filter by P/E Ratio
filtered_df = filtered_df[
    (filtered_df["P/E Ratio"] >= pe_min) &
    (filtered_df["P/E Ratio"] <= pe_max)
]

# Filter by P/B Ratio
filtered_df = filtered_df[
    (filtered_df["P/B Ratio"] >= pb_min) &
    (filtered_df["P/B Ratio"] <= pb_max)
]

# Filter by Dividend Yield
filtered_df["Dividend Yield Numeric"] = filtered_df["Dividend Yield"].str.replace("%", "").astype(float)
filtered_df = filtered_df[
    (filtered_df["Dividend Yield Numeric"] >= div_min) &
    (filtered_df["Dividend Yield Numeric"] <= div_max)
]

# Filter by Beta
filtered_df = filtered_df[
    (filtered_df["Beta"] >= beta_min) &
    (filtered_df["Beta"] <= beta_max)
]

# Filter by Revenue Growth
filtered_df["Revenue Growth Numeric"] = filtered_df["Revenue Growth"].str.replace("%", "").astype(float)
filtered_df = filtered_df[
    (filtered_df["Revenue Growth Numeric"] >= rev_growth_min) &
    (filtered_df["Revenue Growth Numeric"] <= rev_growth_max)
]

# Filter by Debt/Equity
filtered_df["Debt/Equity Numeric"] = filtered_df["Debt/Equity"].astype(float)
filtered_df = filtered_df[
    (filtered_df["Debt/Equity Numeric"] >= debt_equity_min) &
    (filtered_df["Debt/Equity Numeric"] <= debt_equity_max)
]

# Filter by ROE
filtered_df["ROE Numeric"] = filtered_df["ROE"].str.replace("%", "").astype(float)
filtered_df = filtered_df[
    (filtered_df["ROE Numeric"] >= roe_min) &
    (filtered_df["ROE Numeric"] <= roe_max)
]

# Filter by P/S Ratio
filtered_df["P/S Ratio Numeric"] = filtered_df["P/S Ratio"].astype(float)
filtered_df = filtered_df[
    (filtered_df["P/S Ratio Numeric"] >= ps_min) &
    (filtered_df["P/S Ratio Numeric"] <= ps_max)
]

# Drop temporary numeric columns
filtered_df = filtered_df.drop(columns=["Market Cap Numeric", "Dividend Yield Numeric", "Revenue Growth Numeric", "Debt/Equity Numeric", "ROE Numeric", "P/S Ratio Numeric"])

# Sort options
sort_column = st.selectbox("Sort By:", ["Ticker", "Market Cap", "P/E Ratio", "P/B Ratio", "Dividend Yield", "Beta", "Revenue Growth", "Debt/Equity", "ROE", "P/S Ratio"])
sort_order = st.radio("Sort Order:", ["Ascending", "Descending"])
sort_ascending = True if sort_order == "Ascending" else False

# Convert columns for proper sorting
if sort_column == "Market Cap":
    filtered_df["Market Cap Sort"] = filtered_df["Market Cap"].apply(
        lambda x: float(x.replace("$", "").replace("T", "e12").replace("B", "e9").replace("M", "e6"))
    )
    sort_key = "Market Cap Sort"
elif sort_column == "Dividend Yield":
    filtered_df["Dividend Yield Sort"] = filtered_df["Dividend Yield"].str.replace("%", "").astype(float)
    sort_key = "Dividend Yield Sort"
elif sort_column == "Revenue Growth":
    filtered_df["Revenue Growth Sort"] = filtered_df["Revenue Growth"].str.replace("%", "").astype(float)
    sort_key = "Revenue Growth Sort"
elif sort_column == "Debt/Equity":
    filtered_df["Debt/Equity Sort"] = filtered_df["Debt/Equity"].astype(float)
    sort_key = "Debt/Equity Sort"
elif sort_column == "ROE":
    filtered_df["ROE Sort"] = filtered_df["ROE"].str.replace("%", "").astype(float)
    sort_key = "ROE Sort"
elif sort_column == "P/S Ratio":
    filtered_df["P/S Ratio Sort"] = filtered_df["P/S Ratio"].astype(float)
    sort_key = "P/S Ratio Sort"
else:
    sort_key = sort_column

# Apply sorting
filtered_df = filtered_df.sort_values(by=sort_key, ascending=sort_ascending)

# Drop temporary sort columns if they exist
columns_to_drop = ["Market Cap Sort", "Dividend Yield Sort", "Revenue Growth Sort", "Debt/Equity Sort", "ROE Sort", "P/S Ratio Sort"]
filtered_df = filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns])

# Display results
st.markdown("### Filtered Stocks")
if not filtered_df.empty:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(help="Stock ticker symbol"),
            "Company Name": st.column_config.TextColumn(help="Name of the company"),
            "Sector": st.column_config.TextColumn(help="Industry sector"),
            "Market Cap": st.column_config.TextColumn("Market Cap", help="Total market capitalization"),
            "P/E Ratio": st.column_config.NumberColumn("P/E Ratio", help="Price-to-Earnings ratio"),
            "P/B Ratio": st.column_config.NumberColumn("P/B Ratio", help="Price-to-Book ratio"),
            "Dividend Yield": st.column_config.TextColumn("Dividend Yield", help="Annual dividend yield percentage"),
            "Beta": st.column_config.NumberColumn("Beta", help="Measure of stock volatility relative to the market"),
            "Revenue Growth": st.column_config.TextColumn("Revenue Growth", help="Annual revenue growth percentage"),
            "Debt/Equity": st.column_config.TextColumn("Debt/Equity", help="Debt-to-Equity ratio"),
            "ROE": st.column_config.TextColumn("ROE", help="Return on Equity percentage"),
            "P/S Ratio": st.column_config.TextColumn("P/S Ratio", help="Price-to-Sales ratio")
        }
    )

    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Stocks as CSV",
        data=csv,
        file_name=f"stock_screener_results_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.warning("No stocks match the selected criteria. Try adjusting the filters.")