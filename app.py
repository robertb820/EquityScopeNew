# Import Section
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import requests
import time
import logging
import socket
import json
import os
import smtplib
from email.mime.text import MIMEText

# Email Configuration
FROM_EMAIL = "coderedsupps@gmail.com"  # Replace with your Gmail address
PASSWORD = "ygcr hggh infk rvie"  # Replace with your Gmail password or App Password

def send_email(to_email, subject, content):
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM_EMAIL, PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def get_verification_email_content(email):
    return f"""
<html>
<body>
<h2>Stock Lookup Service Verification</h2>
<p>Thank you for using our stock lookup service, {email}!</p>
<p>You now have access to unlimited stock lookups.</p>
<p>Best regards,<br>
Stock Lookup Team</p>
</body>
</html>
"""

def get_welcome_email_content(email):
    return f"""
<html>
<body>
<h2>Welcome to Stock Lookup Service</h2>
<p>Dear {email},</p>
<p>We're glad you're interested in our service.</p>
<p>Best regards,<br>
Stock Lookup Team</p>
</body>
</html>
"""

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="EquityScope: Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state at the top
if "user_email" not in st.session_state:
    logger.info("Initializing user_email in session state to None")
    st.session_state.user_email = None
if "email_submitted" not in st.session_state:
    st.session_state.email_submitted = False
if "stock_lookups" not in st.session_state:
    logger.info("Initializing stock_lookups in session state to 0")
    st.session_state.stock_lookups = 0
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
if "api_requests_made" not in st.session_state:
    st.session_state.api_requests_made = 0
if "last_request_date" not in st.session_state:
    st.session_state.last_request_date = datetime.now().date()

# Clear cache on startup to avoid stale data
st.session_state.cache = {}

# Reset API request counter
st.session_state.api_requests_made = 0
st.session_state.last_request_date = datetime.now().date()

# Reddit API setup
reddit = praw.Reddit(
    client_id="V3rxmA_qYIzBNTNW79LWIg",
    client_secret="xKBA3Nx7f7VQS0fXnOgmJhYZOmGasA",
    user_agent="python:EquityScopeBot:v1.0"
)

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "4OEV4A0EMNBMJP78"

# Test network connectivity to Alpha Vantage
def test_connectivity():
    try:
        socket.create_connection(("www.alphavantage.co", 443), timeout=5)
        return True, "Successfully connected to Alpha Vantage."
    except Exception as e:
        return False, f"Failed to connect to Alpha Vantage: {str(e)}"

# Validate Alpha Vantage API key
def validate_alpha_vantage_api_key():
    test_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol=IBM&apikey=4OEV4A0EMNBMJP78"
    try:
        response = requests.get(test_url, timeout=30)
        logger.info(f"Alpha Vantage API key validation response: {response.status_code}")
        logger.debug(f"API key validation response content: {response.text}")
        if response.status_code == 200:
            data = response.json()
            if "Note" in data or "Information" in data:
                return False, "Rate limit exceeded or free tier limit reached."
            if "Error Message" in data or not data:
                return False, "Invalid Alpha Vantage API key."
            return True, "API key is valid."
        else:
            return False, f"HTTP Error {response.status_code}: Unable to validate API key."
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        return False, f"Error validating API key: {str(e)}"

# Check connectivity and API key at startup with error handling
try:
    conn_success, conn_message = test_connectivity()
    if not conn_success:
        st.error(f"Connectivity Issue: {conn_message}")
        st.markdown("**Next Steps**: Ensure you're connected to the internet, disable any VPN/firewall, or fix DNS issues (e.g., use Google's DNS: 8.8.8.8).")
except Exception as e:
    st.error(f"Error checking connectivity: {str(e)}")

try:
    is_valid, validation_message = validate_alpha_vantage_api_key()
    if not is_valid:
        st.error(f"Alpha Vantage API Key Issue: {validation_message}")
    else:
        st.success("Alpha Vantage API key validated successfully!")
except Exception as e:
    st.error(f"Error validating API key: {str(e)}")

# Retry decorator with exponential backoff
def retry_with_backoff(func, max_attempts=5, initial_delay=5, backoff_factor=2):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_attempts):
            try:
                result = func(*args, **kwargs)
                if result is None:
                    raise ValueError("API returned None")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning(f"Attempt {attempt+1} failed for {func.__name__}: {str(e)}")
                if "429" in error_msg or "too many requests" in error_msg:
                    logger.info(f"Rate limit hit for {func.__name__}, waiting {delay} seconds...")
                elif "json" in error_msg or "expecting value" in error_msg:
                    logger.info(f"JSON parsing issue for {func.__name__}, possibly empty response...")
                if attempt == max_attempts - 1:
                    logger.error(f"Max attempts reached for {func.__name__}: {str(e)}")
                    return None, str(e)
                time.sleep(delay)
                delay *= backoff_factor
        return None, "Max retries exceeded"
    return wrapper

# Alpha Vantage request counter
def check_alpha_vantage_limit():
    today = datetime.now().date()
    if st.session_state.last_request_date != today:
        st.session_state.api_requests_made = 0
        st.session_state.last_request_date = today
    if st.session_state.api_requests_made >= 25:  # Free tier limit
        return False, "Daily limit of 25 requests reached for Alpha Vantage."
    return True, ""
# Cache stock info
@retry_with_backoff
def get_stock_info_alpha_vantage(ticker):
    can_proceed, limit_message = check_alpha_vantage_limit()
    if not can_proceed:
        return None, limit_message
    logger.info(f"Fetching stock info for {ticker} from Alpha Vantage")
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey=4OEV4A0EMNBMJP78"
        response = requests.get(url, timeout=30)
        logger.debug(f"Stock info response status for {ticker}: {response.status_code}")
        logger.debug(f"Stock info response content: {response.text}")
        st.session_state.api_requests_made += 1
        if response.status_code == 200:
            data = response.json()
            if "Symbol" in data:
                # Standardize market cap format
                if "MarketCapitalization" in data:
                    data["marketCap"] = float(data["MarketCapitalization"].replace(",", "").replace("$", "")) if data["MarketCapitalization"] else 0
                return data, None
            elif "Note" in data or "Information" in data:
                return None, "Alpha Vantage rate limit exceeded or free tier limit reached."
            else:
                return None, "No data returned from Alpha Vantage."
        else:
            return None, f"Alpha Vantage HTTP Error {response.status_code}"
    except Exception as e:
        logger.error(f"Alpha Vantage error for {ticker}: {str(e)}")
        return None, f"Alpha Vantage error: {str(e)}"

@retry_with_backoff
def get_stock_info_yfinance(ticker):
    logger.info(f"Fetching stock info for {ticker} from yfinance")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or "symbol" not in info:
            return None, "No data returned from yfinance."
        # Standardize market cap format
        if "marketCap" in info:
            info["marketCap"] = float(info["marketCap"]) if info["marketCap"] else 0
        return info, None
    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {str(e)}")
        return None, f"yfinance error: {str(e)}"

def get_stock_info(ticker):
    cache_key = f"info_{ticker}"
    if cache_key not in st.session_state.cache:
        # Try yfinance first
        info, error = get_stock_info_yfinance(ticker)
        if info and isinstance(info, dict):  # Ensure info is a dictionary
            st.session_state.cache[cache_key] = info
        else:
            logger.warning(f"yfinance failed for {ticker}: {error}")
            # Try Alpha Vantage fallback
            info, error = get_stock_info_alpha_vantage(ticker)
            if info and isinstance(info, dict):  # Ensure info is a dictionary
                st.session_state.cache[cache_key] = info
            else:
                logger.error(f"Alpha Vantage also failed for {ticker}: {error}")
                return None, f"Could not fetch stock info: {error}"
    return st.session_state.cache[cache_key], None


# Cache stock history
@retry_with_backoff
def get_stock_history_alpha_vantage(ticker, period):
    can_proceed, limit_message = check_alpha_vantage_limit()
    if not can_proceed:
        return None, limit_message
    logger.info(f"Fetching stock history for {ticker} - {period} from Alpha Vantage")
    try:
        time_series_map = {
            "1d": ("TIME_SERIES_INTRADAY", "5min", "1D"),
            "5d": ("TIME_SERIES_DAILY", None, "1W"),
            "1mo": ("TIME_SERIES_DAILY", None, "1M"),
            "6mo": ("TIME_SERIES_DAILY", None, "6M"),
            "1y": ("TIME_SERIES_DAILY", None, "1Y"),
            "5y": ("TIME_SERIES_WEEKLY", None, "5Y"),
            "10y": ("TIME_SERIES_WEEKLY", None, "10Y"),
            "max": ("TIME_SERIES_WEEKLY", None, "All")
        }
        function, interval, display_period = time_series_map.get(period, ("TIME_SERIES_DAILY", None, period))
        if interval:
            url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&interval={interval}&apikey=4OEV4A0EMNBMJP78"
        else:
            url = f"https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey=4OEV4A0EMNBMJP78"
        response = requests.get(url, timeout=30)
        logger.debug(f"Stock history response status for {ticker} ({period}): {response.status_code}")
        logger.debug(f"Stock history response content: {response.text}")
        st.session_state.api_requests_made += 1
        if response.status_code == 200:
            data = response.json()
            time_series_key = f"Time Series ({interval})" if interval else "Time Series" if function == "TIME_SERIES_DAILY" else "Weekly Time Series"
            if time_series_key in data:
                df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={
                    "4. close": "Close",
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "5. volume": "Volume"
                })
                df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                return df.sort_index(), None
            elif "Note" in data or "Information" in data:
                return None, "Alpha Vantage rate limit exceeded or free tier limit reached."
            else:
                return None, "No data returned from Alpha Vantage."
        else:
            return None, f"Alpha Vantage HTTP Error {response.status_code}"
    except Exception as e:
        logger.error(f"Alpha Vantage history error for {ticker}: {str(e)}")
        return None, f"Alpha Vantage error: {str(e)}"

@retry_with_backoff
def get_stock_history_yfinance(ticker, period):
    logger.info(f"Fetching stock history for {ticker} - {period} from yfinance")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None, "No data returned from yfinance."
        if "Adj Close" in hist.columns:
            hist = hist.drop(columns=["Adj Close"])
        return hist, None
    except Exception as e:
        logger.error(f"yfinance history error for {ticker}: {str(e)}")
        return None, f"yfinance error: {str(e)}"

def get_stock_history(ticker, period):
    cache_key = f"history_{ticker}_{period}"
    if cache_key not in st.session_state.cache:
        # Try Alpha Vantage first
        hist, error = get_stock_history_alpha_vantage(ticker, period)
        if hist is not None:
            st.session_state.cache[cache_key] = hist
        else:
            logger.warning(f"Alpha Vantage failed for {ticker} history: {error}")
            # Fallback to yfinance
            hist, error = get_stock_history_yfinance(ticker, period)
            if hist is not None:
                st.session_state.cache[cache_key] = hist
            else:
                logger.error(f"yfinance also failed for {ticker} history: {error}")
                return None, f"Could not fetch stock history: {error}"
    return st.session_state.cache[cache_key], None

import streamlit as st
import smtplib
from email.mime.text import MIMEText
import logging
import os

# Configure logging
if not os.path.exists('email.log'):
    with open('email.log', 'w') as f:
        f.write('')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='email.log'
)
logger = logging.getLogger(__name__)

# Email Configuration
FROM_EMAIL = "coderedsupps@gmail.com"
PASSWORD = "ygcr hggh infk rvie"

def send_email(to_email, subject, content):
    msg = MIMEText(content, 'html')  # Specify HTML content type
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email
    
    try:
        logger.info(f"Attempting to send email to {to_email}")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(FROM_EMAIL, PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Error sending email to {to_email}: {e}")
        return False

def get_verification_email_content(email):
    return f"""
<html>
<body>
<h2>EquityScope Service Verification</h2>
<p>Thank you for using EquityScope, {email}!</p>
<p>You now have access to unlimited stock lookups.</p>
<p>Best regards,<br>
EquityScope Team</p>
</body>
</html>
"""

def get_welcome_email_content(email):
    return f"""
<html>
<body>
<h2>Welcome to EquityScope</h2>
<p>Dear {email},</p>
<p>We're glad you're interested in our service.</p>
<p>Best regards,<br>
Stock Lookup Team</p>
</body>
</html>
"""




# Streamlit page title
st.title("📈 EquityScope: Stock Analyzer")

# Sidebar Theme Toggle
st.sidebar.title("Settings")
# Ensure theme is initialized
if "theme" not in st.session_state:
    st.session_state.theme = "light"
# Let user choose theme
selected_theme = st.sidebar.radio("Select Theme:", ["Light", "Dark"],
                                index=0 if st.session_state.theme == "light" else 1)
st.session_state.theme = selected_theme.lower()

# Define themes
themes = {
    "light": {
        "bg": "#FFFFFF",
        "text": "#111827",
        "plot_bg": "rgba(240, 240, 240, 0.5)",
        "grid": "rgba(150, 150, 150, 0.5)",
        "line": "#00A8E8",
        "sma20": "#FF6F61",
        "sma50": "#6B7280",
        "sma200": "#34D399",
        "bar_colors": ["#00C4B4", "#FF6F61", "#F4A261", "#34D399", "#6B7280", "#A78BFA", "#EC4899", "#EF4444"],
        "calc_header": "#00A8E8",
        "header": "#000000",
        "input_bg": "#F3F4F6",
        "button_bg": "#E5E7EB",
        "button_text": "#111827"
    },
    "dark": {
        "bg": "#1F2937",
        "text": "#FFFFFF",
        "plot_bg": "rgba(31, 41, 55, 0.8)",
        "grid": "rgba(107, 114, 128, 0.5)",
        "line": "#60A5FA",
        "sma20": "#F87171",
        "sma50": "#9CA3AF",
        "sma200": "#34D399",
        "bar_colors": ["#2DD4BF", "#F87171", "#FBBF24", "#34D399", "#9CA3AF", "#C4B5FD", "#F472B6", "#F87171"],
        "calc_header": "#60A5FA",
        "header": "#FFFFFF",
        "input_bg": "#374151",
        "button_bg": "#4B5563",
        "button_text": "#FFFFFF"
    }
}

# Set styles in session
st.session_state.theme_styles = themes[st.session_state.theme]

# Apply CSS theme styles
def apply_custom_theme_styles():
    styles = st.session_state.theme_styles
    st.markdown(
        f"""
        <style>
            .main {{
                background-color: {styles["bg"]};
                color: {styles["text"]};
            }}
            
            /* Dropdown visibility fixes */
            .stSelectbox > div {{
                background-color: inherit !important;
            }}
            
            [data-baseweb="select"] {{
                color: #000000 !important;
                background-color: #ffffff !important;
            }}
            
            [data-theme="dark"] [data-baseweb="select"] {{
                color: #ffffff !important;
                background-color: #374151 !important;
            }}
            
            .stSelectbox div div {{
                background-color: #ffffff !important;
                color: #000000 !important;
            }}
            
            [data-theme="dark"] .stSelectbox div div {{
                background-color: #374151 !important;
                color: #ffffff !important;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                color: {styles["header"]} !important;
            }}
            
            input[type="text"], 
            input[type="number"], 
            textarea {{
                background-color: {styles["input_bg"]} !important;
                color: {styles["text"]} !important;
            }}
            
            .stTextInput > div > input {{
                background-color: {styles["input_bg"]} !important;
                color: {styles["text"]} !important;
            }}
            
            .stDownloadButton > button,
            .stButton > button {{
                background-color: {styles["button_bg"]} !important;
                color: {styles["button_text"]} !important;
                border: none;
                border-radius: 5px;
            }}
            
            .stDownloadButton > button:hover,
            .stButton > button:hover {{
                opacity: 0.9;
                cursor: pointer;
            }}
            
            .stRadio > div > label,
            .stSelectbox > div,
            .stMarkdown, 
            .stText, 
            label, 
            div, 
            span {{
                color: {styles["text"]} !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_theme_styles()

# Additional global style adjustments
st.markdown(f"""
<style>
    .stApp {{
        background-color: {st.session_state.theme_styles['bg']} !important;
        color: {st.session_state.theme_styles['text']} !important;
    }}
    
    .stMarkdown, 
    .stRadio > label, 
    .stAlert, 
    .company-info, 
    .description, 
    .metric-value, 
    .calc-header, 
    .st-expander {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    
    .calc-header {{
        color: {st.session_state.theme_styles['calc_header']} !important;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {'#F9FAFB' if st.session_state.theme == 'light' else '#374151'} !important;
    }}
    
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] h1 {{
        color: {st.session_state.theme_styles['text']} !important;
        opacity: 1 !important;
    }}
    
    [data-testid="stDataFrame"] a {{
        color: {st.session_state.theme_styles['text']} !important;
        text-decoration: underline !important;
    }}
    
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .ticktext,
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-xtitle,
    .stApp[style*="background-color: #FFFFFF"] .js-plotly-plot .plotly .g-ytitle {{
        fill: #000000 !important;
        color: #000000 !important;
    }}
</style>
""",
unsafe_allow_html=True)

# Inject theme styles
apply_custom_theme_styles()

def get_verification_email_content(email):
    return f"""
    <html>
        <body>
            <h2>EquityScope Service Verification</h2>
            <p>Thank you for using EquityScope, {email}!</p>
            <p>You now have access to unlimited stock lookups.</p>
            <p>Best regards,<br>
            EquityScope Team</p>
        </body>
    </html>
    """

def get_welcome_email_content(email):
    return f"""
    <html>
        <body>
            <h2>Welcome to EquityScope</h2>
            <p>Dear {email},</p>
            <p>We're glad you're interested in our service.</p>
            <p>Best regards,<br>
            EquityScope Team</p>
        </body>
    </html>
    """
# Debug Section
st.sidebar.subheader("Debug Info")
if st.sidebar.checkbox("Show Debug Logs"):
    st.sidebar.markdown("Check your terminal for detailed logs.")
st.sidebar.markdown(f"Alpha Vantage Requests Made Today: {st.session_state.api_requests_made}/25")

# Manual cache reset
if st.sidebar.button("Clear Cache"):
    st.session_state.cache = {}
    st.session_state.api_requests_made = 0
    st.experimental_rerun()

# Company Info with Email Feature
st.subheader("🔍 Company Information")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Enter a stock ticker to analyze (e.g., AAPL for Apple).</p>', unsafe_allow_html=True)

def market_cap_display(market_cap):
    if isinstance(market_cap, (int, float)):
        if market_cap >= 1_000_000_000_000:
            return f"${market_cap / 1_000_000_000_000:.2f}T"
        elif market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        else:
            return f"${market_cap / 1_000_000:.2f}M"
    return "N/A"

stock_ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="AAPL").strip().upper()

if stock_ticker:
    logger.info(f"Attempting stock lookup for {stock_ticker}, current lookups: {st.session_state.stock_lookups}")
    
    # Check if user has reached limit and hasn't submitted email
    if st.session_state.stock_lookups > 3 and not st.session_state.email_submitted:
        st.error("You have reached the limit of 3 free stock lookups.")
        st.markdown("To request more lookups, please provide your email address here:")
        email_input = st.text_input("Enter your email address:", key="email_input")
        
        if st.button("Submit Email"):
            if email_input and "@" in email_input and "." in email_input:
                st.session_state.user_email = email_input
                st.session_state.email_submitted = True
                logger.info(f"User submitted email: {email_input} for more lookups")
                
                # Send welcome email
                success = send_email(
                    to_email=email_input,
                    subject="Welcome to EquityScope",
                    content=get_welcome_email_content(email_input)
                )
                
                if success:
                    st.success(f"Welcome email sent to {email_input}! Check your inbox (and spam folder).")
                else:
                    st.error("Failed to send welcome email. Please check logs for details.")
            else:
                st.error("Please enter a valid email address.")
            st.stop()
        
        # Stop here if user hasn't submitted email
        st.stop()
    
    # Only allow stock lookup if user is under limit or has submitted email
    with st.spinner("Fetching company info..."):
        stock_info, error = get_stock_info(stock_ticker)
        if stock_info:
            st.session_state.stock_lookups += 1
            st.session_state.stock_info = stock_info
            st.session_state.ticker = stock_ticker
            logger.info(f"Lookup successful, incremented to: {st.session_state.stock_lookups}")
            
            company_name = stock_info.get("Name", stock_info.get("longName", "N/A"))
            sector = stock_info.get("Sector", stock_info.get("sector", "N/A"))
            industry = stock_info.get("Industry", stock_info.get("industry", "N/A"))
            market_cap = float(stock_info.get("MarketCapitalization", stock_info.get("marketCap", 0)))
            summary = stock_info.get("Description", stock_info.get("longBusinessSummary", "No description available."))
            website = stock_info.get("Website", stock_info.get("website", "N/A"))
            
            st.markdown(f'<p class="company-info"><strong>Company:</strong> {company_name} | <strong>Sector:</strong> {sector} | <strong>Industry:</strong> {industry} | <strong>Market Cap:</strong> {market_cap_display(market_cap)} | <strong>Website:</strong> <a href="{website}" target="_blank">{website}</a></p>', unsafe_allow_html=True)
            st.markdown(f'<p class="description">{summary}</p>', unsafe_allow_html=True)
        else:
            st.error(f"Could not fetch company info for {stock_ticker}: {error}")
            st.markdown("Check your terminal logs for detailed error messages. Ensure you're connected to the internet, or try clearing the cache using the button in the sidebar.")

    with st.spinner("Fetching company info..."):
        stock_info, error = get_stock_info(stock_ticker)
        if stock_info:
            st.session_state.stock_lookups += 1
            st.session_state.stock_info = stock_info
            st.session_state.ticker = stock_ticker
            logger.info(f"Lookup successful, incremented to: {st.session_state.stock_lookups}")
            company_name = stock_info.get("Name", stock_info.get("longName", "N/A"))
            sector = stock_info.get("Sector", stock_info.get("sector", "N/A"))
            industry = stock_info.get("Industry", stock_info.get("industry", "N/A"))
            market_cap = float(stock_info.get("MarketCapitalization", stock_info.get("marketCap", 0)))
            summary = stock_info.get("Description", stock_info.get("longBusinessSummary", "No description available."))
            website = stock_info.get("Website", stock_info.get("website", "N/A"))
            st.markdown(
                f'<p class="company-info"><strong>Company:</strong> {company_name} | '
                f'<strong>Sector:</strong> {sector} | '
                f'<strong>Industry:</strong> {industry} | '
                f'<strong>Market Cap:</strong> {market_cap_display(market_cap)} | '
                f'<strong>Website:</strong> <a href="{website}" target="_blank">{website}</a></p>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p class="description">{summary}</p>', unsafe_allow_html=True)
        else:
            st.error(f"Could not fetch company info for {stock_ticker}: {error}")
            st.markdown("**Next Steps**: Check the terminal logs for detailed error messages. Ensure you're connected to the internet, or try clearing the cache using the button in the sidebar.")
else:
    st.session_state.stock_info = None
    st.session_state.ticker = None

# Key Metrics
st.subheader("📊 Key Metrics")
if stock_ticker:
    with st.spinner("Fetching key metrics..."):
        stock_info, error = get_stock_info(stock_ticker)
        if stock_info and isinstance(stock_info, dict):  # Ensure stock_info is a dictionary
            metrics = {}
            # Safely extract metrics with default values
            pe_ratio = stock_info.get("PERatio", stock_info.get("trailingPE", "N/A"))
            metrics["P/E Ratio"] = float(pe_ratio) if pe_ratio != "N/A" and pe_ratio is not None else "N/A"
            
            pb_ratio = stock_info.get("PriceToBookRatio", stock_info.get("priceToBook", "N/A"))
            metrics["P/B Ratio"] = float(pb_ratio) if pb_ratio != "N/A" and pb_ratio is not None else "N/A"
            
            dividend_yield = stock_info.get("DividendYield", stock_info.get("dividendYield", 0))
            metrics["Dividend Yield"] = float(dividend_yield) * 100 if dividend_yield != "N/A" and dividend_yield is not None else 0
            
            beta = stock_info.get("Beta", stock_info.get("beta", "N/A"))
            metrics["Beta"] = float(beta) if beta != "N/A" and beta is not None else "N/A"
            
            roe = stock_info.get("ReturnOnEquityTTM", stock_info.get("returnOnEquity", "N/A"))
            metrics["ROE"] = float(roe) * 100 if roe != "N/A" and roe is not None else "N/A"
            
            debt_equity = stock_info.get("DebtToEquity", stock_info.get("debtToEquity", "N/A"))
            metrics["Debt/Equity"] = float(debt_equity) if debt_equity != "N/A" and debt_equity is not None else "N/A"
            
            ps_ratio = stock_info.get("PriceToSalesRatioTTM", stock_info.get("priceToSalesTrailing12Months", "N/A"))
            metrics["P/S Ratio"] = float(ps_ratio) if ps_ratio != "N/A" and ps_ratio is not None else "N/A"
            
            # Format metrics for display
            metrics_display = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    if key in ["Dividend Yield", "ROE"]:
                        metrics_display.append(f"{key}: {value:.2f}%")
                    else:
                        metrics_display.append(f"{key}: {value:.2f}")
                else:
                    metrics_display.append(f"{key}: N/A")
            st.markdown(
                f'<p class="metric-value">{" | ".join(metrics_display)}</p>',
                unsafe_allow_html=True
            )
        else:
            st.warning(f"Could not fetch key metrics for {stock_ticker}: {error}")

# Price History
st.subheader("📈 Price History")
time_frame_options = ["1D", "1W", "1M", "6M", "1Y", "5Y", "10Y", "All"]
time_frame_map = {"1D": "1d", "1W": "5d", "1M": "1mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "10Y": "10y", "All": "max"}
selected_time_frame = st.selectbox("Select time frame:", time_frame_options, index=4)
selected_period = time_frame_map[selected_time_frame]

if stock_ticker:
    with st.spinner(f"Fetching price history for {selected_time_frame}..."):
        hist, error = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color=st.session_state.theme_styles["line"], width=2)
            ))
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.3,
                marker_color=st.session_state.theme_styles["sma50"]
            ))
            fig.update_layout(
                title=dict(
                    text=f"{stock_ticker} Price and Volume ({selected_time_frame})",
                    font=dict(size=20, color=axis_text_color),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    tickfont=dict(color=axis_text_color, size=12)
                ),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
                font=dict(family="Arial", size=12, color=axis_text_color),
                legend=dict(
                    x=0,
                    y=1.1,
                    orientation="h",
                    font=dict(color=axis_text_color)
                ),
                xaxis=dict(
                    title=dict(text="Date", font=dict(color=axis_text_color, size=14)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text="Price (USD)", font=dict(color=axis_text_color, size=14)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not fetch price history for {stock_ticker}: {error}")
# Moving Averages
st.subheader("📈 Moving Averages")
if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: 50-day and 200-day SMAs may be less reliable for short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating moving averages..."):
        hist, error = get_stock_history(stock_ticker, period=selected_period)
        if hist is not None and not hist.empty:
            sma_20 = hist["Close"].rolling(window=20).mean()
            sma_50 = hist["Close"].rolling(window=50).mean()
            sma_200 = hist["Close"].rolling(window=200).mean()
            axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color=st.session_state.theme_styles["line"], width=2)
            ))
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=sma_20,
                mode="lines",
                name="20-day SMA",
                line=dict(color=st.session_state.theme_styles["sma20"], width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=sma_50,
                mode="lines",
                name="50-day SMA",
                line=dict(color=st.session_state.theme_styles["sma50"], width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=sma_200,
                mode="lines",
                name="200-day SMA",
                line=dict(color=st.session_state.theme_styles["sma200"], width=1.5)
            ))
            fig.update_layout(
                title=dict(
                    text=f"{stock_ticker} Moving Averages ({selected_time_frame})",
                    font=dict(size=20, color=axis_text_color),
                    x=0.5,
                    xanchor="center"
                ),
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1.1,
                    orientation="h",
                    font=dict(color=axis_text_color)
                ),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title=dict(text="Date", font=dict(color=axis_text_color, size=14)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                yaxis=dict(
                    title=dict(text="Price (USD)", font=dict(color=axis_text_color, size=14)),
                    tickfont=dict(family="Arial", size=12, color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Moving Averages Explanation**: SMAs smooth price data to identify trends. A golden cross (50-day SMA crossing above 200-day SMA) is bullish, while a death cross (50-day SMA crossing below 200-day SMA) is bearish.")
        else:
            st.warning(f"Could not fetch price history for moving averages: {error}")

# Technical Indicators Graph
st.subheader("📉 Technical Indicators Graph")

if selected_time_frame in ["1D", "1W", "1M"]:
    st.warning("Note: Bollinger Bands and MACD may be less reliable for short time frames.")

if stock_ticker:
    with st.spinner(f"Calculating technical indicators..."):
        hist, error = get_stock_history(stock_ticker, period=selected_period)

        if hist is not None and not hist.empty:
            # Calculate technical indicators
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

            axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,  # Increased spacing between graphs
                subplot_titles=(
                    f"{stock_ticker} Price with Bollinger Bands",
                    "MACD (Moving Average Convergence Divergence)"
                ),
                row_heights=[0.65, 0.35]
            )

            # Price and Bollinger Bands
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color=st.session_state.theme_styles["line"], width=2)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=upper_band,
                mode="lines",
                name="Upper Band",
                line=dict(color=st.session_state.theme_styles["sma50"], width=1, dash="dash")
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=lower_band,
                mode="lines",
                name="Lower Band",
                line=dict(color=st.session_state.theme_styles["sma50"], width=1, dash="dash")
            ), row=1, col=1)

            # MACD and Histogram
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=macd,
                mode="lines",
                name="MACD",
                line=dict(color=st.session_state.theme_styles["line"], width=1.5)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=hist.index,
                y=signal,
                mode="lines",
                name="Signal Line",
                line=dict(color=st.session_state.theme_styles["sma20"], width=1.5)
            ), row=2, col=1)

            fig.add_trace(go.Bar(
                x=hist.index,
                y=histogram,
                name="Histogram",
                marker_color=st.session_state.theme_styles["sma50"]
            ), row=2, col=1)

            # Final Layout
            fig.update_layout(
                height=700,
                title=dict(
                    text=f"{stock_ticker} Technical Indicators ({selected_time_frame})",
                    font=dict(size=20, color=axis_text_color),
                    x=0.5,
                    xanchor="center",
                    y=0.95
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="left",
                    x=0,
                    font=dict(color=axis_text_color)
                ),
                margin=dict(t=100, b=60, l=60, r=60),
                plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Arial", size=12, color=axis_text_color),
                xaxis=dict(
                    title="Date",
                    title_font=dict(color=axis_text_color, size=14),
                    tickfont=dict(color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                xaxis2=dict(
                    title="Date",
                    title_font=dict(color=axis_text_color, size=14),
                    tickfont=dict(color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                yaxis=dict(
                    title="Price (USD)",
                    title_font=dict(color=axis_text_color, size=14),
                    tickfont=dict(color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                ),
                yaxis2=dict(
                    title="MACD",
                    title_font=dict(color=axis_text_color, size=14),
                    tickfont=dict(color=axis_text_color),
                    gridcolor=st.session_state.theme_styles["grid"],
                    zeroline=False
                )
            )

            # Display chart
            st.plotly_chart(fig, use_container_width=True)

            # Explanations
            st.markdown("**Bollinger Bands**: When the price approaches the upper band, it may indicate overbought conditions. Near the lower band suggests oversold conditions.")
            st.markdown("**MACD**: A bullish signal occurs when the MACD crosses above the signal line. A bearish signal occurs when it crosses below.")

        else:
            st.warning(f"Could not fetch price history for technical indicators: {error}")


# Technical Indicators (RSI)
st.subheader("📉 Technical Indicators")
if selected_time_frame in ["1D", "1W"]:
    st.warning("Note: RSI may be less reliable for very short time frames.")
if stock_ticker:
    with st.spinner(f"Calculating RSI..."):
        hist, error = get_stock_history(stock_ticker, period=selected_period)
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
            st.warning(f"Could not fetch price history for RSI: {error}")

# Valuation Section
st.subheader("💰 Valuation")
st.markdown("Estimate the fair value of the stock using multiple valuation methods.")
if stock_ticker:
    with st.spinner("Calculating valuations..."):
        stock_info, error = get_stock_info(stock_ticker)
        stock = yf.Ticker(stock_ticker)
        try:
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            financials = stock.financials
        except Exception as e:
            logger.error(f"Failed to fetch financial statements for {stock_ticker}: {str(e)}")
            balance_sheet = None
            cash_flow = None
            financials = None
            st.warning(f"Could not fetch financial statements for {stock_ticker}.")
        if stock_info:
            current_price = float(stock_info.get("PreviousClose", stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))))
            eps = float(stock_info.get("EPS", stock_info.get("trailingEps", 0)))
            forward_eps = float(stock_info.get("ForwardEPS", stock_info.get("forwardEps", eps)))
            shares_outstanding = int(stock_info.get("SharesOutstanding", stock_info.get("sharesOutstanding", 0)))
            book_value = float(stock_info.get("BookValue", stock_info.get("bookValue", 0)))
            dividend_rate = float(stock_info.get("DividendRate", stock_info.get("dividendRate", 0)))
            five_year_avg_dividend = float(stock_info.get("FiveYearAvgDividendYield", stock_info.get("fiveYearAvgDividendYield", 0))) / 100
            pe_ratio = float(stock_info.get("PERatio", stock_info.get("trailingPE", 0)))
            growth_rate = min(float(stock_info.get("EarningsGrowth", stock_info.get("earningsGrowth", 0.20))), 0.25)
            beta = float(stock_info.get("Beta", stock_info.get("beta", 1.0)))
            discount_rate = 0.05 + beta * 0.02
            perpetual_growth = 0.03
            total_debt = float(stock_info.get("TotalDebt", stock_info.get("totalDebt", 0)))
            total_cash = float(stock_info.get("TotalCash", stock_info.get("totalCash", 0)))
            revenue = float(stock_info.get("RevenueTTM", stock_info.get("totalRevenue", 0)))
            valuations = []
            calculation_details = []
            # Intrinsic Value (Simple EPS Multiplier)
            try:
                intrinsic_value = eps * 15
                intrinsic_value = f"${intrinsic_value:.2f}" if isinstance(intrinsic_value, (int, float)) else "N/A"
                calc = f"""
                **Intrinsic Value (EPS Multiplier)** 📊
                - **Formula**: EPS × 15
                - **EPS**: ${eps:.2f}
                - **Multiplier**: 15 (a conservative average P/E ratio)
                - **Calculation**: ${eps:.2f} × 15 = {intrinsic_value}
                - **Result**: {intrinsic_value} per share
                **Explanation**: This method multiplies the earnings per share by a standard P/E ratio to estimate a fair value. It’s simple but doesn’t account for growth or risk.
                """
                valuations.append({
                    "Method": "Intrinsic Value (EPS Multiplier)",
                    "Intrinsic Value per Share": intrinsic_value,
                    "Description": "EPS multiplied by 15."
                })
                calculation_details.append(("Intrinsic Value (EPS Multiplier)", calc))
            except:
                valuations.append({
                    "Method": "Intrinsic Value (EPS Multiplier)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "EPS multiplied by 15."
                })
                calculation_details.append(("Intrinsic Value (EPS Multiplier)", "**Why It’s Missing** 🚫\nError in calculation: EPS data unavailable or invalid."))
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
                    **Discounted Cash Flow (DCF)** 📊
                    - **Formula**: Sum of discounted future cash flows + terminal value, adjusted for net debt
                    - **Free Cash Flow (Year 0)**: ${fcf:,.2f} (estimated as 60% of Forward EPS × Shares Outstanding)
                    - **Growth Rates (Years 1-5)**: {[f"{g*100:.1f}%" for g in growth_rates]}
                    - **Discount Rate**: {discount_rate*100:.1f}% (calculated as 5% risk-free rate + Beta × 2%)
                    - **Terminal Value**: ${terminal_value:,.2f} (using perpetual growth rate of {perpetual_growth*100:.1f}%)
                    - **Enterprise Value**: ${enterprise_value:,.2f}
                    - **Net Debt**: ${net_debt:,.2f} (Total Debt - Total Cash)
                    - **Equity Value**: ${equity_value:,.2f}
                    - **Per Share**: ${dcf_value:.2f} (${equity_value:,.2f} ÷ {shares_outstanding:,} shares)
                    **Explanation**: DCF discounts projected future cash flows to present value, accounting for growth, risk, and debt. It’s detailed but sensitive to assumptions.
                    """
                else:
                    dcf_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo cash flow data or missing Forward EPS/Shares Outstanding."
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": dcf_value,
                    "Description": "Discounts future cash flows."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", calc))
            except:
                valuations.append({
                    "Method": "Discounted Cash Flow (DCF)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Discounts future cash flows."
                })
                calculation_details.append(("Discounted Cash Flow (DCF)", "**Why It’s Missing** 🚫\nError in calculation: Data unavailable or invalid."))
            # DDM
            try:
                if dividend_rate and dividend_rate > 0 and forward_eps:
                    expected_dividend = dividend_rate * (1 + 0.10)
                    ddm_value = expected_dividend / (discount_rate - perpetual_growth)
                    ddm_value = f"${ddm_value:.2f}" if isinstance(ddm_value, (int, float)) else "N/A"
                    calc = f"""
                    **Dividend Discount Model (DDM)** 📊
                    - **Formula**: Expected Dividend ÷ (Discount Rate - Growth Rate)
                    - **Current Dividend**: ${dividend_rate:.2f}
                    - **Expected Dividend**: ${expected_dividend:.2f} (assumes 10% dividend growth)
                    - **Discount Rate**: {discount_rate*100:.1f}%
                    - **Perpetual Growth Rate**: {perpetual_growth*100:.1f}%
                    - **Calculation**: ${expected_dividend:.2f} ÷ ({discount_rate*100:.1f}% - {perpetual_growth*100:.1f}%) = {ddm_value}
                    - **Result**: {ddm_value} per share
                    **Explanation**: DDM values the stock based on the present value of future dividends, assuming constant growth. Best for dividend-paying stocks.
                    """
                else:
                    ddm_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo dividend data or invalid Forward EPS."
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": ddm_value,
                    "Description": "Values stock via dividends."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", calc))
            except:
                valuations.append({
                    "Method": "Dividend Discount Model (DDM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Values stock via dividends."
                })
                calculation_details.append(("Dividend Discount Model (DDM)", "**Why It’s Missing** 🚫\nError in calculation: Data unavailable or invalid."))
            # RIM
            try:
                if book_value and forward_eps and shares_outstanding:
                    roe = forward_eps / book_value if book_value != 0 else 0
                    retention_ratio = 1 - (dividend_rate / forward_eps if dividend_rate and forward_eps else 0)
                    residual_income = forward_eps - (discount_rate * book_value)
                    rim_value = book_value + (residual_income * retention_ratio * (1 + growth_rate) / (discount_rate - perpetual_growth))
                    rim_value = f"${rim_value:.2f}" if isinstance(rim_value, (int, float)) else "N/A"
                    calc = f"""
                    **Residual Income Model (RIM)** 📊
                    - **Formula**: Book Value + (Residual Income × Retention Ratio × (1 + Growth Rate)) ÷ (Discount Rate - Growth Rate)
                    - **Book Value per Share**: ${book_value:.2f}
                    - **Forward EPS**: ${forward_eps:.2f}
                    - **ROE**: {roe*100:.1f}% (Forward EPS ÷ Book Value)
                    - **Retention Ratio**: {retention_ratio:.2f} (1 - Dividend Payout Ratio)
                    - **Residual Income**: ${residual_income:.2f} (Forward EPS - (Discount Rate × Book Value))
                    - **Growth Rate**: {growth_rate*100:.1f}%
                    - **Discount Rate**: {discount_rate*100:.1f}%
                    - **Perpetual Growth Rate**: {perpetual_growth*100:.1f}%
                    - **Calculation**: ${book_value:.2f} + (${residual_income:.2f} × {retention_ratio:.2f} × (1 + {growth_rate:.2f})) ÷ ({discount_rate*100:.1f}% - {perpetual_growth*100:.1f}%) = {rim_value}
                    - **Result**: {rim_value} per share
                    **Explanation**: RIM adds the present value of residual income (earnings above required return) to book value. It’s useful for companies with significant book value.
                    """
                else:
                    rim_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo book value, Forward EPS, or Shares Outstanding data."
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": rim_value,
                    "Description": "Uses book value and income."
                })
                calculation_details.append(("Residual Income Model (RIM)", calc))
            except:
                valuations.append({
                    "Method": "Residual Income Model (RIM)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Uses book value and income."
                })
                calculation_details.append(("Residual Income Model (RIM)", "**Why It’s Missing** 🚫\nError in calculation: Data unavailable or invalid."))
            # Graham
            try:
                if forward_eps and growth_rate:
                    graham_value = forward_eps * (10 + 2.5 * growth_rate * 100)
                    graham_value = f"${graham_value:.2f}" if isinstance(graham_value, (int, float)) else "N/A"
                    calc = f"""
                    **Graham Method** 📊
                    - **Formula**: Forward EPS × (10 + 2.5 × Growth Rate × 100)
                    - **Forward EPS**: ${forward_eps:.2f}
                    - **Growth Rate**: {growth_rate*100:.1f}%
                    - **Multiplier**: 10 + 2.5 × {growth_rate*100:.1f} = {10 + 2.5 * growth_rate * 100:.1f}
                    - **Calculation**: ${forward_eps:.2f} × {10 + 2.5 * growth_rate * 100:.1f} = {graham_value}
                    - **Result**: {graham_value} per share
                    **Explanation**: The Graham Method adjusts the P/E ratio based on growth, inspired by Benjamin Graham’s value investing principles.
                    """
                else:
                    graham_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo Forward EPS or growth rate data."
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": graham_value,
                    "Description": "EPS with growth multiplier."
                })
                calculation_details.append(("Graham Method", calc))
            except:
                valuations.append({
                    "Method": "Graham Method",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "EPS with growth multiplier."
                })
                calculation_details.append(("Graham Method", "**Why It’s Missing** 🚫\nError in calculation: Data unavailable or invalid."))
            # Comps (using P/E and P/S)
            try:
                if pe_ratio and revenue and shares_outstanding:
                    industry_pe = 25.0  # Static average for simplicity
                    comps_pe_value = eps * industry_pe
                    industry_ps = 5.0  # Static average for simplicity
                    comps_ps_value = (revenue * industry_ps) / shares_outstanding
                    comps_value = (comps_pe_value + comps_ps_value) / 2
                    comps_value = f"${comps_value:.2f}" if isinstance(comps_value, (int, float)) else "N/A"
                    calc = f"""
                    **Comparable Company Analysis (Comps)** 📊
                    - **Formula**: Average of (EPS × Industry P/E) and (Revenue × Industry P/S ÷ Shares Outstanding)
                    - **EPS**: ${eps:.2f}
                    - **Industry P/E**: {industry_pe:.1f}
                    - **P/E Value**: ${comps_pe_value:.2f} (EPS × Industry P/E)
                    - **Revenue**: ${revenue:,.2f}
                    - **Industry P/S**: {industry_ps:.1f}
                    - **Shares Outstanding**: {shares_outstanding:,}
                    - **P/S Value**: ${comps_ps_value:.2f} (Revenue × Industry P/S ÷ Shares Outstanding)
                    - **Average Value**: (${comps_pe_value:.2f} + ${comps_ps_value:.2f}) ÷ 2 = {comps_value}
                    - **Result**: {comps_value} per share
                    **Explanation**: Comps uses industry averages (P/E and P/S ratios) to estimate value relative to similar companies.
                    """
                else:
                    comps_value = "N/A"
                    calc = "**Why It’s Missing** 🚫\nNo P/E, revenue, or shares outstanding data."
                valuations.append({
                    "Method": "Comparable Company Analysis (Comps)",
                    "Intrinsic Value per Share": comps_value,
                    "Description": "Uses industry P/E and P/S ratios."
                })
                calculation_details.append(("Comparable Company Analysis (Comps)", calc))
            except:
                valuations.append({
                    "Method": "Comparable Company Analysis (Comps)",
                    "Intrinsic Value per Share": "N/A",
                    "Description": "Uses industry P/E and P/S ratios."
                })
                calculation_details.append(("Comparable Company Analysis (Comps)", "**Why It’s Missing** 🚫\nError in calculation: Data unavailable or invalid."))
            # Valuation Table and Chart
            valuation_df = pd.DataFrame(valuations)
            st.markdown(f"**Current Market Price**: ${current_price:.2f}")
            st.markdown("### Valuation Estimates")
            st.dataframe(valuation_df[["Method", "Intrinsic Value per Share", "Description"]], width=1200)
            st.markdown("### Valuation Comparison Chart")
            chart_data = valuation_df[valuation_df["Intrinsic Value per Share"] != "N/A"]
            if not chart_data.empty:
                methods = chart_data["Method"].tolist()
                intrinsic_values = [float(val.replace("$", "")) for val in chart_data["Intrinsic Value per Share"]]
                chart_methods = methods + ["Current Price"]
                chart_values = intrinsic_values + [current_price]
                axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=chart_methods,
                        y=chart_values,
                        marker_color=st.session_state.theme_styles["bar_colors"][:len(chart_methods)],
                        text=[f"${v:.2f}" for v in chart_values],
                        textposition="auto",
                        textfont=dict(color=axis_text_color)
                    )
                )
                fig.add_hline(
                    y=current_price,
                    line_dash="dash",
                    line_color=st.session_state.theme_styles["bar_colors"][-1],
                    annotation_text="Current Price",
                    annotation_position="top right",
                    annotation_font=dict(color=axis_text_color)
                )
                fig.update_layout(
                    title=dict(
                        text=f"{stock_ticker} Intrinsic Value vs. Current Price",
                        font=dict(size=20, color=axis_text_color),
                        x=0.5,
                        xanchor="center"
                    ),
                    xaxis_title="Valuation Method",
                    yaxis_title="Price per Share (USD)",
                    showlegend=False,
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=14, color=axis_text_color),
                    height=500,
                    xaxis=dict(
                        title=dict(text="Valuation Method", font=dict(color=axis_text_color, size=14)),
                        tickfont=dict(family="Arial", size=12, color=axis_text_color),
                        tickangle=45,
                        gridcolor=st.session_state.theme_styles["grid"],
                        zeroline=False
                    ),
                    yaxis=dict(
                        title=dict(text="Price per Share (USD)", font=dict(color=axis_text_color, size=14)),
                        tickfont=dict(family="Arial", size=12, color=axis_text_color),
                        gridcolor=st.session_state.theme_styles["grid"],
                        zeroline=False
                    ),
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for valuation chart.")
            st.markdown("### How Intrinsic Values Are Calculated")
            for method, calc in calculation_details:
                with st.expander(f"{method} Calculation"):
                    st.markdown(f'<span class="calc-header">{method}</span>', unsafe_allow_html=True)
                    st.markdown(calc)
        else:
            st.warning(f"Could not fetch data for valuations: {error}")

# Learning Section
st.subheader("📚 Learn")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Test your investing knowledge with interactive quizzes and games.</p>', unsafe_allow_html=True)

# Quiz Section
quiz_level = st.selectbox("Select Quiz Level:", ["Beginner", "Intermediate", "Expert"])

quizzes = {
    "Beginner": [
        {"question": "What is a stock?", "options": ["A loan to a company", "Ownership in a company", "A type of bond", "A government security"], "answer": "Ownership in a company"},
        {"question": "What does P/E ratio measure?", "options": ["Profit margin", "Price per earnings", "Portfolio value", "Price per equity"], "answer": "Price per earnings"},
        {"question": "What is a dividend?", "options": ["A loan repayment", "A share of profits paid to shareholders", "A stock split", "A tax credit"], "answer": "A share of profits paid to shareholders"},
        {"question": "What is a bull market?", "options": ["Falling prices", "Rising prices", "Stable prices", "No trading"], "answer": "Rising prices"},
        {"question": "What does market cap represent?", "options": ["Company's total debt", "Total value of shares", "Annual revenue", "Dividend payout"], "answer": "Total value of shares"},
        {"question": "What is a bear market?", "options": ["Rising prices", "Falling prices", "Stable prices", "High volatility"], "answer": "Falling prices"},
        {"question": "What does EPS stand for?", "options": ["Equity Price Standard", "Earnings Per Share", "Economic Profit System", "Exchange Price System"], "answer": "Earnings Per Share"},
        {"question": "What is the stock market?", "options": ["A place to buy groceries", "A platform to trade company shares", "A type of bank", "A government agency"], "answer": "A platform to trade company shares"},
        {"question": "What is a portfolio?", "options": ["A single stock", "A collection of investments", "A type of loan", "A financial report"], "answer": "A collection of investments"},
        {"question": "What does diversification mean?", "options": ["Investing in one stock", "Spreading investments across assets", "Selling all stocks", "Buying only bonds"], "answer": "Spreading investments across assets"}
    ],
    "Intermediate": [
        {"question": "What is a golden cross?", "options": ["50-day SMA crossing above 200-day SMA", "A sharp price drop", "A dividend increase", "A stock split"], "answer": "50-day SMA crossing above 200-day SMA"},
        {"question": "What does RSI above 70 indicate?", "options": ["Oversold", "Overbought", "Neutral", "High volume"], "answer": "Overbought"},
        {"question": "What is beta?", "options": ["A measure of debt", "A measure of stock volatility", "A type of option", "A dividend metric"], "answer": "A measure of stock volatility"},
        {"question": "What does a high P/B ratio suggest?", "options": ["Undervalued stock", "Overvalued stock", "Low debt", "High dividends"], "answer": "Overvalued stock"},
        {"question": "What is a death cross?", "options": ["50-day SMA crossing below 200-day SMA", "A stock merger", "A new IPO", "A market crash"], "answer": "50-day SMA crossing below 200-day SMA"},
        {"question": "What does MACD measure?", "options": ["Market cap", "Momentum and trend changes", "Dividend yield", "Debt ratio"], "answer": "Momentum and trend changes"},
        {"question": "What is a stop-loss order?", "options": ["A buy order", "An order to sell if price drops", "A dividend reinvestment", "A market prediction"], "answer": "An order to sell if price drops"},
        {"question": "What does a low beta indicate?", "options": ["High volatility", "Low volatility", "High dividends", "Low earnings"], "answer": "Low volatility"},
        {"question": "What are Bollinger Bands used for?", "options": ["Measuring earnings", "Identifying overbought/oversold conditions", "Calculating dividends", "Tracking volume"], "answer": "Identifying overbought/oversold conditions"},
        {"question": "What does ROE measure?", "options": ["Return on Equity", "Revenue over Expenses", "Risk of Earnings", "Return on Exchange"], "answer": "Return on Equity"}
    ],
    "Expert": [
        {"question": "What is the DCF valuation method?", "options": ["Dividend discount model", "Discounted cash flow", "Debt-to-equity calculation", "Direct cost formula"], "answer": "Discounted cash flow"},
        {"question": "What does a PEG ratio below 1 suggest?", "options": ["Overvalued stock", "Undervalued stock", "High debt", "Low growth"], "answer": "Undervalued stock"},
        {"question": "What is the purpose of Bollinger Bands?", "options": ["Measure earnings", "Identify overbought/oversold conditions", "Calculate dividends", "Predict taxes"], "answer": "Identify overbought/oversold conditions"},
        {"question": "What does a high debt/equity ratio indicate?", "options": ["Low risk", "High financial leverage", "High dividends", "Low volatility"], "answer": "High financial leverage"},
        {"question": "What is intrinsic value?", "options": ["Market price", "True worth of a stock", "Dividend payout", "Trading volume"], "answer": "True worth of a stock"},
        {"question": "What does a bullish MACD crossover suggest?", "options": ["Sell signal", "Buy signal", "Neutral market", "High volatility"], "answer": "Buy signal"},
        {"question": "What is the Graham Method?", "options": ["A growth-based valuation", "EPS with growth multiplier", "A debt calculation", "A dividend model"], "answer": "EPS with growth multiplier"},
        {"question": "What does a high P/S ratio indicate?", "options": ["Undervalued stock", "Overvalued stock", "Low revenue", "High dividends"], "answer": "Overvalued stock"},
        {"question": "What is a comps valuation?", "options": ["Comparing to industry peers", "Discounting cash flows", "Using dividends", "Calculating book value"], "answer": "Comparing to industry peers"},
        {"question": "What does a low RSI (below 30) suggest?", "options": ["Overbought", "Oversold", "Neutral", "High volume"], "answer": "Oversold"}
    ]
}

if quiz_level:
    st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>{quiz_level} Quiz</b></p>', unsafe_allow_html=True)
    score = 0
    total_questions = len(quizzes[quiz_level])
    for i, q in enumerate(quizzes[quiz_level], 1):
        st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>Question {i}: {q["question"]}</b></p>', unsafe_allow_html=True)
        answer = st.radio(f"Select an answer for question {i}:", q["options"], key=f"quiz_{quiz_level}_{i}")
        if st.button(f"Check Answer {i}", key=f"check_{quiz_level}_{i}"):
            if answer == q["answer"]:
                st.success("Correct!")
                score += 1
            else:
                st.error(f"Incorrect. The answer is: {q['answer']}.")
    if st.button("Submit Quiz", key=f"submit_{quiz_level}"):
        st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Your Score: {score}/{total_questions} ({score/total_questions*100:.1f}%)</p>', unsafe_allow_html=True)
        if score == total_questions:
            st.success("Perfect score! You're a stock market pro!")
        elif score >= 7:
            st.info("Great job! You're learning fast.")
        else:
            st.warning("Keep practicing! Try the game below to improve.")

# Learning Game Section
st.markdown("### Stock Market Challenge Game")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Invest virtual $10,000 by answering questions correctly. Earn or lose based on your choices!</p>', unsafe_allow_html=True)

if "game_balance" not in st.session_state:
    st.session_state.game_balance = 10000
if "game_round" not in st.session_state:
    st.session_state.game_round = 0

game_questions = [
    {"question": "Should you buy a stock with a P/E ratio of 50? (High P/E may indicate overvaluation)", "options": ["Yes", "No"], "correct": "No", "impact": -500},
    {"question": "Is a MACD bullish crossover a buy signal?", "options": ["Yes", "No"], "correct": "Yes", "impact": 300},
    {"question": "Should you invest in a company with negative EPS?", "options": ["Yes", "No"], "correct": "No", "impact": -400},
    {"question": "Does a high dividend yield always mean a good investment?", "options": ["Yes", "No"], "correct": "No", "impact": -200},
    {"question": "Is a low beta stock less volatile?", "options": ["Yes", "No"], "correct": "Yes", "impact": 250},
    {"question": "Should you buy if RSI is above 70?", "options": ["Yes", "No"], "correct": "No", "impact": -300},
    {"question": "Is a golden cross a bullish signal?", "options": ["Yes", "No"], "correct": "Yes", "impact": 400},
    {"question": "Does a high P/B ratio suggest undervaluation?", "options": ["Yes", "No"], "correct": "No", "impact": -350},
    {"question": "Is a stock with a PEG ratio below 1 likely undervalued?", "options": ["Yes", "No"], "correct": "Yes", "impact": 500},
    {"question": "Should you sell if a death cross occurs?", "options": ["Yes", "No"], "correct": "Yes", "impact": 200}
]

if st.session_state.game_round < len(game_questions):
    current_q = game_questions[st.session_state.game_round]
    st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>Round {st.session_state.game_round + 1}: {current_q["question"]}</b></p>', unsafe_allow_html=True)
    user_choice = st.radio("Your choice", current_q["options"], key=f"game_{st.session_state.game_round}")
    if st.button("Submit Answer", key=f"game_submit_{st.session_state.game_round}"):
        if user_choice == current_q["correct"]:
            st.session_state.game_balance += abs(current_q["impact"])  # Gain money for correct answer
            st.success(f"Correct! You earned ${abs(current_q['impact'])}. New balance: ${st.session_state.game_balance}")
        else:
            st.session_state.game_balance -= abs(current_q["impact"])  # Lose money for incorrect answer
            st.error(f"Wrong! You lost ${abs(current_q['impact'])}. New balance: ${st.session_state.game_balance}")
        st.session_state.game_round += 1
else:
    st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}"><b>Game Over! Final Balance: ${st.session_state.game_balance}</b></p>', unsafe_allow_html=True)
    if st.session_state.game_balance > 10000:
        st.success("You beat the market! Great job!")
    elif st.session_state.game_balance < 10000:
        st.warning("You lost money. Try the quiz again to improve!")
    if st.button("Restart Game"):
        st.session_state.game_balance = 10000
        st.session_state.game_round = 0
# Portfolio
st.subheader("💼 Portfolio")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Track your investments and analyze sentiment.</p>', unsafe_allow_html=True)

@retry_with_backoff
def get_sentiment(ticker):
    cache_key = f"sentiment_{ticker}"
    if cache_key not in st.session_state.cache:
        logger.info(f"Fetching sentiment for {ticker}")
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
        except Exception as e:
            logger.error(f"Failed to fetch sentiment for {ticker}: {str(e)}")
            st.session_state.cache[cache_key] = "N/A"
    return st.session_state.cache[cache_key]

portfolio_input = st.text_input("Enter tickers (comma-separated, e.g., AAPL,MSFT):", value="AAPL,MSFT").strip().upper()
shares_input = st.text_input("Enter shares for each ticker (comma-separated, e.g., 100,50):", value="100,50").strip()
purchase_price_input = st.text_input("Enter purchase price per share for each ticker (comma-separated, e.g., 150.00,300.00):", value="150.00,300.00").strip()

if portfolio_input and shares_input and purchase_price_input:
    tickers = [t.strip() for t in portfolio_input.split(",")]
    try:
        shares = [int(s.strip()) for s in shares_input.split(",")]
        purchase_prices = [float(p.strip()) for p in purchase_price_input.split(",")]
        if len(tickers) != len(shares) or len(tickers) != len(purchase_prices):
            st.error("Number of tickers, shares, and purchase prices must match.")
        else:
            portfolio_data = []
            with st.spinner("Fetching portfolio data..."):
                for ticker, share, purchase_price in zip(tickers, shares, purchase_prices):
                    stock_info, error = get_stock_info(ticker)
                    if stock_info:
                        current_price = float(stock_info.get("PreviousClose", stock_info.get("regularMarketPrice", stock_info.get("regularMarketPreviousClose", 0))))
                        pe_ratio = float(stock_info.get("PERatio", stock_info.get("trailingPE", "N/A")))
                        pb_ratio = float(stock_info.get("PriceToBookRatio", stock_info.get("priceToBook", "N/A")))
                        sentiment = get_sentiment(ticker)
                        gain_loss = (current_price - purchase_price) * share
                        portfolio_data.append({
                            "Ticker": ticker,
                            "Shares": share,
                            "Purchase Price": f"${purchase_price:.2f}",
                            "Current Price": f"${current_price:.2f}",
                            "Value": f"${current_price * share:.2f}",
                            "Gain/Loss": f"${gain_loss:.2f}",
                            "P/E Ratio": f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio,
                            "P/B Ratio": f"{pb_ratio:.2f}" if isinstance(pb_ratio, (int, float)) else pb_ratio,
                            "Sentiment": sentiment
                        })
                    else:
                        st.warning(f"Could not fetch data for {ticker}: {error}")
            if portfolio_data:
                st.markdown("### Portfolio Summary")
                portfolio_df = pd.DataFrame(portfolio_data)
                st.dataframe(portfolio_df, use_container_width=True)
                csv = portfolio_df.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio Summary as CSV",
                    data=csv,
                    file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                # Portfolio Allocation Chart
                st.markdown("### Portfolio Allocation")
                values = [float(item["Value"].replace("$", "")) for item in portfolio_data]
                labels = [item["Ticker"] for item in portfolio_data]
                axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        textinfo='label+percent',
                        marker=dict(colors=st.session_state.theme_styles["bar_colors"][:len(labels)]),
                        textfont=dict(color=axis_text_color)
                    )
                ])
                fig.update_layout(
                    title=dict(
                        text="Portfolio Allocation by Value",
                        font=dict(size=20, color=axis_text_color),
                        x=0.5,
                        xanchor="center"
                    ),
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=12, color=axis_text_color),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                # Portfolio Simulator
                st.markdown("### Portfolio Simulator")
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Estimate your portfolio’s growth over time.</p>', unsafe_allow_html=True)
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
                # Growth Over Time Chart
                years_range = range(years + 1)
                growth_values = [total_value * (1 + growth_rate / 100) ** t for t in years_range]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(years_range),
                    y=growth_values,
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color=st.session_state.theme_styles["line"], width=2)
                ))
                fig.update_layout(
                    title=dict(
                        text="Portfolio Growth Over Time",
                        font=dict(size=20, color=axis_text_color),
                        x=0.5,
                        xanchor="center"
                    ),
                    xaxis_title="Years",
                    yaxis_title="Portfolio Value (USD)",
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=12, color=axis_text_color),
                    xaxis=dict(
                        title=dict(text="Years", font=dict(color=axis_text_color, size=14)),
                        tickfont=dict(family="Arial", size=12, color=axis_text_color),
                        gridcolor=st.session_state.theme_styles["grid"],
                        zeroline=False
                    ),
                    yaxis=dict(
                        title=dict(text="Portfolio Value (USD)", font=dict(color=axis_text_color, size=14)),
                        tickfont=dict(family="Arial", size=12, color=axis_text_color),
                        gridcolor=st.session_state.theme_styles["grid"],
                        zeroline=False
                    ),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
    except ValueError:
        st.error("Invalid input for shares or purchase prices. Please enter numbers (e.g., 100,50 for shares; 150.00,300.00 for prices).")

# Reddit Sentiment
st.subheader("🗣️ Reddit Sentiment")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Analyze sentiment from Reddit posts.</p>', unsafe_allow_html=True)
sentiment_ticker = st.text_input("Enter a ticker for sentiment analysis:", value="AAPL").strip().upper()

if sentiment_ticker:
    with st.spinner(f"Fetching Reddit posts for {sentiment_ticker}..."):
        try:
            subreddits = ["wallstreetbets", "stocks"]
            posts = []
            sentiment_scores = []
            for subreddit in subreddits:
                for submission in reddit.subreddit(subreddit).search(sentiment_ticker, limit=5):
                    text = submission.title + " " + (submission.selftext[:200] if submission.selftext else "")
                    score = analyzer.polarity_scores(text)["compound"]
                    sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
                    sentiment_scores.append(score)
                    posts.append({
                        "Title": submission.title,
                        "URL": submission.url,
                        "Sentiment": f"{sentiment} ({score:.2f})",
                        "Date": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d"),
                        "Subreddit": subreddit
                    })
            if posts:
                avg_score = np.mean(sentiment_scores)
                overall_sentiment = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
                st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Overall Sentiment: {overall_sentiment} ({avg_score:.2f})</p>', unsafe_allow_html=True)
                st.markdown("**Sentiment Explanation**: Scores range from -1 (most negative) to +1 (most positive). Above 0.05 is Positive, below -0.05 is Negative, and between -0.05 and 0.05 is Neutral.")
                st.markdown("### Recent Reddit Posts")
                reddit_df = pd.DataFrame(posts)
                st.dataframe(reddit_df, use_container_width=True)
                csv = reddit_df.to_csv(index=False)
                st.download_button(
                    label="Download Reddit Sentiment as CSV",
                    data=csv,
                    file_name=f"reddit_sentiment_{sentiment_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                # Sentiment Distribution
                st.markdown("### Sentiment Distribution")
                sentiment_counts = reddit_df["Sentiment"].apply(lambda x: x.split(" (")[0]).value_counts()
                axis_text_color = "#000000" if st.session_state.theme == "light" else "#FFFFFF"
                fig = go.Figure(data=[
                    go.Pie(
                        labels=sentiment_counts.index,
                        values=sentiment_counts.values,
                        textinfo='label+percent',
                        marker=dict(colors=st.session_state.theme_styles["bar_colors"][:len(sentiment_counts)]),
                        textfont=dict(color=axis_text_color)
                    )
                ])
                fig.update_layout(
                    title=dict(
                        text="Sentiment Distribution",
                        font=dict(size=20, color=axis_text_color),
                        x=0.5,
                        xanchor="center"
                    ),
                    plot_bgcolor=st.session_state.theme_styles["plot_bg"],
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Arial", size=12, color=axis_text_color),
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No Reddit posts found for {sentiment_ticker}.")
        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {sentiment_ticker}: {str(e)}")
            st.error(f"Error fetching Reddit posts: {str(e)}.")

# News Section
import streamlit as st
import logging
import os
from urllib.parse import quote

# Create log file if it doesn't exist
if not os.path.exists('stock_news.log'):
    with open('stock_news.log', 'w') as f:
        f.write('')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='stock_news.log'
)
logger = logging.getLogger(__name__)

# Initialize session state for API requests (for compatibility, though unused)
if "api_requests_made" not in st.session_state:
    st.session_state.api_requests_made = 0

# Function to generate Google News search URL
def generate_google_news_url(ticker: str) -> str:
    try:
        logger.info(f"Generating Google News URL for ticker: {ticker}")
        # Create query like "Recent TSLA news"
        query = f"Recent {ticker} news"
        # URL-encode the query
        encoded_query = quote(query)
        # Construct Google News search URL
        url = f"https://www.google.com/search?q={encoded_query}&tbm=nws"
        logger.info(f"Generated URL: {url}")
        return url
    except Exception as e:
        logger.error(f"Error generating Google News URL for {ticker}: {str(e)}")
        return ""

# UI: Ticker input and Google News link
st.subheader("📰 Stock News")
st.markdown(
    '<p style="color:#000000">Enter a stock ticker to search for recent news on Google.</p>',
    unsafe_allow_html=True
)

news_ticker = st.text_input("Enter a ticker for news:", value="AAPL").strip().upper()
if news_ticker:
    with st.spinner(f"Generating news search for {news_ticker}..."):
        google_news_url = generate_google_news_url(news_ticker)
        if google_news_url:
            st.markdown("### Recent News Search")
            st.markdown(
                f'<p style="color:#000000">\n'
                f'Click the link below to view recent news for {news_ticker} on Google News:<br>'
                f'<a href="{google_news_url}" target="_blank">Search Google News for {news_ticker}</a></p>',
                unsafe_allow_html=True
            )
        else:
            st.error("Error generating news search URL. Please try again.")


# Stock Screener
st.subheader("🔎 Stock Screener")
st.markdown(f'<p style="color:{st.session_state.theme_styles["text"]}">Screen stocks based on your criteria.</p>', unsafe_allow_html=True)

screener_tickers = st.text_input("Enter tickers to screen (comma-separated, e.g., AAPL,MSFT,GOOGL):", value="AAPL,MSFT,GOOGL").strip().upper()
min_pe = st.number_input("Minimum P/E Ratio:", value=0.0, step=1.0)
max_pe = st.number_input("Maximum P/E Ratio:", value=50.0, step=1.0)
min_pb = st.number_input("Minimum P/B Ratio:", value=0.0, step=0.1)
max_pb = st.number_input("Maximum P/B Ratio:", value=10.0, step=0.1)
min_dividend = st.number_input("Minimum Dividend Yield (%):", value=0.0, step=0.1)

if screener_tickers:
    tickers = [t.strip() for t in screener_tickers.split(",")]
    screener_data = []
    with st.spinner("Screening stocks..."):
        for ticker in tickers:
            stock_info, error = get_stock_info(ticker)
            if stock_info:
                pe_ratio = float(stock_info.get("PERatio", stock_info.get("trailingPE", float('inf'))))
                pb_ratio = float(stock_info.get("PriceToBookRatio", stock_info.get("priceToBook", float('inf'))))
                dividend_yield = float(stock_info.get("DividendYield", stock_info.get("dividendYield", 0))) * 100
                if (min_pe <= pe_ratio <= max_pe) and (min_pb <= pb_ratio <= max_pb) and (dividend_yield >= min_dividend):
                    screener_data.append({
                        "Ticker": ticker,
                        "P/E Ratio": f"{pe_ratio:.2f}" if pe_ratio != float('inf') else "N/A",
                        "P/B Ratio": f"{pb_ratio:.2f}" if pb_ratio != float('inf') else "N/A",
                        "Dividend Yield": f"{dividend_yield:.2f}%"
                    })
            else:
                logger.warning(f"Could not fetch data for {ticker} during screening: {error}")
    if screener_data:
        st.markdown("### Screening Results")
        screener_df = pd.DataFrame(screener_data)
        st.dataframe(screener_df, use_container_width=True)
        csv = screener_df.to_csv(index=False)
        st.download_button(
            label="Download Screening Results as CSV",
            data=csv,
            file_name=f"stock_screener_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No stocks match your criteria.")

# Footer
st.markdown("---")
st.markdown(
    f'<p style="color:{st.session_state.theme_styles["text"]};text-align:center">'
    f'Powered by Alpha Vantage, yfinance, Reddit, and Streamlit | © 2025 EquityScope</p>',
    unsafe_allow_html=True
)
