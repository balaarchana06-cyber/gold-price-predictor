from pdb import run
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_KEY = "goldapi-x0gsmlnw3u1s-io"
BASE_URL = "https://www.goldapi.io/api/XAU/INR"
HEADERS = {"x-access-token": API_KEY, "Content-Type": "application/json"}
GST_RATE = 0.03  # 3% GST

# Indian states and union territories
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
    'Andaman and Nicobar Islands', 'Chandigarh', 'Delhi', 'Jammu and Kashmir', 'Ladakh', 'Puducherry'
]

def fetch_real_time_gold_price():
    """Fetch real-time gold price from GoldAPI.io."""
    try:
        response = requests.get(BASE_URL, headers=HEADERS)
        data = response.json()
        if "price" in data:
            gold_price_per_gram = data["price"] / 31.1035  # Convert per ounce to INR per gram
            return gold_price_per_gram * 10  # Convert to 10 grams
    except Exception as e:
        logger.error(f"Error fetching gold price: {e}")
    return 89000  # Fallback price if API fails

def generate_sample_data():
    """Generate synthetic gold price data for Indian states."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    real_time_price = fetch_real_time_gold_price()
    base_price = real_time_price + (real_time_price * GST_RATE)

    data = []
    state_factors = {state: np.random.uniform(-0.02, 0.02) for state in INDIAN_STATES}  # State variations

    for state in INDIAN_STATES:
        state_factor = state_factors[state]
        state_base = base_price * (1 + state_factor)

        for date in dates:
            days_passed = (date - start_date).days
            trend = days_passed * 10
            daily_variation = np.random.normal(0, 500)
            seasonal_factor = np.sin(2 * np.pi * date.dayofyear / 365) * 1000
            price = state_base + trend + daily_variation + seasonal_factor
            price = max(price, base_price * 0.9)
            data.append({'state': state, 'date': date, 'price': round(price, 2)})

    return pd.DataFrame(data)

def predict_future_prices(state_data):
    """Predict future gold prices for the next 10 days using Linear Regression."""
    state_data = state_data.copy()
    state_data['days'] = (state_data['date'] - state_data['date'].min()).dt.days

    model = LinearRegression()
    model.fit(state_data[['days']], state_data['price'])

    future_dates = [state_data['date'].max() + timedelta(days=i) for i in range(1, 11)]
    future_days = [(date - state_data['date'].min()).days for date in future_dates]
    future_prices = model.predict(np.array(future_days).reshape(-1, 1))

    return pd.DataFrame({'date': future_dates, 'predicted_price': future_prices})

def main():
    st.title('Gold Price Prediction with Real-Time Data')
    st.caption('Live gold price analysis and forecast for Indian states')

    gold_data = generate_sample_data()
    if gold_data.empty:
        st.error("Unable to generate gold price data.")
        return

    st.sidebar.header('Settings')
    state = st.sidebar.selectbox('Select a state', INDIAN_STATES)

    min_date = gold_data['date'].min().date()
    max_date = gold_data['date'].max().date()
    start_date, end_date = st.sidebar.date_input('Select date range', (min_date, max_date), min_value=min_date, max_value=max_date)

    mask = (gold_data['state'] == state) & (gold_data['date'].dt.date >= start_date) & (gold_data['date'].dt.date <= end_date)
    state_data = gold_data.loc[mask].copy()

    if not state_data.empty:
        st.subheader(f'Historical Gold Prices in {state}')
        fig = plt.figure(figsize=(10, 6))
        plt.plot(state_data['date'], state_data['price'], 'b-', label='Historical Prices')
        plt.xlabel('Date')
        plt.ylabel('Price (INR per 10g)')
        plt.legend()
        st.pyplot(fig)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"₹{state_data['price'].iloc[-1]:,.2f}")
        with col2:
            st.metric("Avg Price", f"₹{state_data['price'].mean():,.2f}")
        with col3:
            st.metric("High Price", f"₹{state_data['price'].max():,.2f}")
        with col4:
            st.metric("Low Price", f"₹{state_data['price'].min():,.2f}")

        st.subheader('Future Gold Price Prediction (Next 10 Days)')
        future_prices = predict_future_prices(state_data)
        st.table(future_prices.style.format({'predicted_price': '₹{:.2f}'}))

        fig_future = plt.figure(figsize=(10, 6))
        plt.plot(future_prices['date'], future_prices['predicted_price'], 'r-', label='Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price (INR per 10g)')
        plt.legend()
        st.pyplot(fig_future)

    st.markdown("---")
    st.caption("**Disclaimer:** This is an approximation using real-time data with simulated variations.")

if __name__ == "__main__":
    main()

    #python -m pip install streamlit pandas numpy scikit-learn matplotlib seaborn requests
#python -m streamlit run "c:\Users\acer\OneDrive\Desktop\gold price project\gold.py"