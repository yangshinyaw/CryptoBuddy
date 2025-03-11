import requests
import pandas as pd
import time

# Attempt to import Prophet, but do not exit if unavailable
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ModuleNotFoundError:
    print("Prophet module not found. Skipping price prediction.")
    PROPHET_AVAILABLE = False

# CoinGecko API to get crypto prices
def get_crypto_price(crypto="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get(crypto, {}).get('usd', 'Price unavailable')
    except requests.RequestException as e:
        print(f"Error fetching crypto price: {e}")
        return "Price unavailable"

# CryptoPanic API to get latest news (Requires API key)
def get_crypto_news():
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": "YOUR_API_KEY", "public": "true"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        news_list = [item['title'] for item in data.get('results', [])]
        return news_list[:5]  # Get top 5 news headlines
    except requests.RequestException as e:
        print(f"Error fetching crypto news: {e}")
        return []

# Fetch historical price data
def get_historical_data(crypto="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart?vs_currency=usd&days=90&interval=daily"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        prices = data.get('prices', [])
        if not prices:
            print("Error: No historical data available.")
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except requests.RequestException as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# Apply simple moving average (SMA)
def apply_technical_analysis(df):
    if df.empty:
        return df
    df['SMA_20'] = df['price'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['price'])
    return df

# Compute RSI manually
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Price prediction using Prophet (only if available)
def predict_price(df):
    if not PROPHET_AVAILABLE:
        print("Skipping price prediction due to missing Prophet module.")
        return pd.DataFrame()
    if df.empty:
        print("Insufficient data for price prediction.")
        return pd.DataFrame()
    df = df.rename(columns={'timestamp': 'ds', 'price': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(5)  # Show next 5 days predictions

# Main function
def main():
    print("Fetching crypto data...")
    price = get_crypto_price()
    news = get_crypto_news()
    df = get_historical_data()
    df = apply_technical_analysis(df)
    prediction = predict_price(df)
    
    print(f"Current Bitcoin Price: ${price}")
    print("Latest Crypto News:")
    for n in news:
        print(f"- {n}")
    print("\nPrice Predictions:")
    print(prediction if not prediction.empty else "No predictions available.")
    
# Run every 6 hours
if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
        time.sleep(21600)  # 6 hours
