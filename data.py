import requests
import ta
import pandas as pd
from datetime import datetime


def convert_datetime_to_timestamp(datetime_str: str) -> int:
    """
    Convert the given date and time to a timestamp in milliseconds.
    """
    datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    timestamp = datetime_obj.timestamp() * 1000
    return int(timestamp)


def get_binance_klines():
    """
    Get 1h klines in the past year, which requires 27000 candles, from Binance.
    """
    url = "https://fapi.binance.com/fapi/v1/klines"

    current_timestamp = convert_datetime_to_timestamp("2024-09-20 00:00:00")

    klines = []

    for _ in range(18):
        recent_1500_klines = requests.get(
            url,
            params={
                "symbol": "BTCUSDT",
                "interval": "1h",
                "limit": 1500,
                "endTime": current_timestamp - 1,
            },
        ).json()

        klines = recent_1500_klines + klines

        # Minus current_timestamp by 1500 * 1 hour to get the start time
        current_timestamp -= 1500 * 60 * 60 * 1000

    return klines


def create_lagged_features(df, num_lags):
    # Create a copy of the DataFrame to avoid modifying the original data
    df_lagged = df.copy()
    
    # For each column (except 'next_diff'), create lagged features using a vectorized approach
    columns = df.columns.difference(["next_diff"])
    
    for lag in range(1, num_lags + 1):
        shifted = df[columns].shift(lag)  # Shift all columns at once for the given lag
        shifted.columns = [f"{col}_lag{lag}" for col in columns]  # Rename shifted columns
        df_lagged = pd.concat([df_lagged, shifted], axis=1)  # Concatenate with original df

    return df_lagged


def prepare_data():
    """
    Prepare data for training the model.
    1. Get klines from Binance.
    2. Calculate corresponding RSI values.
    3. Make pandas DataFrame.

    k_lines format:
        [
            [
                1499040000000,      // Open time
                "0.01634790",       // Open
                "0.80000000",       // High
                "0.01575800",       // Low
                "0.01577100",       // Close
                "148976.11427815",  // Volume
                1499644799999,      // Close time
                "2434.19055334",    // Quote asset volume
                308,                // Number of trades
                "1756.87402397",    // Taker buy base asset volume
                "28.46694368",      // Taker buy quote asset volume
                "17928899.62484339" // Ignore.
            ]
        ]
        
        BTC -> base / USDT -> quote
    """
    klines = get_binance_klines()

    # Make a DataFrame only includes close prices and quote asset volumes
    df = pd.DataFrame()
    df["body"] = [(float(kline[4]) - float(kline[1])) for kline in klines]
    df["top_tail"] = [
        (float(kline[2]) - max([float(kline[1]), float(kline[4])])) for kline in klines
    ]
    df["bottom_tail"] = [
        (min([float(kline[1]), float(kline[4])]) - float(kline[3])) for kline in klines
    ]
    df["close"] = [float(kline[4]) for kline in klines]
    df["volume"] = [float(kline[7]) for kline in klines]

    # Calculate RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # Calculate MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd_diff()  # Add the MACD difference (MACD - Signal line)

    # Add label "trend" which is the difference between the next close price and the current close price
    df["next_diff"] = df["close"].shift(-3) - df["close"]

    # Drop the first 300 rows
    df = df.iloc[300:]

    # Drop the last 3 rows
    df = df.iloc[:-3]

    slided_df = create_lagged_features(df, 14)

    # Drop the first 15 rows
    return slided_df.iloc[15:]


prepare_data()
