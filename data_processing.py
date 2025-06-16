import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler

# Gets all the data from both tables in the database and combines them into a single dataframe with an added ticker column
def fetch_data(db_name='stock_data.db'):
    # Connect to the database
    conn = sqlite3.connect(db_name)
    # Fetch data for 'JPY=X' stock
    jpy_data = pd.read_sql('SELECT * FROM stock_prices_jpy', conn, parse_dates=['Date'])
    jpy_data['Ticker'] = 'JPY=X'
    # Fetch data for 'QQQ' stock
    qqq_data = pd.read_sql('SELECT * FROM stock_prices_qqq', conn, parse_dates=['Date'])
    qqq_data['Ticker'] = 'QQQ'
    # Combine both datasets into a single DataFrame
    data = pd.concat([jpy_data, qqq_data], ignore_index=True)
    # Close the database connection
    conn.close()
    return data

# Processes the DataFrame for machine learning by scaling 'QQQ' stock prices and preparing the target variable 'JPY=X' stock prices.
def process_data(data):
    # Check if the DataFrame is empty
    if data.empty:
        raise ValueError("No data available to process.")
    # Filter data to only include 'QQQ' and 'JPY=X'
    filtered_data = data[data['Ticker'].isin(['QQQ', 'JPY=X'])]
    # Pivot the DataFrame to have 'Date' as index and tickers as columns
    pivot_data = filtered_data.pivot(index='Date', columns='Ticker', values='Close')
    # Resample data to daily frequency and take the mean for each day
    pivot_data = pivot_data.resample('D').mean()
    # Drop any rows with missing values
    pivot_data.dropna(inplace=True)
    # Extract feature (X) and target (y) arrays
    X = pivot_data[['QQQ']].values
    y = pivot_data['JPY=X'].values
    # Scale the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

if __name__ == "__main__":
    # Fetch the stock price data
    data = fetch_data()
    # Process the data for machine learning
    X, y = process_data(data)
    # Print the processed feature and target arrays
    print(X)
    print(y)
