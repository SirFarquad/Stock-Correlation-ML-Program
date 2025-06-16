import yfinance as yf
import sqlite3

# Get the data from yahoo finance by taking the arguments of ticker, start date, and end date
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Ticker'] = ticker
    return data[['Close', 'Ticker']]

# Make the database called stock_data and create tables for each stock
def create_table(db_name='stock_data.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE stock_prices_jpy (
        Date TEXT PRIMARY KEY,
        Close REAL
    )
    ''')
    cursor.execute('''
    CREATE TABLE stock_prices_qqq (
        Date TEXT PRIMARY KEY,
        Close REAL
    )
    ''')
    conn.commit()
    conn.close()

# Insert the data into the database
def store_data(data, ticker, db_name='stock_data.db'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    data.index = data.index.strftime('%Y-%m-%d')
    
    # Determine the table name based on the ticker
    table_name = 'stock_prices_jpy' if ticker == 'JPY=X' else 'stock_prices_qqq'
    
    for date, row in data.iterrows():
        try:
            cursor.execute(f'''
                INSERT INTO {table_name} (Date, Close) 
                VALUES (?, ?)
                ''', (date, row['Close']))
        except sqlite3.IntegrityError:
            # If the record already exists, update it
            cursor.execute(f'''
                UPDATE {table_name} 
                SET Close = ? 
                WHERE Date = ?
                ''', (row['Close'], date))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Create tables
    create_table()

    # Fetch data
    usd_jpy_data = fetch_data('JPY=X', '2000-01-01', '2024-01-01')
    qqq_data = fetch_data('QQQ', '2000-01-01', '2024-01-01')

    # Store data in the appropriate tables
    store_data(usd_jpy_data, 'JPY=X')
    store_data(qqq_data, 'QQQ')
