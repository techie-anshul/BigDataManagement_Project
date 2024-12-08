from sqlalchemy import create_engine, text
import pandas as pd

def connect_to_database():
    engine = create_engine("postgresql://postgres:Gray@localhost:5432/stock_db")
    return engine

def setup_database(engine):
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume INT,
    symbol VARCHAR(10)
     );
        """))
    print("Database and table setup completed.")

def insert_stock_data(engine, symbol, stock_data):
    stock_data['symbol'] = symbol
    stock_data.to_sql('stock_prices', con=engine, if_exists='append', index=False)
    print(f"Data inserted for symbol: {symbol}")

