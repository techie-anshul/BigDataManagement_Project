from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine, text
import pandas as pd
import os
import numpy as np
import yfinance as yf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, mean as spark_mean
from pyspark.sql.window import Window
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pyarrow as pa
import pyarrow.parquet as pq
import json
from plotly.utils import PlotlyJSONEncoder

# Initialize Flask App
app = Flask(__name__)

# Initialize Spark Session for Distributed Processing
spark = SparkSession.builder \
    .appName("StockMarketPredictor") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# PostgreSQL Connection
engine = create_engine("postgresql://postgres:Gray@localhost:5432/stock_db")

# Save DataFrame to HDFS (Updated for pyarrow.hdfs removal)
def save_to_hdfs(df, hdfs_path):
    try:
        # Save to local directory (as an alternative to HDFS for simplicity)
        local_path = hdfs_path.replace("/", "_")  # Simulate HDFS path in local
        pq.write_table(pa.Table.from_pandas(df), local_path)
        print(f"Data successfully written to local path: {local_path}")
    except Exception as e:
        print(f"Failed to write data: {e}")

# Database Setup
def setup_database():
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
            )
        """))
        print("Database setup completed.")

# Fetch Stock Data for the Last 10 Years
def fetch_historical_data(symbols):
    for symbol in symbols:
        print(f"Fetching 10 years of data for {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period="10y", interval="1d")
        hist.reset_index(inplace=True)
        hist.rename(columns={
            "Date": "timestamp",
            "Open": "open_price",
            "High": "high_price",
            "Low": "low_price",
            "Close": "close_price",
            "Volume": "volume"
        }, inplace=True)
        hist = hist[['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        
        # Save to HDFS (local simulation)
        hdfs_path = f"/stock_data/{symbol}_10y.parquet"
        save_to_hdfs(hist, hdfs_path)

        # Store Data in PostgreSQL
        hist.to_sql("stock_prices", engine, if_exists="append", index=False)
        print(f"Data for {symbol} inserted into database.")

# Generate Synthetic Data
def generate_synthetic_data(df, num_days=100):
    synthetic_data = []
    for _ in range(num_days):
        new_row = {
            "timestamp": df["timestamp"].max() + pd.Timedelta(days=1),
            "open_price": np.random.uniform(df["open_price"].min(), df["open_price"].max()),
            "high_price": np.random.uniform(df["high_price"].min(), df["high_price"].max()),
            "low_price": np.random.uniform(df["low_price"].min(), df["low_price"].max()),
            "close_price": np.random.uniform(df["close_price"].min(), df["close_price"].max()),
            "volume": int(np.random.uniform(df["volume"].min(), df["volume"].max()))
        }
        synthetic_data.append(new_row)
    print("Synthetic data generation completed.")
    return pd.DataFrame(synthetic_data)

# Preprocess Data with Spark
def preprocess_data_spark(symbol):
    print(f"Starting preprocessing for {symbol}")
    query = f"SELECT * FROM stock_prices WHERE symbol = '{symbol}'"
    df = pd.read_sql_query(query, engine)

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Handle missing values
    spark_df = spark_df.filter(col("close_price").isNotNull())

    # Add Moving Averages and Returns
    window_spec = Window.orderBy("timestamp")
    spark_df = spark_df \
        .withColumn("return", (col("close_price") - col("open_price")) / col("open_price")) \
        .withColumn("moving_avg_5", spark_mean(col("close_price")).over(window_spec.rowsBetween(-4, 0))) \
        .withColumn("moving_avg_20", spark_mean(col("close_price")).over(window_spec.rowsBetween(-19, 0)))

    # Drop rows with null values after feature engineering
    spark_df = spark_df.dropna()

    # Convert back to Pandas for ML usage
    preprocessed_df = spark_df.toPandas()
    print("Preprocessing completed.")
    return preprocessed_df

# Train LSTM Model with Preprocessed Data
def train_lstm_model(df):
    print("Starting LSTM model training.")
    data = df['close_price'].values.reshape(-1, 1)

    # Train-Test Split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Scaling
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Prepare Data for LSTM
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(x), np.array(y)

    seq_length = 30  # Use 30 days as the sequence length
    x_train, y_train = create_sequences(train_data, seq_length)
    x_test, y_test = create_sequences(test_data, seq_length)

    # Model Definition
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Training
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Predict
    predictions = model.predict(x_test)

    # Scale back to original data
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)

    print("Model training completed.")
    return model, predictions, y_test_actual

# Save Model
def save_model(model, symbol):
    model.save(f"./models/{symbol}_model.h5")
    print(f"Model saved for {symbol}")

# Route for Handling Predictions
@app.route("/predict/<symbol>", methods=["GET"])
def predict(symbol):
    try:
        # Preprocess the data
        df = preprocess_data_spark(symbol)

        # Train model
        model, predictions, y_test_actual = train_lstm_model(df)

        # Save model
        save_model(model, symbol)

        return jsonify({
            "message": f"Model saved at ./models/{symbol}_model.h5",
            "predictions": predictions.tolist(),
            "actual": y_test_actual.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Start the Flask App
if __name__ == "__main__":
    setup_database()  # Initial setup for the database
    fetch_historical_data(['AAPL', 'MSFT', 'GOOGL'])  # Fetch historical data for these stocks
    app.run(debug=True)

