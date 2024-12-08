from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine
import boto3

# Initialize Spark Session for Distributed Processing
spark = SparkSession.builder \
    .appName("StockMarketPredictor") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# PostgreSQL Connection
db_engine = create_engine("postgresql://postgres:Gray@localhost:5432/stock_db")

cursor = db_engine.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock_prices (
    symbol TEXT,
    datetime TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
)
""")
db_engine.commit()

'''
# AWS S3 Configuration
s3_client = boto3.client(
    "s3",
    aws_access_key_id="your_aws_access_key_id",
    aws_secret_access_key="your_aws_secret_access_key",
    region_name="your_aws_region"
)
bucket_name = "your-s3-bucket-name"
'''
# Fetch Stock Data for the Last 10 Years
def fetch_historical_data(symbols):
    for symbol in symbols:
        print(f"Fetching 10 years of data for {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period="10y", interval="1d")
        hist.reset_index(inplace=True)
        hist["Symbol"] = symbol

        # Convert to Spark DataFrame for Distributed Processing
        spark_df = spark.createDataFrame(hist)
        
        # Optional: Write raw data to S3
        #file_path = f"s3://{bucket_name}/{symbol}_10y.csv"
        #spark_df.write.csv(file_path, mode="overwrite", header=True)

        # Store Data in PostgreSQL
        hist.to_sql("stock_prices", db_engine, if_exists="append", index=False)

# Preprocess Data with Spark
def preprocess_data_spark(symbol):
    query = f"(SELECT * FROM stock_prices WHERE symbol = '{symbol}') AS stock_data"
    df = pd.read_sql_query(query, db_engine)

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Perform preprocessing (e.g., handle missing values, add features)
    spark_df = spark_df.filter(col("close").isNotNull())
    spark_df = spark_df.withColumn("Return", (col("close") - col("open")) / col("open"))
    spark_df = spark_df.dropna()

    # Convert back to Pandas for ML usage
    preprocessed_df = spark_df.toPandas()
    return preprocessed_df

# Train LSTM Model with Preprocessed Data
def train_lstm_model(df):
    data = df['close'].values.reshape(-1, 1)

    # Train-Test Split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Scaling
    from sklearn.preprocessing import MinMaxScaler
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

    # Build LSTM Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    return model, scaler

# Predict Stock Price
def predict_stock_price(symbol):
    df = preprocess_data_spark(symbol)
    model, scaler = train_lstm_model(df)
    last_seq = df['close'].values[-30:].reshape(-1, 1)
    last_seq_scaled = scaler.transform(last_seq)
    predicted_price = model.predict(last_seq_scaled[np.newaxis, :, :])[0][0]
    print(f"Predicted price for {symbol}: {scaler.inverse_transform([[predicted_price]])[0][0]}")
    return scaler.inverse_transform([[predicted_price]])[0][0]

# Main Function
if __name__ == "__main__":
    STOCK_SYMBOLS = ["INFY.NS", "TCS.NS", "RELIANCE.NS"]

    # Fetch Historical Data
    fetch_historical_data(STOCK_SYMBOLS)

    # Predict Future Prices
    for symbol in STOCK_SYMBOLS:
        predict_stock_price(symbol)

