import pandas as pd
import numpy as np

def preprocess_data(stock_data):
    # Your preprocessing logic here
    # Example: Ensure 'stock_data' is a DataFrame
    if not isinstance(stock_data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")
    
    # Perform preprocessing
    stock_data['log_return'] = np.log(stock_data['close_price'] / stock_data['close_price'].shift(1))
    stock_data = stock_data.dropna()
    
    # Save processed data
    stock_data.to_parquet("/home/vm-gg23ai2066-3/Downloads/smapp_output/processed_stock_data.parquet", index=False)
    print("Preprocessing completed. Data saved to Parquet format.")
    return stock_data

'''
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def preprocess_data():
    spark = SparkSession.builder \
        .appName("StockDataProcessing") \
        .config("spark.jars.packages", "org.postgresql:postgresql:42.5.4") \
        .getOrCreate()
    
    jdbc_url = "jdbc:postgresql://localhost:5432/stock_db"
    connection_properties = {"user": "postgres", "password": "Gray", "driver": "org.postgresql.Driver"}
    
    stock_df = spark.read.jdbc(url=jdbc_url, table="stock_prices", properties=connection_properties)
    stock_df = stock_df.withColumn("timestamp", col("timestamp").cast("timestamp")) \
                       .orderBy(col("timestamp").asc())
    
    stock_df = stock_df.select("timestamp", "close_price") \
                       .groupBy("timestamp") \
                       .mean("close_price") \
                       .withColumnRenamed("avg(close_price)", "close_price")
    
    stock_df.write.mode("overwrite").parquet("processed_stock_data.parquet")
    print("Preprocessing completed. Data saved to Parquet format.")
'''
