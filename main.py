from flask import Flask, render_template, jsonify, request
from db_conn import connect_to_database, setup_database, insert_stock_data
from preprocess_spark import preprocess_data
from train_test_model import train_and_test_model
from flask import Flask, render_template, jsonify, request
from db_conn import connect_to_database, setup_database, insert_stock_data
from preprocess_spark import preprocess_data
from train_test_model import train_and_test_model
from stock_price_predict import predict_stock_price
import pandas as pd
import numpy as np
from kafka import KafkaProducer
import json
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# In-memory storage for predictions
predictions = []

def simulate_stock_data():
    symbols = ["INFY", "TCS", "RELIANCE", "WIPRO", "TATASTEEL"]
    for symbol in symbols:
        stock_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'open_price': np.random.rand(100) * 1000,
            'high_price': np.random.rand(100) * 1000,
            'low_price': np.random.rand(100) * 1000,
            'close_price': np.random.rand(100) * 1000,
            'volume': np.random.randint(1000, 5000, size=100)
        })
        engine = connect_to_database()
        insert_stock_data(engine, symbol, stock_data)

        # Preprocess and predict stock prices
        processed_df = preprocess_data(stock_data)
        model = train_and_test_model(processed_df)
        
        # Prepare future data for prediction
        future_data = np.random.rand(60) * 1000  # Assuming the model was trained with 60 time steps
        future_data_reshaped = future_data.reshape(1, 60,1)  # Reshape for LSTM model input
        preds = predict_stock_price(model, future_data_reshaped)
        predictions.append({
            'symbol': symbol,
            'prediction': preds[0]
        })

        # Plot the prediction
        plt.figure()
        plt.plot(stock_data['timestamp'], stock_data['close_price'], label='Actual Price')
        plt.plot(stock_data['timestamp'][-60:], [preds[0]] * 60, label='Predicted Price')  # Ensure prediction is a scalar
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Prediction for {symbol}')
        plt.legend()

        # Save the plot
        plot_path = f'/home/vm-gg23ai2066-3/Downloads/smapp_output/static/{symbol}.png'
        plt.savefig(os.path.join(app.root_path, plot_path))
        plt.close()  # Close the plot to free memory

@app.route("/")
def index():
    return render_template("index.html", predictions=predictions)

@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    return jsonify([{'symbol': pred['symbol'], 'prediction': float(pred['prediction'])} for pred in predictions])

@app.route("/predict", methods=["POST"])
def predict():
    stock_symbol = request.form['stock_symbol']
    # Assuming `predict_stock_price` can take stock_symbol or another suitable parameter to fetch and predict
    stock_data = fetch_stock_data_from_db(stock_symbol)  # Implement this function to fetch stock data from the database
    processed_data = preprocess_data(stock_data)
    model = train_and_test_model(processed_data)
    future_data = np.random.rand(60) * 1000  # Sample data; replace with actual future stock data
    future_data_reshaped = future_data.reshape(1, 60, 1)
    prediction = predict_stock_price(model, future_data_reshaped)
    
    return render_template("prediction.html", prediction=prediction[0], stock_symbol=stock_symbol)

if __name__ == "__main__":
    simulate_stock_data()  # Prepare data and predictions
    app.run(debug=True)


'''
from flask import Flask, render_template, jsonify, request
from db_conn import connect_to_database, setup_database, insert_stock_data
from preprocess_spark import preprocess_data
from train_test_model import train_and_test_model
from stock_price_predict import predict_stock_price
import pandas as pd
import numpy as np
from kafka import KafkaProducer
import json

app = Flask(__name__)

# In-memory storage for predictions
predictions = []

def simulate_stock_data():
    symbols = ["INFY", "TCS", "RELIANCE", "WIPRO", "TATASTEEL"]
    for symbol in symbols:
        stock_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'open_price': np.random.rand(100) * 1000,
            'high_price': np.random.rand(100) * 1000,
            'low_price': np.random.rand(100) * 1000,
            'close_price': np.random.rand(100) * 1000,
            'volume': np.random.randint(1000, 5000, size=100)
        })
        engine = connect_to_database()
        insert_stock_data(engine, symbol, stock_data)

        # Preprocess and predict stock prices
        processed_df = preprocess_data(stock_data)
        model = train_and_test_model(processed_df)
        
        # Prepare future data for prediction
        future_data = np.random.rand(60) * 1000  # Assuming the model was trained with 60 time steps
        future_data_reshaped = future_data.reshape(1, 60,1)  # Reshape for LSTM model input
        preds = predict_stock_price(model, future_data_reshaped)
        predictions.append(preds)


# Simulate stock data and store predictions
def simulate_stock_data():
    symbols = ["INFY", "TCS", "RELIANCE", "WIPRO", "TATASTEEL"]
    for symbol in symbols:
        stock_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'open_price': np.random.rand(100) * 1000,
            'high_price': np.random.rand(100) * 1000,
            'low_price': np.random.rand(100) * 1000,
            'close_price': np.random.rand(100) * 1000,
            'volume': np.random.randint(1000, 5000, size=100)
        })
        engine = connect_to_database()
        insert_stock_data(engine, symbol, stock_data)

        # Preprocess and predict stock prices
        processed_df = preprocess_data() #stock_data
        model = train_and_test_model(processed_df)
        future_data = np.random.rand(10) * 1000
        preds = predict_stock_price(model, future_data.reshape(-1, 1))
        predictions.extend(preds)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    return jsonify([float(price) for price in predictions])

if __name__ == "__main__":
    simulate_stock_data()  # Prepare data and predictions
    app.run(debug=True)
'''
