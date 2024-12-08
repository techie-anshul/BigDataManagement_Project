import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def train_and_test_model(processed_df):
    if not isinstance(processed_df, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")
    
    # Prepare data for LSTM model
    data = processed_df[['close_price']].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(60, len(scaled_data)):
        x.append(scaled_data[i-60:i])
        y.append(scaled_data[i])
    x, y = np.array(x), np.array(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Build the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)
    
    # Evaluate the model
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, predictions)
    print(f"Model MSE: {mse}")
    
    # Save the trained model
    model.save("stock_price_model.keras")
    #model.save("stock_price_model.h5")
    print("Model training and testing completed.")
    
    return model

