import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def predict_stock_price(model, data):
    # Load the trained model
    # model = tf.keras.models.load_model("/path/to/your/model/stock_price_model.h5")  # Adjust the path if needed
    
    # Load the scaler used for training
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Prepare the input for prediction (assumes the model expects a specific input shape)
    if data.ndim == 3 and data.shape[1] == 60 and data.shape[2] == 1:  # If the input data is of size 10, reshape it to the required shape
        data = data.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    x_input = scaled_data[-60:].reshape(1, 60, 1)  # Assuming the model was trained with 60 time steps
    
    # Predict the future stock price
    predicted_price = model.predict(x_input)
    
    # Inverse transform to get the actual stock price
    predicted_price = scaler.inverse_transform(predicted_price)
    print(f"Predicted Stock Price: {predicted_price[0][0]}")
    
    return predicted_price[0] # Get the scalar value from the prediction array

