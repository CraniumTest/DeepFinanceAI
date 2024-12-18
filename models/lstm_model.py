import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def load_and_prepare_data(self):
        data = pd.read_csv(self.data_path)
        data = self.scaler.fit_transform(data)
        return data
    
    def build_model(self, input_shape):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.scaler.inverse_transform(self.model.predict(X_test))
