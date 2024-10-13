import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
import keras


class LSTMModel:
    def __init__(self, time_steps=10):
        self.time_steps = time_steps
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.modelName = ""

    def load_data(self, filename):
        try:
            df = pd.read_csv(filename)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from file: {e}")

    def preprocess_data(self, df):
        feature_cols = [
            "sales",
            "customers",
            "average_order_value",
            "customer_satisfaction",
        ]
        if not all(col in df.columns for col in feature_cols):
            raise ValueError(
                f"DataFrame must contain the following columns: {feature_cols}"
            )

        data = df[feature_cols].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - self.time_steps):
            X.append(scaled_data[i : i + self.time_steps])
            y.append(scaled_data[i + self.time_steps, 0])  # Predicting sales

        return np.array(X), np.array(y)

    def train(self, filename, epochs=50, batch_size=32):
        df = self.load_data(filename)
        X, y = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
        )

    def build_model(self, input_shape):
        # Define the model without Embedding layer for time-series data
        input = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(input)
        x = Dropout(0.5)(x)
        x = LSTM(128)(x)
        x = Dropout(0.5)(x)
        output = Dense(1)(
            x
        )  # No activation as we're predicting a continuous value (sales)

        model = Model(inputs=input, outputs=output)
        model.compile(
            optimizer="adam", loss="mean_squared_error"
        )  # Use MSE for regression
        return model

    def save_model(self, filename):
        if self.model:
            self.model.save(f"{filename}.h5", save_format="h5")
            print(f"LSTM model saved as '{filename}.h5'.")
        else:
            raise ValueError("No model found to save.")

    def load_model(self, filename):
        try:
            self.model = keras.models.load_model(f"{filename}.h5")
            self.modelName = f"{filename}.h5"
            print(f"Model loaded successfully from '{filename}.h5'.")
        except Exception as e:
            raise IOError(f"Error loading model: {e}")

    def retrain_lstm(self, filename=None, data_buffer=None, epochs=50, batch_size=32):
        if filename is None and data_buffer is None:
            raise ValueError(
                "Must provide either a filename or data_buffer for retraining."
            )

        if filename is not None:
            # Load data from the specified file
            df = self.load_data(filename)
        elif data_buffer is not None:
            # Convert the data_buffer to DataFrame for preprocessing
            df = pd.DataFrame(data_buffer)
        else:
            raise ValueError("No valid data source provided for retraining.")

        # Preprocess the loaded or buffered data
        X, y = self.preprocess_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check if the model exists; if not, build it
        if self.model is None:
            self.model = self.build_model(
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
        from keras.optimizers import Adam

        self.model.compile(optimizer=Adam(), loss="mse")

        # Proceed with retraining the model
        self.model.fit(
            data_buffer,
            epochs=10,  # Or however many epochs you want
            batch_size=32,  # Example batch size
            verbose=1,
        )
        # Retrain the model
        self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
        )
        print("Model retrained successfully.")

        # Optional: Save the retrained model
        self.save_model("retrained_lstm_model")

    # def predict(self, input_data):
    #     try:
    #         # input_data_scaled = self.scaler.transform(input_data)
    #         input_data_reshaped = input_data.reshape(
    #             (1, self.time_steps, input_data.shape[1])
    #         )
    #         prediction = self.model.predict(input_data_reshaped)
    #         print(prediction, "\n\n\n\n\n\n\n\n\n")
    #         return self.scaler.inverse_transform([[prediction[0][0]]])[0][0]
    #     except Exception as e:
    #         raise ValueError(f"Error during prediction: {e}")
    def predict(self, input_data):
        try:
            # Check the shape of the input data before proceeding
            print("Input data shape before reshaping:", input_data.shape)

            # Ensure input_data has at least 2 dimensions
            if len(input_data.shape) < 2:
                raise ValueError("Input data must have at least 2 dimensions.")

            # Reshape the input data for LSTM (1 sample, time_steps, num_features)
            input_data_reshaped = input_data.reshape(
                (1, self.time_steps, input_data.shape[1])
            )
            print("Input data shape after reshaping:", input_data_reshaped.shape)

            # Make the prediction
            prediction = self.model.predict(input_data_reshaped)
            print("Prediction result:", prediction)

            return self.scaler.inverse_transform([[prediction[0][0]]])[0][0]
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")


if __name__ == "__main__":
    lstm_model = LSTMModel()
    try:
        lstm_model.train("mock_data.txt", epochs=5, batch_size=32)
        lstm_model.save_model("lstm_model")
    except Exception as e:
        print(f"An error occurred: {e}")
