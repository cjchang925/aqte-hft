from data import prepare_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import Callback
import time


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_time_start)
        mean_epoch_time = np.mean(self.epoch_times)
        remaining_time = mean_epoch_time * (self.params["epochs"] - epoch - 1)
        print(
            f"Epoch {epoch + 1}/{self.params['epochs']} - Time: {self.epoch_times[-1]:.2f}s - Remaining time: {remaining_time / 60:.2f} min"
        )


def train():
    """
    Using 4h candles in the past year to train a LSTM model used for Bitcoin price prediction.
    """
    print("Start getting data")

    data = prepare_data()

    print("Data preparation completed, start preprocessing data")

    # Normalize the features
    scaler = MinMaxScaler()

    # Fit the scaler on the data
    scaler.fit(data)

    # Transform the data
    data_normalized = scaler.transform(data)

    # Create sequences for LSTM model
    sequence_length = 30
    X, y = [], []

    for i in range(len(data_normalized) - sequence_length):
        X.append(data_normalized[i : i + sequence_length])
        y.append(
            data_normalized[i + sequence_length][3]
        )  # Using 'Close' price as the target

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and test sets
    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print("Data preprocessing completed, start training model")

    # Build the LSTM model
    model = Sequential(
        [
            LSTM(
                50,
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2]),
            ),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    time_callback = TimeHistory()

    # Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=300,
        validation_data=(X_test, y_test),
        callbacks=[time_callback],
    )

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)

    print(f"Test loss: {loss}")

    # Save the model
    model.save("bitcoin_epoch300_segment30.h5")


if __name__ == "__main__":
    train()
