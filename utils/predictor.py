import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler = MinMaxScaler(feature_range=(0, 1))

def train_model(df):

    data = df["Close"].values.reshape(-1, 1)

    scaled_data = scaler.fit_transform(data)

    x_train = []
    y_train = []

    for i in range(60, len(scaled_data)):

        x_train.append(scaled_data[i-60:i, 0])

        y_train.append(scaled_data[i, 0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = np.reshape(
        x_train,
        (x_train.shape[0], x_train.shape[1], 1)
    )

    model = tf.keras.models.Sequential()

    model.add(
        tf.keras.layers.LSTM(
            50,
            return_sequences=True,
            input_shape=(x_train.shape[1], 1)
        )
    )

    model.add(
        tf.keras.layers.LSTM(50)
    )

    model.add(
        tf.keras.layers.Dense(25)
    )

    model.add(
        tf.keras.layers.Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    model.fit(
        x_train,
        y_train,
        batch_size=16,
        epochs=1,
        verbose=0
    )

    return model, scaled_data