from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

'''
Solving for closing price
Weekday is known
Predict Volume only with weekday (and maybe with previous day's volume) and add random factor
Predict Low and High w/ previous close, volume, and weekday
Set open to Close t-1 with random factor
'''


def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# ---------- FEATURE LSTM MODELS ----------
features = ['Open t-1', 'High t-1', 'Low t-1', 'Volume t-1', 'Weekday']
timesteps = 30
epoch = 5

# open_scaler_X = MinMaxScaler()
# open_scaler_y = MinMaxScaler()
# X_open = open_scaler_X.fit_transform(data[features])
# y_open = open_scaler_y.fit_transform(data[['Open']])
# X_open_seq, y_open_seq = create_sequences(X_open, y_open, timesteps)
# split = int(0.8 * len(X_open_seq))
# X_open_train, X_open_test = X_open_seq[:split], X_open_seq[split:]
# y_open_train, y_open_test = y_open_seq[:split], y_open_seq[split:]
# model_open = Sequential()
# model_open.add(LSTM(50, activation='tanh', input_shape=(X_open_train.shape[1], X_open_train.shape[2])))
# model_open.add(Dense(1))
# model_open.compile(optimizer='adam', loss='mse')
# history = model_open.fit(X_open_train, y_open_train, epochs=epoch, validation_split=0.1, batch_size=16)

high_scaler_X = MinMaxScaler()
high_scaler_y = MinMaxScaler()
X_high = high_scaler_X.fit_transform(data[['Volume t-1', 'Weekday', 'Open t-1']])
y_high = high_scaler_y.fit_transform(data[['High t-1']])
X_high_seq, y_high_seq = create_sequences(X_high, y_high, timesteps)
split = int(0.8 * len(X_high_seq))
X_high_train, X_high_test = X_high_seq[:split], X_high_seq[split:]
y_high_train, y_high_test = y_high_seq[:split], y_high_seq[split:]
model_high = Sequential()
model_high.add(LSTM(50, activation='tanh', input_shape=(X_high_train.shape[1], X_high_train.shape[2])))
model_high.add(Dense(1))
model_high.compile(optimizer='adam', loss='mse')
history = model_high.fit(X_high_train, y_high_train, epochs=epoch, validation_split=0.1, batch_size=16)

low_scaler_X = MinMaxScaler()
low_scaler_y = MinMaxScaler()
X_low = low_scaler_X.fit_transform(data[['Volume t-1', 'Weekday', 'Open t-1']])
y_low = low_scaler_y.fit_transform(data[['Low t-1']])
X_low_seq, y_low_seq = create_sequences(X_low, y_low, timesteps)
split = int(0.8 * len(X_low_seq))
X_low_train, X_low_test = X_low_seq[:split], X_low_seq[split:]
y_low_train, y_low_test = y_low_seq[:split], y_low_seq[split:]
model_low = Sequential()
model_low.add(LSTM(50, activation='tanh', input_shape=(X_low_train.shape[1], X_low_train.shape[2])))
model_low.add(Dense(1))
model_low.compile(optimizer='adam', loss='mse')
history = model_low.fit(X_low_train, y_low_train, epochs=epoch, validation_split=0.1, batch_size=16)

volume_scaler_X = MinMaxScaler()
volume_scaler_y = MinMaxScaler()
X_volume = volume_scaler_X.fit_transform(data[['Weekday']])
y_volume = volume_scaler_y.fit_transform(data[['Volume t-1']])
X_volume_seq, y_volume_seq = create_sequences(X_volume, y_volume, timesteps)
split = int(0.8 * len(X_volume_seq))
X_volume_train, X_volume_test = X_volume_seq[:split], X_volume_seq[split:]
y_volume_train, y_volume_test = y_volume_seq[:split], y_volume_seq[split:]
model_volume = Sequential()
model_volume.add(LSTM(50, activation='tanh', input_shape=(X_volume_train.shape[1], X_volume_train.shape[2])))
model_volume.add(Dense(1))
model_volume.compile(optimizer='adam', loss='mse')
history = model_volume.fit(X_volume_train, y_volume_train, epochs=epoch, validation_split=0.1, batch_size=16)


# loss_open = model_open.evaluate(X_open_test, y_open_test)
loss_high = model_high.evaluate(X_high_test, y_high_test)
loss_low = model_low.evaluate(X_low_test, y_low_test)
loss_volume = model_volume.evaluate(X_volume_test, y_volume_test)
# print(f"Test Open MSE: {loss_open}")
print(f"Test High MSE: {loss_high}")
print(f"Test Low MSE: {loss_low}")
print(f"Test Volume MSE: {loss_volume}")
