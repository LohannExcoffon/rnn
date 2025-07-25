import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# dicts for feature lstms trained above
feature_models = {'Open': model_open, 'High': model_high, 'Low': model_low, 'Volume': model_volume}
scalers = {'Open': {'X': open_scaler_X, 'y': open_scaler_y}, 'High': {'X': high_scaler_X, 'y': high_scaler_y}, 'Low': {'X': low_scaler_X, 'y': low_scaler_y}, 'Volume': {'X': volume_scaler_X, 'y': volume_scaler_y}}


# function to create sequences
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# take 80% of data for training
features = ['Open','High','Low','Volume']
target = 'Close'
timesteps = 20
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index + timesteps]
test_data  = data.iloc[split_index:]

# normalize data for lstm
scaler_X_main = MinMaxScaler()
scaler_y_main = MinMaxScaler()
X_main = scaler_X_main.fit_transform(train_data[features])
y_main = scaler_y_main.fit_transform(train_data[[target]])
X_seq_main, y_seq_main = create_sequences(X_main, y_main, timesteps) # create sequences for training

# train the lstm for Close value
model = Sequential()
model.add(LSTM(50, activation='tanh', input_shape=(X_seq_main.shape[1], X_seq_main.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_seq_main, y_seq_main, epochs=10, batch_size=16, verbose=1)

# predict future features then use to predict close values
# starting from last training window
predicted_features = []
window = train_data[features].values[-timesteps:]  # shape (timesteps, 4)
for i in range(len(test_data) - timesteps):
    # prepare input for each feature lstm
    next_feats = []
    for idx, feat in enumerate(features):
        scaler_X = scalers[feat]['X']
        scaler_y = scalers[feat]['y']
        window_df = pd.DataFrame(window, columns=features)
        norm_win = scaler_X.transform(window_df)
        input_win = norm_win.reshape(1, timesteps, len(features))
        pred_norm = feature_models[feat].predict(input_win, verbose=0)
        # de-normalize to append for next window
        pred_real = scaler_y.inverse_transform(pd.DataFrame(pred_norm, columns=[feat]))
        next_feats.append(pred_real)
    next_feats = np.array(next_feats).reshape(1, -1)
    predicted_features.append(next_feats.flatten())
    # update window for next prediction
    window = np.vstack([window[1:], next_feats])

predicted_features = np.array(predicted_features)  # shape (num_pred_points, 4)

# predict future Close prices using predicted features
X_test_main = []
window = train_data[features].values[-timesteps:]
for i in range(predicted_features.shape[0]):
    w = np.vstack([window, predicted_features[:i+1]])[-timesteps:]
    X_test_main.append(w)
X_test_main = np.array(X_test_main)

# norm w/ main scaler
num_pred_points = X_test_main.shape[0]
X_test_main_norm = np.zeros_like(X_test_main)
for i in range(num_pred_points):
    X_test_main_norm[i] = scaler_X_main.transform(X_test_main[i])

# predict close price on future features
y_pred_norm = model.predict(X_test_main_norm)
y_pred_close = scaler_y_main.inverse_transform(y_pred_norm)

# compare with real future close prices
actual_close = test_data[target].values[timesteps:]
plt.figure(figsize=(12, 6))
plt.plot(actual_close, label='Actual Close', color='blue')
plt.plot(y_pred_close.flatten(), label='Predicted Close (with predicted features)', color='red')
plt.title('Stacked LSTM: Close Price Prediction Using Predicted Features')
plt.xlabel('Time Step (Future)')
plt.ylabel('Close Price')
plt.legend()
plt.show()
