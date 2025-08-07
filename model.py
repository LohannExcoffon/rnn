import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# dicts for feature lstms trained above
feature_models = {'High t-1': model_high, 'Low t-1': model_low, 'Volume t-1': model_volume}
scalers = {'High t-1': {'X': high_scaler_X, 'y': high_scaler_y}, 'Low t-1': {'X': low_scaler_X, 'y': low_scaler_y}, 'Volume t-1': {'X': volume_scaler_X, 'y': volume_scaler_y}}

# function to create sequences
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

# take 80% of data for training
features = ['Open t-1', 'High t-1', 'Low t-1', 'Volume t-1', 'Weekday']
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
model.fit(X_seq_main, y_seq_main, epochs=10, batch_size=50, verbose=1)

# predict future features then use to predict close values
predicted_closes = []
generated_features = [] #preallocated for predicted future features
window = train_data[features].values[-timesteps:].copy() # last training window

for step in range(len(test_data) - timesteps):
    next_feature_row = [] #preallocated for current step
    for feat in features:
        # get each feature from their respective lstm or logic
        if feat == 'High t-1':
            input_feats = ['Volume t-1', 'Weekday', 'Open t-1']
            feat_model = feature_models['High t-1']
            feat_scaler_X = scalers['High t-1']['X']
            feat_scaler_y = scalers['High t-1']['y']
        elif feat == 'Low t-1':
            input_feats = ['Volume t-1', 'Weekday', 'Open t-1']
            feat_model = feature_models['Low t-1']
            feat_scaler_X = scalers['Low t-1']['X']
            feat_scaler_y = scalers['Low t-1']['y']
        elif feat == 'Volume t-1':
            input_feats = ['Weekday']
            feat_model = feature_models['Volume t-1']
            feat_scaler_X = scalers['Volume t-1']['X']
            feat_scaler_y = scalers['Volume t-1']['y']
        else:
            # plug previous Close for Open t-1
            if feat == 'Open t-1':
                if predicted_closes:
                    prev_close = predicted_closes[-1]
                else:
                    prev_close = window[-1, features.index('Open t-1')]
                next_feature_row.append(prev_close)
            # use prev weekday and log 5 to find next weekday
            if feat == 'Weekday':
                #NEED TO FIX FOR HOLIDAYS??? MAYBE USE INDEX DATES INSTEAD\
                # because if i start ignoring days off than they will add up significantly and my weekdays
                # will be wrong and the volume freature predictions will be bad too
                prev_day = window[-1, features.index('Weekday')] 
                day = (prev_day + 1) % 5 #mod 5 for weekdays
                next_feature_row.append(day)
            continue

        # prep the input window and get relevant features
        idxs = [features.index(f) for f in input_feats]
        window_part = window[:, idxs]
        input_norm = feat_scaler_X.transform(pd.DataFrame(window_part, columns=input_feats))
        input_norm = input_norm.reshape(1, timesteps, len(input_feats))
        pred_norm = feat_model.predict(input_norm, verbose=0)
        pred_real = feat_scaler_y.inverse_transform(pred_norm)
        next_feature_row.append(pred_real[0, 0])

    # add features to total
    window = np.vstack([window[1:], next_feature_row])
    generated_features.append(next_feature_row)

    # prredict the Close for newest row
    window_norm = scaler_X_main.transform(pd.DataFrame(window, columns=features))
    x_input = window_norm.reshape(1, timesteps, len(features))
    y_pred_norm = model.predict(x_input, verbose=0)
    y_pred_close = scaler_y_main.inverse_transform(y_pred_norm)[0, 0]
    predicted_closes.append(y_pred_close)

# compare with real future close prices
actual_close = test_data[target].values[timesteps:]
plt.figure(figsize=(12, 6))
plt.plot(actual_close, label='Actual Close', color='blue')
plt.plot(predicted_closes, label='Predicted Close (with predicted features)', color='red')
plt.title('Stacked LSTM: Close Price Prediction Using Predicted Features')
plt.xlabel('Time Step (Future)')
plt.ylabel('Close Price')
plt.legend()
plt.show()
