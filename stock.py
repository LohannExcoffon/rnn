import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np

path = "data.parquet"

def update_data(path, threshold=5):
    today = datetime.today().date()

    if os.path.exists(path):
        data = pd.read_parquet(path)
        last_date = data.index.max().date()
        delta = (today - last_date).days

        if delta > threshold:
            new_data = yf.download('TSLA', start=last_date.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"))
            if not new_data.empty:
                new_data.columns = new_data.columns.get_level_values(0)
                clean_data(new_data)
                # Append and remove duplicates
                updated_data = pd.concat([data, new_data])
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                updated_data.to_parquet(path)
                return updated_data
            else:
                return data
        else:
            return data
    else:
        print("Data file not found")
        data = yf.download('TSLA', start='2020-06-28', end=today.strftime("%Y-%m-%d"))
        data.columns = data.columns.get_level_values(0)
        clean_data(data)
        data.to_parquet(path)
        return data

def clean_data(data):
    data['Weekday'] = ''
    for index, row in data.iterrows():
        data.at[index, 'Weekday'] = index.weekday()

# ---------- GET DATA -------------
data = update_data("data.parquet")
print(data.head())

# visualize graphs with price and volume
data['Open'].plot(figsize=(12,6), title='TSLA Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price (USD)')
plt.show()

data['Volume'].plot(figsize=(12,6), title='TSLA Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()



# ------------- LINEAR REGRESSION MODEL -------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

features = ['Open', 'High', 'Low', 'Volume', 'Weekday']
features_to_scale = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

scale_X = MinMaxScaler()
scale_y = MinMaxScaler()

data = data.dropna()
X = data[features].copy()
y = data[target]
X[features_to_scale] = scale_X.fit_transform(data[features_to_scale])
y = scale_y.fit_transform(data[['Close']]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

def linearRegression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(results.head())
    
    # plot entire stock w/ predictions
    y_pred = model.predict(X)
    x_indices = range(len(X))
    plt.plot(x_indices, y_pred)
    plt.plot(x_indices, y)
    plt.xlabel('Time Step')
    plt.ylabel('Prediction')
    plt.title('Predictions')


# ------------- LSTM MODEL -------------
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def lstm():
    time_steps = 20
    
    X_seq, y_seq = create_sequences(X, y, time_steps)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=16)
    y_pred = model.predict(X_test)
    
    loss = model.evaluate(X_test, y_test)
    print(f"Test MSE: {loss}")
    
    y_scaler = MinMaxScaler()
    y_scaler.fit(data[[target]])
    y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print("Test MSE:", mse)

    # plot
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Close Price')
    plt.show()

linearRegression()
lstm()
