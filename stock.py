import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import pandas as pd

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

# normalize
data_n = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(data_n)


# visualize graphs with price and volume
import matplotlib.pyplot as plt

data['Open'].plot(figsize=(12,6), title='TSLA Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Opening Price (USD)')
plt.show()

data_n['Open'].plot(figsize=(12,6), title='TSLA Normalized Opening Price Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Opening Price (USD)')
plt.show()

data_n['Volume'].plot(figsize=(12,6), title='TSLA Normalized Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Normalized Volume')
plt.show()



# ------------- LINEAR REGRESSION MODEL -------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

data_n = data_n.dropna()
X = data_n[features]
y = data_n[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

y_pred = model.predict(X)
x_indices = range(len(X))
plt.plot(x_indices, y_pred)
plt.plot(x_indices, y)
plt.xlabel('Feature')
plt.ylabel('Prediction')
plt.title('Predictions')





import numpy as np

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

from sklearn.preprocessing import MinMaxScaler

features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# norm
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[features])
y_scaled = scaler.fit_transform(data[[target]])

time_steps = 20

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=16)
y_pred = model.predict(X_test)

loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss}")

y_scaler = MinMaxScaler()
y_scaler.fit(data[[target]])
y_pred_inv = y_scaler.inverse_transform(y_pred)
y_test_inv = y_scaler.inverse_transform(y_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# not normalzied
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_inv, y_pred_inv)

print("Test RMSE:", rmse)
print("Test MAE:", mae)
print("Test R² Score:", r2)


plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Close Price')
plt.show()
