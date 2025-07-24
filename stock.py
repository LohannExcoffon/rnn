import yfinance as yf
data = yf.download('TSLA', start='2020-06-28', end='2025-07-22')
data.columns = data.columns.get_level_values(0)
print(data.head())

# norm
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
import pandas as pd
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
