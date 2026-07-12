# ============================================================
# Exercise 1: Linear Regression with Time Series
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : Book Sales + Store Sales (Ecuador retail)
# Concept : Time dummy, lag features, serial dependence
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------
# Load datasets
# book_sales: Hardcover sales with Time and Lag_1 features
# store_sales: Ecuador grocery store sales 2013-2017
# average_sales: daily average across all stores/families
# ------------------------------------------------------------
data_dir = Path('../input/ts-course-data/')
comp_dir = Path('../input/store-sales-time-series-forecasting')

book_sales = pd.read_csv(
    data_dir / 'book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
book_sales['Time'] = np.arange(len(book_sales.index))
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])

ar = pd.read_csv(data_dir / 'ar.csv')

dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']

# ------------------------------------------------------------
# Q1: Time dummy — linear trend
# Equation: Hardcover = 3.33 * Time + 150.5
# Over 6 days: 3.33 * 6 = ~20 units change on average
# ------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None,
                 scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')
plt.savefig('ex1_hardcover_trend.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q2: Lag feature — serial dependence
# weight = 0.95: next value likely same sign (smooth series)
# weight = -0.95: next value likely opposite sign (oscillating)
# Series 1 = 0.95, Series 2 = -0.95
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'])
ax1.set_title('Series 1 — weight = 0.95 (same sign)')
ax2.plot(ar['ar2'])
ax2.set_title('Series 2 — weight = -0.95 (opposite sign)')
plt.savefig('ex1_ar_series.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q3: Fit time-step feature to Store Sales
# time = 0, 1, 2, 3... (position in sequence)
# Linear regression fits a straight trend line to average sales
# ------------------------------------------------------------
df = average_sales.to_frame()
time = np.arange(len(df.index))
df['time'] = time

X = df.loc[:, ['time']]
y = df.loc[:, 'sales']

model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

print("Time feature model:")
print(f"  Coefficient: {model.coef_[0]:.4f}")
print(f"  Intercept:   {model.intercept_:.4f}")

# ------------------------------------------------------------
# Q4: Fit lag feature to Store Sales
# lag_1 = yesterday's sales as today's feature
# shift(1) moves values down by 1 row
# Strong positive correlation = sales trend is persistent
# ------------------------------------------------------------
df = average_sales.to_frame()
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1

X = df.loc[:, ['lag_1']].dropna()
y = df.loc[:, 'sales']
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

print("\nLag feature model:")
print(f"  Coefficient: {model.coef_[0]:.4f}")
print(f"  Intercept:   {model.intercept_:.4f}")

fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1',
       title='Lag Plot of Average Sales')
plt.savefig('ex1_lag_plot.png', dpi=150)
plt.show()
