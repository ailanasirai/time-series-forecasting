# ============================================================
# Exercise 2: Trend
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : US Retail Sales + Ecuador Store Sales
# Concept : Moving average, DeterministicProcess, polynomial trend
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess

# ------------------------------------------------------------
# Load datasets
# retail_sales: US monthly retail sales by category
# food_sales: Food & Beverage subset
# average_sales: Ecuador store daily average
# ------------------------------------------------------------
data_dir = Path('../input/ts-course-data/')
comp_dir = Path('../input/store-sales-time-series-forecasting')

retail_sales = pd.read_csv(
    data_dir / "us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month',
).to_period('D')
food_sales = retail_sales.loc[:, 'FoodAndBeverage']

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
# Q1: Moving average on Food & Beverage sales
# window=12: monthly data, 12 months = 1 full year
# center=True: window centered on current point (not lagged)
# min_periods=6: allow partial windows at series boundaries
# Result: seasonal variation removed, long-term trend visible
# ------------------------------------------------------------
trend = food_sales.rolling(
    window=12,
    center=True,
    min_periods=6,
).mean()

plt.figure()
food_sales.plot(alpha=0.5, label='Food Sales')
trend.plot(linewidth=3, label='Trend (12-month MA)')
plt.title("US Food and Beverage Sales — Trend")
plt.ylabel("Millions of Dollars")
plt.legend()
plt.savefig('ex2_food_trend.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q2: Polynomial order for Food Sales trend
# Linear: too simple, growth accelerates over time
# Quadratic: better, captures curve
# Exponential: best — compound growth fits food sales naturally
# ------------------------------------------------------------

# ------------------------------------------------------------
# Average sales moving average (365-day window)
# 365 days = 1 year to remove annual seasonality
# min_periods=183: half-year minimum for partial windows
# ------------------------------------------------------------
trend = average_sales.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()

plt.figure()
average_sales.plot(alpha=0.5, label='Average Sales')
trend.plot(linewidth=3, label='Trend (365-day MA)')
plt.title("Ecuador Store Sales — 365-day Trend")
plt.savefig('ex2_average_trend.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q3: DeterministicProcess — cubic trend (order=3)
# Creates features: const, t, t^2, t^3
# in_sample(): features for training dates
# out_of_sample(steps=90): features for 90 days ahead
# drop=True: removes collinear features automatically
# ------------------------------------------------------------
y = average_sales.copy()

dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=3,
    drop=True,
)

X = dp.in_sample()
X_fore = dp.out_of_sample(steps=90)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

plt.figure()
y.plot(alpha=0.5, label='Actual Sales')
y_pred.plot(linewidth=3, label='Cubic Trend', color='C0')
y_fore.plot(linewidth=3, label='90-day Forecast', color='C3')
plt.title("Average Sales — Cubic Trend + Forecast")
plt.ylabel("Items Sold")
plt.legend()
plt.savefig('ex2_cubic_forecast.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q4: High-order polynomial (order=11) — overfitting risk
# Fits training data perfectly (wiggles to match every point)
# But extrapolates wildly outside training range
# 90-day forecast goes to unrealistic values
# Rule: low-order polynomials for forecasting, not high-order
# ------------------------------------------------------------
dp11 = DeterministicProcess(index=y.index, order=11)
X11 = dp11.in_sample()

model11 = LinearRegression()
model11.fit(X11, y)
y_pred11 = pd.Series(model11.predict(X11), index=X11.index)

X_fore11 = dp11.out_of_sample(steps=90)
y_fore11 = pd.Series(model11.predict(X_fore11), index=X_fore11.index)

plt.figure()
y.plot(alpha=0.5, label='Actual Sales')
y_pred11.plot(linewidth=3, label='Order-11 Trend', color='C0')
y_fore11.plot(linewidth=3, label='Order-11 Forecast', color='C3')
plt.title("Average Sales — Order 11 Polynomial (Overfitting)")
plt.ylabel("Items Sold")
plt.legend()
plt.savefig('ex2_order11_forecast.png', dpi=150)
plt.show()
