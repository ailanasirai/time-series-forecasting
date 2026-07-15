# ============================================================
# Exercise 4: Time Series as Features
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : Ecuador Store Sales — School and Office Supplies
# Concept : Lag features, leading indicators, rolling statistics
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

comp_dir = Path('../input/store-sales-time-series-forecasting')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017', ['sales', 'onpromotion']]
)

# ------------------------------------------------------------
# Isolate School and Office Supplies
# Deseasonalize to expose cyclic patterns
# ------------------------------------------------------------
supply_sales = family_sales.loc(axis=1)[:, 'SCHOOL AND OFFICE SUPPLIES']
y = supply_sales.loc[:, 'sales'].squeeze()

fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    constant=True,
    index=y.index,
    order=1,
    seasonal=True,
    drop=True,
    additional_terms=[fourier],
)
X_time = dp.in_sample()
X_time['NewYearsDay'] = (X_time.index.dayofyear == 1)

model = LinearRegression(fit_intercept=False)
model.fit(X_time, y)
y_deseason = y - model.predict(X_time)
y_deseason.name = 'sales_deseasoned'

# ------------------------------------------------------------
# Q1: 7-day moving average to visualize cycles
# window=7: smooth weekly seasonality
# center=True: window centered on current date
# No min_periods: NaN at edges is acceptable
# ------------------------------------------------------------
y_ma = y.rolling(window=7, center=True).mean()

plt.figure()
y_ma.plot()
plt.title("Seven-Day Moving Average — Supply Sales")
plt.savefig('ex4_moving_average.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q4: Lag and lead features
# make_lags(y_deseason, 1): yesterday's deseasonalized sales
# make_leads(onpromotion, 1): tomorrow's promotion count
# Company knows future promotions — not lookahead leakage
# ------------------------------------------------------------
from learntools.time_series.utils import make_lags, make_leads

onpromotion = supply_sales.loc[:, 'onpromotion'].squeeze().rename('onpromotion')

X_lags = make_lags(y_deseason, lags=1)

X_promo = pd.concat([
    make_lags(onpromotion, lags=1),
    onpromotion,
    make_leads(onpromotion, leads=1),
], axis=1)

X = pd.concat([X_lags, X_promo], axis=1)
y, X = y.align(X, join='inner')

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=30, shuffle=False)

model = LinearRegression(fit_intercept=False).fit(X_train, y_train)
y_fit = pd.Series(model.predict(X_train), index=X_train.index).clip(0.0)
y_pred = pd.Series(model.predict(X_valid), index=X_valid.index).clip(0.0)

rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
rmsle_valid = mean_squared_log_error(y_valid, y_pred) ** 0.5
print(f'Training RMSLE: {rmsle_train:.5f}')
print(f'Validation RMSLE: {rmsle_valid:.5f}')

plt.figure()
y.plot(alpha=0.5, label='Actual')
y_fit.plot(label='Fitted', color='C0')
y_pred.plot(label='Forecast', color='C3')
plt.title("School Supplies Sales — Lag + Promo Features")
plt.legend()
plt.savefig('ex4_forecast.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q5: Rolling statistics
# shift(1) first to avoid lookahead leakage on sales
# center=True for promotions only (future promo known in advance)
# ------------------------------------------------------------
y_lag = supply_sales.loc[:, 'sales'].shift(1)
onpromo = supply_sales.loc[:, 'onpromotion']

mean_7   = y_lag.rolling(7).mean()
median_14 = y_lag.rolling(14).median()
std_7    = y_lag.rolling(7).std()
promo_7  = onpromo.rolling(7, center=True).sum()

print("Rolling features created:")
print(f"  mean_7 shape:    {mean_7.shape}")
print(f"  median_14 shape: {median_14.shape}")
print(f"  std_7 shape:     {std_7.shape}")
print(f"  promo_7 shape:   {promo_7.shape}")
