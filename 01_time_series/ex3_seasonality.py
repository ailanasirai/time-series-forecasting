# ============================================================
# Exercise 3: Seasonality
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : Ecuador Store Sales 2017 + Holiday Events
# Concept : Weekly indicators, Fourier features, holiday dummies
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

comp_dir = Path('../input/store-sales-time-series-forecasting')

# ------------------------------------------------------------
# Load holiday events
# National and Regional holidays affect sales differently
# ------------------------------------------------------------
holidays_events = pd.read_csv(
    comp_dir / "holidays_events.csv",
    dtype={
        'type': 'category',
        'locale': 'category',
        'locale_name': 'category',
        'description': 'category',
        'transferred': 'bool',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
holidays_events = holidays_events.set_index('date').to_period('D')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()
average_sales = (
    store_sales
    .groupby('date').mean()
    .squeeze()
    .loc['2017']
)

# ------------------------------------------------------------
# Q1: Seasonality identification
# Weekly: sales vary by day of week (weekend effect)
# Monthly: sales vary within month (pay period effect)
# Periodogram shows peaks at weekly and monthly frequencies
# ------------------------------------------------------------

# ------------------------------------------------------------
# Q2: Create seasonal features
# seasonal=True: weekly dummy variables (Mon-Sun indicators)
# CalendarFourier(freq='M', order=4): 8 Fourier features
#   for monthly seasonality (handles unequal month lengths)
# order=1: include linear trend alongside seasonality
# drop=True: remove collinear columns automatically
# ------------------------------------------------------------
y = average_sales.copy()

fourier = CalendarFourier(freq='M', order=4)

dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X = dp.in_sample()

# Fit seasonal model
model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index, name='Fitted')

plt.figure()
y.plot(alpha=0.5, label='Actual Sales')
y_pred.plot(label='Seasonal Model')
plt.title("Average Sales — Seasonal Fit")
plt.ylabel("Items Sold")
plt.legend()
plt.savefig('ex3_seasonal_fit.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Deseasonalize: subtract seasonal predictions from actual
# Remaining signal = trend + holidays + residuals
# ------------------------------------------------------------
y_deseason = y - y_pred

# ------------------------------------------------------------
# Q4: Holiday features using pd.get_dummies
# Each holiday becomes a binary column (0 or 1)
# Model learns effect of each specific holiday on sales
# fillna(0.0): non-holiday dates get 0 for all holiday columns
# ------------------------------------------------------------
holidays = (
    holidays_events
    .query("locale in ['National', 'Regional']")
    .loc['2017':'2017-08-15', ['description']]
    .assign(description=lambda x: x.description.cat.remove_unused_categories())
)

X_holidays = pd.get_dummies(holidays)
X2 = X.join(X_holidays, on='date').fillna(0.0)

# Fit model with holidays
model2 = LinearRegression().fit(X2, y)
y_pred2 = pd.Series(model2.predict(X2), index=X2.index, name='Fitted')

plt.figure()
y.plot(alpha=0.5, label='Actual Sales')
y_pred2.plot(label='Seasonal + Holidays')
plt.title("Average Sales — Seasonal + Holiday Fit")
plt.ylabel("Items Sold")
plt.legend()
plt.savefig('ex3_holiday_fit.png', dpi=150)
plt.show()

print("Holiday features added:", X_holidays.shape[1])
print("Total features:", X2.shape[1])
