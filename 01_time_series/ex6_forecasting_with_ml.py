# ============================================================
# Exercise 6: Forecasting with Machine Learning
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : Ecuador Store Sales (33 product families)
# Concept : Multistep targets, DirRec strategy, RegressorChain
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

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
    .loc['2017']
)

test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
test['date'] = test.date.dt.to_period('D')
test = test.set_index(['store_nbr', 'family', 'date']).sort_index()

# ------------------------------------------------------------
# Q1: Match tasks to datasets
# task_a: 3-step, 4 lags, 2-step lead = Dataset 2
# task_b: 1-step, 3 lags, 1-step lead = Dataset 1
# task_c: 3-step, 4 lags, 1-step lead = Dataset 3
# ------------------------------------------------------------
task_a = 2
task_b = 1
task_c = 3

# ------------------------------------------------------------
# Q2: Store Sales forecasting task
# Training ends: 2017-08-15
# Test starts:   2017-08-16
# Forecast horizon: 16 steps
# Lead time: 1 day
# ------------------------------------------------------------

# ------------------------------------------------------------
# Q3: Create multistep dataset
# make_lags(y, 4): use past 4 days as input features
# make_multistep_target(y, 16): predict next 16 days
# align: drop rows where either X or y has NaN
# ------------------------------------------------------------
from learntools.time_series.utils import make_lags, make_multistep_target

y = family_sales.loc[:, 'sales']
X = make_lags(y, lags=4).dropna()
y = make_multistep_target(y, steps=16).dropna()
y, X = y.align(X, join='inner', axis=0)

# Label encode family for XGBoost
le = LabelEncoder()
X = (X
    .stack('family')
    .reset_index('family')
    .assign(family=lambda x: le.fit_transform(x.family))
)
y = y.stack('family')

# ------------------------------------------------------------
# Q4: DirRec strategy with RegressorChain
# Each step gets its own XGBoost model
# Previous step prediction becomes next step feature
# Better than Direct (no dependencies) or Recursive (error compounds)
# ------------------------------------------------------------
model = RegressorChain(base_estimator=XGBRegressor())
model.fit(X, y)

y_pred = pd.DataFrame(
    model.predict(X),
    index=y.index,
    columns=y.columns,
).clip(0.0)

print("Forecast shape:", y_pred.shape)
print("Steps forecasted:", y_pred.shape[1])
