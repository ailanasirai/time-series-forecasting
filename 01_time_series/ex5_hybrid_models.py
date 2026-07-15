# ============================================================
# Exercise 5: Hybrid Models
# Course  : Time Series (Kaggle)
# Repo    : time-series-forecasting
# Dataset : Ecuador Store Sales 2017 (33 product families)
# Concept : BoostedHybrid class, residual learning, XGBoost
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor

comp_dir = Path('../input/store-sales-time-series-forecasting')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
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

family_sales = (
    store_sales
    .groupby(['family', 'date'])
    .mean()
    .unstack('family')
    .loc['2017']
)

# ------------------------------------------------------------
# BoostedHybrid class
# Model 1 (LinearRegression): captures trend
# Model 2 (XGBRegressor): captures residual patterns
# fit: train model_1, compute residuals, train model_2 on them
# predict: model_1 prediction + model_2 residual prediction
# ------------------------------------------------------------
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

def fit(self, X_1, X_2, y):
    self.model_1.fit(X_1, y)

    y_fit = pd.DataFrame(
        self.model_1.predict(X_1),
        index=X_1.index, columns=y.columns,
    )

    y_resid = y - y_fit
    y_resid = y_resid.stack().squeeze()

    self.model_2.fit(X_2, y_resid)

    self.y_columns = y.columns
    self.y_fit = y_fit
    self.y_resid = y_resid

def predict(self, X_1, X_2):
    y_pred = pd.DataFrame(
        self.model_1.predict(X_1),
        index=X_1.index, columns=self.y_columns,
    )
    y_pred = y_pred.stack().squeeze()
    y_pred += self.model_2.predict(X_2)
    return y_pred.unstack()

BoostedHybrid.fit = fit
BoostedHybrid.predict = predict

# ------------------------------------------------------------
# Prepare features
# X_1: linear trend features for Model 1
# X_2: onpromotion + family label + day of month for Model 2
# ------------------------------------------------------------
y = family_sales.loc[:, 'sales']

dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

X_2 = family_sales.drop('sales', axis=1).stack()
le = LabelEncoder()
X_2 = X_2.reset_index('family')
X_2['family'] = le.fit_transform(X_2['family'])
X_2["day"] = X_2.index.day

# ------------------------------------------------------------
# Train LinearRegression + XGBoost hybrid
# ------------------------------------------------------------
model = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=XGBRegressor(),
)
model.fit(X_1, X_2, y)
y_pred = model.predict(X_1, X_2).clip(0.0)

# ------------------------------------------------------------
# Train/valid split and plot 6 product families
# ------------------------------------------------------------
y_train, y_valid = y[:"2017-07-01"], y["2017-07-02":]
X1_train = X_1[:"2017-07-01"]
X1_valid = X_1["2017-07-02":]
X2_train = X_2.loc[:"2017-07-01"]
X2_valid = X_2.loc["2017-07-02":]

model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

families = y.columns[0:6]
axs = y.loc(axis=1)[families].plot(
    subplots=True, sharex=True, figsize=(11, 9), alpha=0.5,
)
_ = y_fit.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[families].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, families):
    ax.legend([])
    ax.set_ylabel(family)
plt.savefig('ex5_hybrid_forecast.png', dpi=150)
plt.show()
