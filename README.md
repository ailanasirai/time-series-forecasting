# time-series-forecasting

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![StatsModels](https://img.shields.io/badge/StatsModels-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

> Time series is not just another regression problem. The past is a feature. The sequence is the signal.

Six exercises. Real datasets. From linear regression on time dummies to hybrid models combining LinearRegression and XGBoost with a custom-built BoostedHybrid class.

---

## Course Completed

| Course | Exercises | Certificate |
|--------|-----------|-------------|
| [Time Series](https://www.kaggle.com/learn/time-series) | 6 | ✅ Earned |

**Competition:** Store Sales — Time Series Forecasting (Kaggle)

---

## Exercise Breakdown

| # | File | Dataset | Problem | Approach |
|---|------|---------|---------|----------|
| 1 | `ex1_linear_regression.py` | Ecuador Store Sales 2013-2017 + Book Sales | Predict average daily sales using time position and past values | Time dummy (`np.arange`) for trend direction. Lag feature (`shift(1)`) for serial dependence. Linear regression on both. |
| 2 | `ex2_trend.py` | US Retail Food Sales + Ecuador Store Sales | Extract long-term trend from noisy sales data | 12-month moving average on monthly food sales. `DeterministicProcess` with order=3 for cubic trend. 90-day forecast. Showed why high-order polynomials (order=11) fail at forecasting despite fitting training data well. |
| 3 | `ex3_seasonality.py` | Ecuador Store Sales 2017 + Holiday Events | Model weekly and monthly seasonal patterns, holiday effects | Weekly indicators via `seasonal=True`. `CalendarFourier(freq='M', order=4)` for monthly Fourier features. `pd.get_dummies` on national and regional Ecuador holidays. Deseasonalized series using periodogram to confirm remaining signal. |
| 4 | `ex4_time_series_as_features.py` | Ecuador School and Office Supplies Sales | Model cyclic behavior after removing trend and seasonality | PACF showed lag 1 and lag 2 significant. `make_lags` for serial features. `make_leads` for onpromotion (valid — company knows future promotions). Rolling 14-day median and 7-day std on lagged sales to avoid lookahead leakage. |
| 5 | `ex5_hybrid_models.py` | Ecuador Store Sales 2017 (33 families) | Combine linear trend model with gradient boosting on residuals | Built `BoostedHybrid` class from scratch with fit and predict methods. Model 1 (LinearRegression) captures trend. Model 2 (XGBRegressor) trains on residuals. Also tested Ridge + KNeighborsRegressor. |
| 6 | `ex6_forecasting_with_ml.py` | Ecuador Store Sales (full dataset) | 16-step multistep forecast with 1-day lead time | `make_multistep_target` for 16-step output. `RegressorChain(XGBRegressor)` for DirRec strategy — separate model per step, previous step prediction as next step feature. |

---

## Key Insights

**What regular ML misses in time series:**
Regular ML treats rows as independent. Time series does not. Yesterday's sales, last week's pattern, and tomorrow's scheduled promotion are all valid signals that standard models ignore.

**Trend vs Seasonality vs Cycles:**
Trend is the long-term direction. Seasonality is patterns that repeat at fixed intervals (weekly, monthly). Cycles are patterns that repeat but at irregular intervals — driven by business factors, not the calendar. All three need different modeling approaches.

**Lookahead leakage:**
Rolling statistics on future values leak information the model should not have at prediction time. `shift(1)` before any rolling calculation prevents this. Exception: features where future values are genuinely known in advance (scheduled promotions, upcoming holidays).

**Why high-order polynomials fail at forecasting:**
Order-11 polynomial fits training data almost perfectly. Outside the training range, it extrapolates wildly. Low-order polynomials generalize; high-order polynomials memorize.

**DirRec vs Direct vs Recursive:**
Direct trains one model per forecast step independently — ignores dependencies between steps. Recursive uses one model repeatedly — compounds errors. DirRec trains one model per step but feeds previous predictions as features — captures dependencies without compounding errors.

**BoostedHybrid logic:**
```
Model 1 fits trend (what the series is doing slowly)
Residuals = actual - trend predictions
Model 2 fits residuals (what Model 1 missed)
Final prediction = Model 1 output + Model 2 output
```

---

## Stack

```
Python 3.10
pandas
numpy
scikit-learn
statsmodels
xgboost
matplotlib
```

---

## Repository Structure

```
time-series-forecasting/
│
├── 01_time_series/
│   ├── ex1_linear_regression.py
│   ├── ex2_trend.py
│   ├── ex3_seasonality.py
│   ├── ex4_time_series_as_features.py
│   ├── ex5_hybrid_models.py
│   └── ex6_forecasting_with_ml.py
│
├── LICENSE
└── README.md
```

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aila-nasir)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/ailanasirai)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ailanasirai)
