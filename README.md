# Hybrid Symbolic-Machine Learning for Time Series Forecasting

## 📄 Abstract

This research project develops hybrid approaches combining gradient boosting algorithms (LightGBM, XGBoost), ARIMA models, and symbolic regression techniques to create adaptive time series forecasting models. The methodology integrates modern machine learning algorithms with traditional statistical models and symbolic regression to improve forecast accuracy and interpretability for financial time series data.

The hybrid approach leverages gradient boosting models and ARIMA to capture different aspects of time series patterns while using symbolic regression to model complex relationships and residuals, resulting in more accurate and interpretable forecasts.

## Structure
```
Hybrid prediction/  
├── data/  
│   ├── raw/  
├── notebooks/              # Jupyter notebooks  
├── charts/  
├── requirements.txt        # Требования к зависимостям  
└── README.md  
```

## Dataset
The dataset from the bybit website is being used.

https://www.bybit.com/en/trade/spot/ETH/USDT

## Requirements
* Python 3.11+
* dependencies from requirements.txt

## Fast start
```
#1. Install
pip install -r dependencies requirements.txt

# 2. Put data
mkdir data
cp ~/Downloads/filename.csv data/

#3. Run the full
python main.py
```

## 🎯 Background and Motivation
Time series forecasting for financial assets requires balancing predictive accuracy with adaptability. This project focuses on forecasting the daily price direction and next closing price of ETHUSDT (Ethereum/USDT) using gradient boosting models enhanced by symbolic regression — without relying on ARIMA or traditional statistical time series methods.
The core limitations being addressed:

Standalone gradient boosting (CatBoost, XGBoost): Strong at capturing complex non-linear patterns in technical indicators, but their residual errors may follow learnable patterns left unexploited.
Symbolic regression (PySR): Produces compact, interpretable mathematical expressions from data, but benefits greatly from a curated, domain-relevant feature space — such as the one derived from price action.
Hybrid combination: By feeding PySR-derived symbolic features into gradient boosting models, and separately modeling gradient boosting residuals with PySR, the system captures signal layers that neither approach finds alone.

## 🔬 Methodology
Data & Feature Engineering
The dataset is ETHUSDT daily OHLCV data starting from 2021, loaded from ETHUSDT_daily_interpolated_from_2021.csv. A rich feature set of 55+ technical indicators is constructed, including:

Multi-period returns (1d to 60d), volatility measures (ATR, rolling std), and volume pressure signals
RSI (7/14/21), MACD, Bollinger Bands, Stochastic oscillator, Williams %R
Moving average crossovers (SMA 5/20/50/200), 52-week distance metrics, momentum streaks
Interaction features: rsi_x_bb, squeeze_mom, vol_mom_pressure, trend_vs_rev, and others

Two targets are used: a binary classification target (up/down next day) and a regression target (next close price / log return).
Hybrid Model Architectures
The research implements three distinct hybrid approaches evaluated in a walk-forward (online) testing framework:
1. Gradient Boosting + PySR Feature Augmentation (GB+PySR)
PySR is trained on a rolling 720-day window to discover symbolic expressions for log returns, volatility, and momentum. The top-K non-redundant equations (filtered by Spearman IC ≥ 0.02) are blended and used as additional features for CatBoost and XGBoost classifiers and regressors. Both CB+PySR and XB+PySR variants are evaluated.
2. Gradient Boosting + Symbolic Residual Modeling (GB+res)
Base CatBoost and XGBoost regressors are trained to predict next-day close price. Their prediction residuals on a validation window are then modeled by PySR using a lighter search configuration (PYSR_RESID_P). Final prediction: price_forecast = gb_forecast + pysr_residual. Variants: CB+res, XB+res.
3. Hybrid Classification-Regression Blending (Hyb CB/XB)
Direction probability from a classifier and directional signal derived from a price regressor are blended with an optimized weight w calibrated on a rolling 90-day window. This blending uses shrinkage toward 0.5 for small sample sizes. Variants: Hyb CB, Hyb CB+, Hyb XB, Hyb XB+ (with and without PySR features).
Walk-Forward Evaluation
All models are evaluated in a strictly time-ordered online loop over the 2025 test period. Gradient boosting models are retrained every 10 steps; PySR is retrained every 30 steps on the most recent data window. Blending weights are recalibrated each retraining cycle.
Evaluation Metrics
Classification: Directional Accuracy (DA) — percentage of correct up/down predictions
Regression: R², RMSE, MAE, MAPE, SMAPE, Log-Cosh, Information Coefficient (IC), Directional Accuracy (DA), Top/Bottom Quintile Hit Rate (QHR@20%)

## 📊 Results
The ten model variants are compared across all metrics, with the best DA model marked with ★. Regression results are reported for: CB, CB+PySR, XB, XB+PySR, CB+res, XB+res. The charts can be found in charts/.

## Key Findings

* PySR feature augmentation consistently improves CatBoost and sometimes XGBoost.
* Residual modeling with PySR provides improvment IC on log returns for XGBoost.
* Hybrid blending of classifier probability and regressor signal outperforms pure classification approaches, with the optimal blend weight adapted per market regime (bull/bear).

## Author
Ivan Zevakin
