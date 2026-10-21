# HW2 - Life Expectancy Prediction Using Multiple Linear Regression

本專題使用 WHO Life Expectancy Dataset，
依照 CRISP-DM 流程進行多元線性迴歸分析，
並將分析結果製作成 Web Dashboard 部署至 GitHub Pages。

## Student

- Student ID: 7115064186
- Name: 蕭丞皓

## Dataset

Life Expectancy (WHO)

原始資料包含 22 個欄位，
移除預測目標 Life expectancy 與 Country 後，
使用 20 個原始輸入特徵進行分析。

## Data Preparation

- 移除 Life expectancy 缺失資料
- Numerical features 使用 Median Imputation
- Status 使用 One-Hot Encoding
- Training / Testing = 80% / 20%

## Feature Selection

比較：

- All 20 Features
- SelectKBest → 14 Features
- RFE → 14 Features

最終採用 RFE 選出的 14 個特徵。

## Models

比較三種模型：

- Linear Regression
- Ridge Regression
- Lasso Regression

最終模型：

RFE 14 Features + Linear Regression

## Model Performance

| Metric | Result |
| --- | ---: |
| MAE | 2.9256 |
| MSE | 15.6035 |
| RMSE | 3.9501 |
| R² | 0.8196 |
| Adjusted R² | 0.8152 |

## Regression Assumptions

- Linearity: Partially Satisfied
- Independence: Satisfied
- Homoscedasticity: Not Satisfied
- Normality: Not Fully Satisfied
- Multicollinearity: Partially Satisfied

Durbin-Watson = 2.014

95% Prediction Interval Coverage = 94.37%

## Web Dashboard

GitHub Pages：

https://hank050392.github.io/HW2-Life-Expectancy-Regression/

## Project Files

- `7115064186_hw2.py`：完整分析程式
- `Life Expectancy Data.csv`：資料集
- `model_comparison.csv`：模型比較結果
- `prediction_interval_results.csv`：Prediction Interval 結果
- `vif_results.csv`：VIF 分析結果
- `index.html`：Web Dashboard
- `style.css`：網站樣式
- `script.js`：網站 JavaScript
- `figures/`：分析圖表