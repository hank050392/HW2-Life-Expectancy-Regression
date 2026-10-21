import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================
# 1. Data Understanding
# ==============================

# 讀取資料
df = pd.read_csv("Life Expectancy Data.csv")

print("===== 前 5 筆資料 =====")
print(df.head())

print("\n===== 資料大小 =====")
print(df.shape)

print("\n===== 欄位名稱 =====")
print(df.columns.tolist())

print("\n===== 資料型態 =====")
print(df.dtypes)

print("\n===== 缺失值數量 =====")
print(df.isnull().sum())

print("\n===== 基本統計資料 =====")
print(df.describe())

print("\n===== 重複資料筆數 =====")
print(df.duplicated().sum())

# ==============================
# 2. Data Preparation
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 清除欄位名稱前後空白
df.columns = df.columns.str.strip()

# Target
target = "Life expectancy"

# Target 缺失不能拿來訓練，因此直接刪除
df = df.dropna(subset=[target]).reset_index(drop=True)

print("\n===== 刪除 Target 缺失值後 =====")
print("資料筆數：", len(df))


# Country 比較接近識別資訊，而且 One-Hot 後會產生大量欄位
# 因此不納入模型
X = df.drop(columns=[target, "Country"])
y = df[target]

print("\n===== 原始輸入特徵 =====")
print(X.columns.tolist())

print("\n原始輸入特徵數量：", X.shape[1])


# ==============================
# 3. Train / Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n===== Train / Test Split =====")
print("Training：", X_train.shape)
print("Testing ：", X_test.shape)


# ==============================
# 4. Preprocessing
# ==============================

categorical_features = ["Status"]

numerical_features = [
    col for col in X.columns
    if col not in categorical_features
]

# 數值欄位：使用 training data 的 median 補缺失值
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# 類別欄位：
# 先用最常出現的值補缺失，再 One-Hot Encoding
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# 只能用 Training Data fit
X_train_processed = preprocessor.fit_transform(X_train)

# Test Data 只能 transform
X_test_processed = preprocessor.transform(X_test)


print("\n===== Preprocessing 完成 =====")
print("處理前 Training：", X_train.shape)
print("處理後 Training：", X_train_processed.shape)

print("處理前 Testing ：", X_test.shape)
print("處理後 Testing ：", X_test_processed.shape)

print("\n===== 處理後是否還有缺失值 =====")
print("Training NaN：", np.isnan(X_train_processed).sum())
print("Testing NaN ：", np.isnan(X_test_processed).sum())

# ==============================
# 5. Feature Selection
# ==============================

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

# 取得前處理後的特徵名稱
feature_names = preprocessor.get_feature_names_out()

# 把名稱整理得比較乾淨
feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names
]

print("\n===== 處理後特徵名稱 =====")
print(feature_names)

print("\n處理後特徵數量：", len(feature_names))


# ------------------------------
# 5.1 SelectKBest
# ------------------------------

kbest = SelectKBest(
    score_func=f_regression,
    k=14
)

X_train_kbest = kbest.fit_transform(
    X_train_processed,
    y_train
)

X_test_kbest = kbest.transform(
    X_test_processed
)

kbest_mask = kbest.get_support()

kbest_features = [
    feature_names[i]
    for i in range(len(feature_names))
    if kbest_mask[i]
]

print("\n===== SelectKBest 選出的 14 個特徵 =====")

for i, feature in enumerate(
    kbest_features,
    start=1
):
    print(f"{i}. {feature}")


# ------------------------------
# 5.2 RFE
# ------------------------------

rfe_estimator = LinearRegression()

rfe = RFE(
    estimator=rfe_estimator,
    n_features_to_select=14
)

X_train_rfe = rfe.fit_transform(
    X_train_processed,
    y_train
)

X_test_rfe = rfe.transform(
    X_test_processed
)

rfe_mask = rfe.get_support()

rfe_features = [
    feature_names[i]
    for i in range(len(feature_names))
    if rfe_mask[i]
]

print("\n===== RFE 選出的 14 個特徵 =====")

for i, feature in enumerate(
    rfe_features,
    start=1
):
    print(f"{i}. {feature}")


# ------------------------------
# 5.3 特徵數量確認
# ------------------------------

print("\n===== Feature Selection Shapes =====")

print(
    "All Features：",
    X_train_processed.shape
)

print(
    "SelectKBest：",
    X_train_kbest.shape
)

print(
    "RFE：",
    X_train_rfe.shape
)

# ==============================
# 6. Modeling & Evaluation
# ==============================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Adjusted R²
def adjusted_r2_score(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# 三種 Feature Selection 結果
feature_sets = {
    "All 20 Features": (
        X_train_processed,
        X_test_processed
    ),

    "SelectKBest 14": (
        X_train_kbest,
        X_test_kbest
    ),

    "RFE 14": (
        X_train_rfe,
        X_test_rfe
    )
}


results = []


for feature_set_name, (Xtr, Xte) in feature_sets.items():

    # --------------------------
    # Standardization
    # --------------------------
    # 只能使用 Training Data fit
    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)


    # --------------------------
    # Models
    # --------------------------
    models = {
        "Linear Regression":
            LinearRegression(),

        "Ridge":
            RidgeCV(
                alphas=np.logspace(-3, 3, 30),
                cv=5
            ),

        "Lasso":
            LassoCV(
                alphas=np.logspace(-4, 1, 60),
                cv=5,
                max_iter=50000
            )
    }


    for model_name, model in models.items():

        # Train
        model.fit(
            Xtr_scaled,
            y_train
        )

        # Predict
        y_pred_model = model.predict(
            Xte_scaled
        )


        # --------------------------
        # Evaluation Metrics
        # --------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred_model
        )

        mse = mean_squared_error(
            y_test,
            y_pred_model
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred_model
        )

        n = len(y_test)
        p = Xte.shape[1]

        adj_r2 = adjusted_r2_score(
            r2,
            n,
            p
        )


        # Alpha
        if hasattr(model, "alpha_"):
            alpha = model.alpha_
        else:
            alpha = "-"


        results.append({
            "Feature Set": feature_set_name,
            "Model": model_name,
            "Features": p,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted R2": adj_r2,
            "Best Alpha": alpha
        })


# ==============================
# 7. Comparison Table
# ==============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

print("\n===== Model Comparison =====")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 儲存結果
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n已儲存：model_comparison.csv")

# ==============================
# 8. Regression Assumptions
# Best Model: RFE 14 + Linear Regression
# ==============================

os.makedirs("figures", exist_ok=True)

# 將 RFE 選出的資料轉為 DataFrame
X_train_best = pd.DataFrame(
    X_train_rfe,
    columns=rfe_features,
    index=y_train.index
)

X_test_best = pd.DataFrame(
    X_test_rfe,
    columns=rfe_features,
    index=y_test.index
)

# 加入 intercept
X_train_sm = sm.add_constant(
    X_train_best,
    has_constant="add"
)

X_test_sm = sm.add_constant(
    X_test_best,
    has_constant="add"
)

# 建立 OLS
ols_model = sm.OLS(
    y_train,
    X_train_sm
).fit()

# Training fitted values / residuals
train_fitted = ols_model.fittedvalues
train_residuals = ols_model.resid


# ==============================
# 8.1 Linearity
# Residual vs Fitted Plot
# ==============================

plt.figure(figsize=(8, 5))

plt.scatter(
    train_fitted,
    train_residuals,
    alpha=0.3,
    s=12
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.tight_layout()

plt.savefig(
    "figures/residuals_vs_fitted.png",
    dpi=300
)

plt.close()

print("\n已儲存：residuals_vs_fitted.png")


# ==============================
# 8.2 Independence
# Durbin-Watson Test
# ==============================

dw = durbin_watson(
    train_residuals
)

print("\n===== Independence of Errors =====")
print(f"Durbin-Watson：{dw:.4f}")

if 1.5 <= dw <= 2.5:
    print("結果：殘差沒有明顯自相關")
else:
    print("結果：殘差可能存在自相關")


# ==============================
# 8.3 Homoscedasticity
# Breusch-Pagan Test
# ==============================

bp_test = het_breuschpagan(
    train_residuals,
    X_train_sm
)

bp_labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]

print("\n===== Breusch-Pagan Test =====")

for label, value in zip(
    bp_labels,
    bp_test
):
    print(f"{label}: {value:.6f}")

bp_pvalue = bp_test[1]

if bp_pvalue > 0.05:
    print("結果：沒有顯著異質變異")
else:
    print("結果：可能存在異質變異")


# ==============================
# 8.4 Normality of Residuals
# Q-Q Plot
# ==============================

fig = sm.qqplot(
    train_residuals,
    line="45",
    fit=True
)

plt.title("Q-Q Plot of Residuals")
plt.tight_layout()

plt.savefig(
    "figures/qq_plot.png",
    dpi=300
)

plt.close()

print("\n已儲存：qq_plot.png")


# Residual Histogram
plt.figure(figsize=(8, 5))

plt.hist(
    train_residuals,
    bins=40,
    edgecolor="black"
)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")

plt.tight_layout()

plt.savefig(
    "figures/residual_histogram.png",
    dpi=300
)

plt.close()

print("已儲存：residual_histogram.png")


# ==============================
# 8.5 Multicollinearity
# Corrected VIF
# ==============================

# VIF 計算時加入 constant
X_vif = sm.add_constant(
    X_train_best,
    has_constant="add"
)

vif_values = []

# 從 1 開始，跳過 constant
for i in range(1, X_vif.shape[1]):

    vif_values.append(
        variance_inflation_factor(
            X_vif.values,
            i
        )
    )

vif_df = pd.DataFrame({
    "Feature": X_train_best.columns,
    "VIF": vif_values
})

vif_df = vif_df.sort_values(
    "VIF",
    ascending=False
)

print("\n===== Corrected Variance Inflation Factor =====")
print(vif_df.to_string(index=False))

vif_df.to_csv(
    "vif_results.csv",
    index=False
)

# ==============================
# 9. Prediction Interval
# ==============================

prediction = ols_model.get_prediction(
    X_test_sm
)

prediction_summary = prediction.summary_frame(
    alpha=0.05
)

# 95% prediction interval
pred_mean = prediction_summary["mean"]

pred_lower = prediction_summary["obs_ci_lower"]

pred_upper = prediction_summary["obs_ci_upper"]


# 儲存結果
prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred_mean.values,
    "Lower_95_PI": pred_lower.values,
    "Upper_95_PI": pred_upper.values
})

prediction_results.to_csv(
    "prediction_interval_results.csv",
    index=False
)


# ==============================
# 9.1 Prediction Interval Plot
# ==============================

# 為了圖不要太亂，只畫前 100 筆
plot_n = 100

plot_df = prediction_results.iloc[:plot_n]

x_axis = np.arange(plot_n)

plt.figure(figsize=(12, 6))

plt.plot(
    x_axis,
    plot_df["Actual"],
    label="Actual",
    marker="o",
    markersize=3
)

plt.plot(
    x_axis,
    plot_df["Predicted"],
    label="Predicted"
)

plt.fill_between(
    x_axis,
    plot_df["Lower_95_PI"],
    plot_df["Upper_95_PI"],
    alpha=0.25,
    label="95% Prediction Interval"
)

plt.xlabel("Test Sample")
plt.ylabel("Life Expectancy")
plt.title(
    "Life Expectancy Prediction with 95% Prediction Interval"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/prediction_interval.png",
    dpi=300
)

plt.close()

print("\n已儲存：prediction_interval.png")


# ==============================
# 9.2 Prediction Interval Coverage
# ==============================

inside_interval = (
    (prediction_results["Actual"] >=
     prediction_results["Lower_95_PI"])
    &
    (prediction_results["Actual"] <=
     prediction_results["Upper_95_PI"])
)

coverage = inside_interval.mean() * 100

print("\n===== 95% Prediction Interval =====")
print(
    f"Coverage：{coverage:.2f}%"
)

print("\n===== OLS Summary =====")
print(ols_model.summary())