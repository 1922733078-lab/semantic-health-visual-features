"""
真实数据建模实验（快速版）
仅使用传统特征，快速验证真实数据的预测性能。

使用方法: python src/models/run_real_modeling.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# 加载数据
feat_df = pd.read_csv("data/features/traditional_features.csv")
ratings_df = pd.read_csv("data/ratings/aggregated_ratings.csv")

# 仅保留真实数据
real_cats = ["painting", "ui"]
feat_df = feat_df[feat_df["category"].isin(real_cats)]

# 合并
meta_cols = ["image_id", "category"]
feat_cols = [c for c in feat_df.columns if c not in meta_cols]
df = feat_df.merge(ratings_df, on=["image_id", "category"], how="inner")

print(f"Data: {len(df)} images")
print(f"Features: {len(feat_cols)}")
print(f"Categories: {df['category'].value_counts().to_dict()}")

X = df[feat_cols].values
X = StandardScaler().fit_transform(X)

target_cols = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
target_names = ["complexity", "beauty", "order", "hierarchy", "emotion"]

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "GBR": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    "SVR": SVR(kernel="rbf", C=1.0),
}

results = []

for t_idx, (tcol, tname) in enumerate(zip(target_cols, target_names)):
    y = df[tcol].values

    for model_name, model in models.items():
        # 5-fold cross validation
        folds = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            r, _ = pearsonr(y_test, y_pred)
            folds.append(r)

        avg_r = np.mean(folds)
        std_r = np.std(folds)
        results.append({
            "Model": model_name,
            "Target": tname,
            "Pearson_r": avg_r,
            "Std": std_r,
        })
        print(f"  {model_name:15s} + {tname:12s}: r = {avg_r:.4f} ± {std_r:.4f}")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("results/tables/real_data_modeling.csv", index=False)
print(f"\nResults saved to results/tables/real_data_modeling.csv")

# 打印汇总表
print("\n" + "="*60)
print("Summary: Pearson r (5-fold CV)")
print("="*60)
pivot = results_df.pivot(index="Model", columns="Target", values="Pearson_r")
print(pivot.round(4).to_string())
