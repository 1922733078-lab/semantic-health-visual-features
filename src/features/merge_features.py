"""
特征融合与预处理
将传统特征和深度特征合并，标准化，PCA 降维，构建 6 种特征方案。

使用方法: python src/features/merge_features.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===== 路径配置 =====
FEATURE_DIR = Path("data/features")
MERGED_DIR = Path("data/features/merged")
MERGED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Feature Merging and Preprocessing")
print("=" * 60)

# ===== 步骤 1：加载传统特征 =====
trad_df = pd.read_csv(FEATURE_DIR / "traditional_features.csv")
meta_cols = ["image_id", "category"]
trad_ids = trad_df[meta_cols].copy()
trad_feat_cols = [c for c in trad_df.columns if c not in meta_cols]
trad_feats = trad_df[trad_feat_cols].values.astype(np.float32)
print(f"Traditional features: {trad_feats.shape} ({len(trad_feat_cols)} dims)")

# ===== 步骤 2：加载深度特征 =====
deep_dir = FEATURE_DIR / "deep"
deep_ids = pd.read_csv(deep_dir / "image_ids.csv")

deep_feats_list = []
model_names = ["clip", "dinov2", "resnet50", "efficientnet_b0"]
for name in model_names:
    feats = np.load(deep_dir / f"{name}_features.npy")
    print(f"  {name}: {feats.shape}")
    deep_feats_list.append(feats)

deep_feats = np.hstack(deep_feats_list)
print(f"Deep features combined: {deep_feats.shape}")

# ===== 步骤 3：对齐样本 =====
trad_id_set = set(trad_ids["image_id"].values)
deep_id_set = set(deep_ids["image_id"].values)
common_ids = sorted(trad_id_set & deep_id_set)
print(f"\nImages with both traditional and deep features: {len(common_ids)}")

trad_mask = trad_ids["image_id"].isin(common_ids).values
deep_mask = deep_ids["image_id"].isin(common_ids).values

# 按 common_ids 排序
trad_ordered = trad_ids[trad_mask].set_index("image_id").loc[common_ids].reset_index()
deep_ordered_ids = deep_ids[deep_mask].set_index("image_id").loc[common_ids].reset_index()

trad_feats_aligned = trad_df[trad_mask].set_index("image_id").loc[common_ids][trad_feat_cols].values.astype(np.float32)
deep_feats_aligned = deep_feats[deep_mask]
# 按 common_ids 重排 deep
deep_idx_map = {id_: i for i, id_ in enumerate(deep_ids[deep_mask]["image_id"].values)}
deep_reordered = np.array([deep_feats_aligned[deep_idx_map[id_]] for id_ in common_ids], dtype=np.float32)

aligned_ids = trad_ordered["image_id"].values
aligned_categories = trad_ordered["category"].values
print(f"Aligned traditional: {trad_feats_aligned.shape}")
print(f"Aligned deep: {deep_reordered.shape}")

# ===== 步骤 4：标准化 =====
scaler_trad = StandardScaler()
scaler_deep = StandardScaler()

trad_feats_scaled = scaler_trad.fit_transform(trad_feats_aligned)
deep_feats_scaled = scaler_deep.fit_transform(deep_reordered)

joblib.dump(scaler_trad, FEATURE_DIR / "scaler_trad.pkl")
joblib.dump(scaler_deep, FEATURE_DIR / "scaler_deep.pkl")

# ===== 步骤 5：PCA 降维 =====
pca_128 = PCA(n_components=128, random_state=42)
deep_feats_pca128 = pca_128.fit_transform(deep_feats_scaled)
print(f"\nDeep features PCA 128: {deep_feats_pca128.shape}")
print(f"  Explained variance: {pca_128.explained_variance_ratio_.sum():.3f}")
joblib.dump(pca_128, FEATURE_DIR / "pca_deep_128.pkl")

pca_64 = PCA(n_components=64, random_state=42)
deep_feats_pca64 = pca_64.fit_transform(deep_feats_scaled)
print(f"Deep features PCA 64: {deep_feats_pca64.shape}")
joblib.dump(pca_64, FEATURE_DIR / "pca_deep_64.pkl")

pca_full = PCA(n_components=0.95, random_state=42)
deep_feats_pca_full = pca_full.fit_transform(deep_feats_scaled)
print(f"Deep features PCA 95%: {deep_feats_pca_full.shape}")

# ===== 步骤 6：构建 6 种特征方案 =====
clip_dim = 768
dinov2_dim = 384

feature_sets = {}
feature_sets["F1_traditional"] = trad_feats_scaled
feature_sets["F2_deep_pca128"] = deep_feats_pca128
feature_sets["F3_fusion_full"] = np.hstack([trad_feats_scaled, deep_feats_pca128])
feature_sets["F4_fusion_compact"] = np.hstack([trad_feats_scaled, deep_feats_pca64])
feature_sets["F5_tradition_clip"] = np.hstack([trad_feats_scaled, deep_feats_scaled[:, :clip_dim]])
feature_sets["F6_tradition_dinov2"] = np.hstack([trad_feats_scaled, deep_feats_scaled[:, clip_dim:clip_dim + dinov2_dim]])

print(f"\nFeature sets:")
for name, feats in feature_sets.items():
    print(f"  {name}: {feats.shape}")

# ===== 步骤 7：加载评分标签 =====
ratings_df = pd.read_csv("data/ratings/aggregated_ratings.csv")
target_cols = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]

# 对齐评分
rated_mask = pd.Series(aligned_ids).isin(ratings_df["image_id"].values).values
rated_ids = aligned_ids[rated_mask]
rated_categories = aligned_categories[rated_mask]

ratings_ordered = ratings_df.set_index("image_id").loc[rated_ids][target_cols].values.astype(np.float32)
print(f"\nImages with ratings: {len(rated_ids)}")

# ===== 步骤 8：保存所有特征方案和标签 =====
for name, feats in feature_sets.items():
    feats_rated = feats[rated_mask]
    np.save(MERGED_DIR / f"{name}_features.npy", feats_rated)
    print(f"  {name}: {feats_rated.shape}")

np.save(MERGED_DIR / "targets.npy", ratings_ordered)
pd.DataFrame({"image_id": rated_ids, "category": rated_categories}).to_csv(
    MERGED_DIR / "sample_info.csv", index=False
)

# 保存特征方案说明
feature_sets_info = {
    name: {"dim": feats[rated_mask].shape[1], "desc": name}
    for name, feats in feature_sets.items()
}
pd.DataFrame(feature_sets_info).T.to_csv(MERGED_DIR / "feature_sets_info.csv")

print(f"\nAll feature sets saved to {MERGED_DIR}")
print(f"Target shape: {ratings_ordered.shape}")
print(f"Target columns: {target_cols}")
print("\nFeature merging complete!")
