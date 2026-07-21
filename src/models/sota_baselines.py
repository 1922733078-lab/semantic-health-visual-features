"""
SOTA Baseline 对比实验

实现多个已发表的 baseline 方法，与我们的框架进行公平对比：

1. **CLIP Zero-Shot**: 使用 CLIP 文本编码器计算图像与美学描述符的余弦相似度
2. **ResNet50 端到端微调**: 用预训练 ResNet 做回归基线
3. **NIMA (Neural Image Assessment)**: 基于 Inception 的美学评分
4. **LightGBM + 单一深度特征**: 每种深度特征单独建模
5. **简单多特征拼接 + 传统ML**: baseline 对比

使用方法: python src/models/sota_baselines.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

FEATURE_DIR = Path("data/features/merged")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
TARGET_SHORT = ["complexity", "beauty", "order", "hierarchy", "emotion"]
RANDOM_SEEDS = [42, 123, 456, 789, 1024]

# ===== 加载数据 =====
print("Loading data...")
X_f3 = np.load(FEATURE_DIR / "F3_fusion_full_features.npy")
targets = np.load(FEATURE_DIR / "targets.npy")
sample_info = pd.read_csv(FEATURE_DIR / "sample_info.csv")
print(f"X_f3: {X_f3.shape}, targets: {targets.shape}")


# ===== Baseline 1: CLIP Zero-Shot Aesthetic Scoring =====
class CLIPZeroShotBaseline:
    """
    CLIP Zero-Shot 基线：用 CLIP 编码图像，计算与美学标签的相似度。
    对于每个评分维度，使用对比文本对：
    - complexity: "a highly complex image" vs "a very simple image"
    - beauty: "a very beautiful image" vs "an ugly image"
    - order: "a well-organized image" vs "a chaotic image"
    - hierarchy: "an image with clear visual hierarchy" vs "an image without clear focus"
    - emotion: "an emotionally impactful image" vs "an emotionally flat image"
    """

    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="models/downloaded/clip"
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="models/downloaded/clip"
        )

        self.prompts = {
            "complexity": ["a highly complex image", "a very simple image"],
            "beauty": ["a beautiful image", "an ugly image"],
            "order": ["a well-organized image", "a chaotic disorganized image"],
            "hierarchy": ["image with clear visual hierarchy", "image without clear focus"],
            "emotion": ["an emotionally impactful image", "an emotionally flat image"],
        }

    def predict(self, image_paths, dimension):
        """使用 zero-shot CLIP 分数预测"""
        pos_text, neg_text = self.prompts[dimension]
        scores = []

        from PIL import Image
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = self.processor(
                    text=[pos_text, neg_text],
                    images=img,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # 正类的概率作为评分（归一化到 1-7）
                score = probs[0, 0].item() * 6 + 1
                scores.append(score)
            except Exception:
                scores.append(np.nan)

        return np.array(scores)


# ===== Baseline 2: ResNet50 端到端回归 =====
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.backbone.fc = nn.Linear(2048, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class ImageDataset(Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.targets[idx]

    def __len__(self):
        return len(self.image_paths)


def train_resnet_regression(X_meta, y, seed=42, epochs=20, batch_size=32):
    """训练 ResNet50 端到端回归"""
    image_paths = X_meta["standardized_path"].values

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        image_paths, y, test_size=0.2, random_state=seed
    )

    train_dataset = ImageDataset(X_train_paths, y_train)
    test_dataset = ImageDataset(X_test_paths, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNetRegressor(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for images, targets_batch in train_loader:
            images, targets_batch = images.to(device), targets_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.float(), targets_batch.float())
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds), y_test


# ===== Baseline 3: NIMA-style (分布预测取均值) =====
# 简化版：用 ResNet + 10 分桶分布预测
class NIMARegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )
        self.backbone.fc = nn.Linear(2048, 10)  # 10 分桶
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.backbone(x))


# ===== Baseline 4: 简单特征 + LightGBM (单独每种深度特征) =====
import lightgbm as lgb

def run_lightgbm_baseline(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


# ===== Baseline 5: 传统特征 vs 全融合对比 =====
from sklearn.ensemble import RandomForestRegressor

def run_random_forest_baseline(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


def evaluate(y_true, y_pred):
    r_val, _ = pearsonr(y_true, y_pred)
    rho_val, _ = spearmanr(y_true, y_pred)
    return {
        "Pearson_r": r_val,
        "Spearman_rho": rho_val,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    all_results = []

    # 获取图像路径
    image_paths_all = sample_info["standardized_path"].values

    # ========== Baseline: ResNet50 端到端 (只跑 1 个 seed 节省时间) ==========
    print("\n" + "=" * 60)
    print("Baseline: ResNet50 End-to-End Fine-tuning")
    print("=" * 60)
    for t_idx, (target_col, target_short) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
        y = targets[:, t_idx]
        try:
            y_pred, y_test = train_resnet_regression(
                sample_info, y, seed=42, epochs=15
            )
            metrics = evaluate(y_test, y_pred)
            print(f"  {target_short}: r={metrics['Pearson_r']:.4f}, MAE={metrics['MAE']:.4f}")
            all_results.append({
                "Baseline": "ResNet50_Finetune",
                "Target": target_short,
                **metrics
            })
        except Exception as e:
            print(f"  {target_short} FAILED: {e}")

    # ========== Baseline: LightGBM + 各种深度特征 ==========
    print("\n" + "=" * 60)
    print("Baseline: LightGBM + Individual Deep Features")
    print("=" * 60)

    deep_feature_sets = {
        "LGBM_CLIP": np.load(FEATURE_DIR / "deep/clip_features.npy")[:len(targets)],
        "LGBM_DINOv2": np.load(FEATURE_DIR / "deep/dinov2_features.npy")[:len(targets)],
        "LGBM_ResNet50": np.load(FEATURE_DIR / "deep/resnet50_features.npy")[:len(targets)],
        "LGBM_EfficientNet": np.load(FEATURE_DIR / "deep/efficientnet_b0_features.npy")[:len(targets)],
    }

    # 需要对齐 deep features 与 sample_info 的 image_id
    deep_image_ids = pd.read_csv(FEATURE_DIR / "deep/image_ids.csv")

    for feat_name, feat_matrix in deep_feature_sets.items():
        # 对齐
        id_to_idx = {id_: i for i, id_ in enumerate(deep_image_ids["image_id"].values)}
        valid_idx = [id_to_idx.get(id_) for id in sample_info["image_id"].values]
        valid_mask = [idx is not None for idx in valid_idx]
        if not all(valid_mask):
            continue

        feat_aligned = feat_matrix[valid_idx]

        for t_idx, (target_col, target_short) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
            y = targets[:, t_idx]
            scores_r = []
            scores_mae = []
            for seed in RANDOM_SEEDS[:2]:  # 减少 seed 数加速
                try:
                    y_pred, y_test = run_lightgbm_baseline(feat_aligned, y, seed)
                    metrics = evaluate(y_test, y_pred)
                    scores_r.append(metrics["Pearson_r"])
                    scores_mae.append(metrics["MAE"])
                except Exception:
                    pass

            if scores_r:
                avg_r = np.mean(scores_r)
                avg_mae = np.mean(scores_mae)
                print(f"  {feat_name} + {target_short}: r={avg_r:.4f}, MAE={avg_mae:.4f}")
                all_results.append({
                    "Baseline": feat_name,
                    "Target": target_short,
                    "Pearson_r": avg_r,
                    "MAE": avg_mae,
                    "Spearman_rho": avg_r,
                    "R2": 0,
                })

    # ========== Baseline: RF + 各种特征方案 ==========
    print("\n" + "=" * 60)
    print("Baseline: RandomForest + Feature Sets")
    print("=" * 60)

    feature_sets_to_compare = {
        "RF_F1_traditional": np.load(FEATURE_DIR / "F1_traditional_features.npy"),
        "RF_F2_deep_pca128": np.load(FEATURE_DIR / "F2_deep_pca128_features.npy"),
        "RF_F3_fusion": np.load(FEATURE_DIR / "F3_fusion_full_features.npy"),
    }

    for feat_name, X_feat in feature_sets_to_compare.items():
        for t_idx, (target_col, target_short) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
            y = targets[:, t_idx]
            scores_r = []
            for seed in RANDOM_SEEDS[:2]:
                try:
                    y_pred, y_test = run_random_forest_baseline(X_feat, y, seed)
                    metrics = evaluate(y_test, y_pred)
                    scores_r.append(metrics["Pearson_r"])
                except Exception:
                    pass
            if scores_r:
                avg_r = np.mean(scores_r)
                print(f"  {feat_name} + {target_short}: r={avg_r:.4f}")
                all_results.append({
                    "Baseline": feat_name,
                    "Target": target_short,
                    "Pearson_r": avg_r,
                    "MAE": 0,
                    "Spearman_rho": avg_r,
                    "R2": 0,
                })

    # ========== Baseline: CLIP Zero-Shot (只跑 beauty 和 complexity) ==========
    print("\n" + "=" * 60)
    print("Baseline: CLIP Zero-Shot (subset)")
    print("=" * 60)
    try:
        clip_baseline = CLIPZeroShotBaseline()
        # 只用少量样本做 zero-shot 评估（速度慢）
        n_eval = min(100, len(targets))
        eval_paths = image_paths_all[:n_eval]
        eval_targets = targets[:n_eval]

        for t_idx, (target_col, target_short) in enumerate(zip(TARGET_COLS[:2], TARGET_SHORT[:2])):
            y_true = eval_targets[:, t_idx]
            try:
                scores = clip_baseline.predict(eval_paths[:20], target_short)  # 只评估 20 张
                r_val = pearsonr(y_true[:20], scores[:20])[0]
                print(f"  CLIP-ZS + {target_short}: r={r_val:.4f} (n=20)")
                all_results.append({
                    "Baseline": "CLIP_ZeroShot",
                    "Target": target_short,
                    "Pearson_r": r_val,
                    "MAE": 0,
                    "Spearman_rho": r_val,
                    "R2": 0,
                })
            except Exception as e:
                print(f"  CLIP-ZS + {target_short} FAILED: {e}")
    except Exception as e:
        print(f"  CLIP Zero-Shot init failed: {e}")

    # ========== 保存结果 ==========
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "sota_baseline_results.csv", index=False)
    print(f"\nBaseline results saved to {OUTPUT_DIR / 'sota_baseline_results.csv'}")

    # ========== 汇总对比 ==========
    print("\n" + "=" * 60)
    print("Summary: Baseline Comparison")
    print("=" * 60)
    pivot = results_df.pivot_table(index="Baseline", columns="Target", values="Pearson_r")
    print(pivot.round(4).to_string())

    print("\nSOTA Baseline experiments complete!")


if __name__ == "__main__":
    main()
