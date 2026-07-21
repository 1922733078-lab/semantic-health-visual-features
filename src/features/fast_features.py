"""
快速传统特征提取（优化版）
针对 11,000+ 张图像优化，去除 GLCM/显著性等慢速计算。
特征维度：28 维（色彩10 + 纹理5 + 构图7 + 版式4 + 简化显著性2）

使用方法: python src/features/fast_features.py
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scipy.stats import entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def extract_all_features(img_path):
    """提取单张图像的全部快速特征"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.resize(img, (256, 256))  # 缩小到256加速
    except Exception:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    feats = {}

    # ===== 1. 色彩特征 (10维) =====
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(pixels, 5, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    cluster_sizes = np.bincount(labels.flatten(), minlength=5)
    feats["num_dominant_colors"] = int(np.sum(cluster_sizes / len(labels) > 0.05))

    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
    feats["hue_mean"] = float(h.mean())
    feats["saturation_mean"] = float(s.mean())
    feats["saturation_std"] = float(s.std())
    feats["value_mean"] = float(v.mean())
    feats["value_std"] = float(v.std())

    L = img_lab[:,:,0].astype(np.float64)
    feats["lightness_contrast"] = float(np.sqrt(np.mean((L - L.mean())**2)))

    h_hist = np.histogram(h.flatten(), bins=36, range=(0,180))[0].astype(float)
    h_hist = h_hist / (h_hist.sum() + 1e-10)
    feats["color_entropy"] = float(entropy(h_hist + 1e-10))

    warm_mask = (h <= 30) | (h >= 150)
    feats["warm_color_ratio"] = float(warm_mask.mean())

    # 色相对比
    top_hues = np.argsort(h_hist)[-3:]
    hue_diffs = []
    for i in range(len(top_hues)):
        for j in range(i+1, len(top_hues)):
            d = abs(int(top_hues[i]) - int(top_hues[j])) * 5
            d = min(d, 180-d) / 90.0
            hue_diffs.append(d)
    feats["hue_contrast"] = float(np.mean(hue_diffs)) if hue_diffs else 0.0

    # 色彩和谐度
    top2 = np.argsort(h_hist)[-2:]
    hd = min(abs(int(top2[0]) - int(top2[1])) * 5, 180 - abs(int(top2[0]) - int(top2[1])) * 5)
    feats["color_harmony"] = 1.0 - min(abs(hd - 0), abs(hd - 30), abs(hd - 60), abs(hd - 120), abs(hd - 180)) / 90.0

    # ===== 2. 纹理特征 (5维，快速替代GLCM) =====
    edges = cv2.Canny(gray, 50, 150)
    feats["edge_density"] = float(edges.mean() / 255.0)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    orientation = np.arctan2(sobel_y, sobel_x)
    mag_thresh = np.percentile(magnitude, 70)
    strong = magnitude > mag_thresh
    if strong.sum() > 0:
        oh = np.histogram(orientation[strong], bins=8, range=(-np.pi, np.pi))[0].astype(float)
        oh = oh / (oh.sum() + 1e-10)
        feats["edge_orientation_entropy"] = float(entropy(oh + 1e-10))
    else:
        feats["edge_orientation_entropy"] = 0.0

    # 快速灰度统计（替代GLCM）
    feats["gray_mean"] = float(gray.mean())
    feats["gray_std"] = float(gray.std())
    gray_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    gray_hist = gray_hist / (gray_hist.sum() + 1e-10)
    feats["gray_entropy"] = float(entropy(gray_hist.flatten() + 1e-10))

    # 梯度能量
    feats["gradient_energy"] = float(magnitude.mean())

    # ===== 3. 构图特征 (7维) =====
    # 对称性（快速：左右像素差）
    flipped = cv2.flip(gray, 1)
    feats["symmetry"] = float(1.0 - np.mean(np.abs(gray.astype(float) - flipped.astype(float)) / 255.0))

    # 三分法则（基于亮度重心）
    cy, cx = _centroid(gray)
    h_g, w_g = gray.shape
    third_h, third_w = h_g / 3, w_g / 3
    dists = [np.sqrt((cx - third_w)**2 + (cy - third_h)**2),
             np.sqrt((cx - 2*third_w)**2 + (cy - third_h)**2),
             np.sqrt((cx - third_w)**2 + (cy - 2*third_h)**2),
             np.sqrt((cx - 2*third_w)**2 + (cy - 2*third_h)**2)]
    feats["rule_of_thirds"] = float(1.0 - min(dists) / np.sqrt(h_g**2 + w_g**2))

    feats["center_offset_x"] = float((cx - w_g/2) / (w_g/2))
    feats["center_offset_y"] = float((cy - h_g/2) / (h_g/2))

    # 留白（低边缘密度区域比例）
    edge_small = cv2.resize(edges.astype(np.float32), (32, 32))
    feats["whitespace_ratio"] = float((edge_small < 10).mean())

    # 前景/背景比（基于亮度）
    feats["fg_bg_ratio"] = float((gray > gray.mean()).mean())

    # 对角线能量
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    hh, ww = gray.shape[:2]
    diag = np.eye(hh, ww, dtype=bool)
    anti = np.fliplr(np.eye(hh, ww, dtype=bool))
    d_mean = mag[diag].mean()
    a_mean = mag[anti].mean()
    feats["diagonal_energy_ratio"] = float(d_mean / (a_mean + 1e-10)) if (d_mean + a_mean) > 0 else 1.0

    # ===== 4. 版式特征 (4维) =====
    feats["text_coverage"] = float(edges.mean() / 255.0)
    h_proj = gray.mean(axis=1)
    feats["text_block_count"] = int(np.sum(np.diff(h_proj > h_proj.mean()) > 0))
    # 局部对比度
    blocks = []
    for y in range(0, 256-32, 32):
        for x in range(0, 256-32, 32):
            blocks.append(float(gray[y:y+32, x:x+32].std()))
    feats["font_size_cv"] = float(np.std(blocks) / np.mean(blocks)) if blocks and np.mean(blocks) > 0 else 0.0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = (binary == 0).mean()
    feats["has_text"] = 1 if 0.02 < dark_ratio < 0.5 else 0

    # ===== 5. 简化显著性 (2维) =====
    # 使用频谱残差法（极快）
    sal = _spectral_saliency(gray)
    feats["saliency_mean"] = float(sal.mean())
    feats["saliency_std"] = float(sal.std())

    return feats


def _centroid(gray):
    total = gray.sum()
    if total == 0:
        h, w = gray.shape
        return h/2, w/2
    rows = np.arange(gray.shape[0])
    cols = np.arange(gray.shape[1])
    cy = np.sum(gray * rows[:, np.newaxis]) / total
    cx = np.sum(gray * cols[np.newaxis, :]) / total
    return cy, cx


def _spectral_saliency(gray):
    """频谱残差显著性（极快替代 FineGrained）"""
    f = np.fft.fft2(gray.astype(np.float32))
    shift = np.fft.fftshift(f)
    magnitude = np.abs(shift)
    phase = np.angle(shift)
    # 频谱残差 = log频谱 - 平滑log频谱
    log_mag = np.log(magnitude + 1e-10)
    avg_filter = cv2.boxFilter(log_mag, -1, (5, 5))
    residual = log_mag - avg_filter
    sal = np.fft.ifftshift(np.exp(residual + 1j * phase))
    sal_mag = np.abs(np.fft.ifft2(sal))
    sal = (sal_mag - sal_mag.min()) / (sal_mag.max() - sal_mag.min() + 1e-10)
    return sal


def main():
    metadata = pd.read_csv("data/processed/metadata.csv")
    print(f"Processing {len(metadata)} images...")

    all_features = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Fast features"):
        img_path = row.get("standardized_path", "")
        if not Path(img_path).exists():
            img_path = str(Path("data/processed") / row["category"] / f"{row['image_id']}.jpg")
        if not Path(img_path).exists():
            continue

        feats = extract_all_features(img_path)
        if feats:
            feats["image_id"] = row["image_id"]
            feats["category"] = row["category"]
            all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv("data/features/traditional_features.csv", index=False)
    print(f"\nSaved {len(df)} images, {len(df.columns)-2} features")
    return df


if __name__ == "__main__":
    main()
