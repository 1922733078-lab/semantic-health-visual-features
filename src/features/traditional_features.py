"""
传统视觉设计特征提取 (33 维)
色彩(10) + 纹理(7) + 构图(8) + 版式(4) + 显著性(4)

注：版式特征在合成数据上不使用 EasyOCR（速度），
    改为基于图像分析的文字区域检测近似。

使用方法: python src/features/traditional_features.py
"""

import numpy as np
import pandas as pd
import cv2
import zlib
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from scipy.ndimage import maximum_filter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 色彩特征 (10 维)
# ============================================================
def extract_color_features(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    features = {}

    # 1.1 主色数量
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # OpenCV's random-centre k-means uses a process-global RNG.  Seed it from
    # the image bytes so feature extraction is invariant to worker scheduling.
    cv2.setRNGSeed(int(zlib.crc32(img_bgr.tobytes()) & 0x7FFFFFFF))
    _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    cluster_sizes = np.bincount(labels.flatten(), minlength=5)
    features["num_dominant_colors"] = int(np.sum(cluster_sizes / len(labels) > 0.05))

    # 1.2 色彩熵
    h_bins, s_bins, v_bins = 12, 8, 8
    h_edges = np.linspace(0, 180, h_bins + 1)
    s_edges = np.linspace(0, 256, s_bins + 1)
    v_edges = np.linspace(0, 256, v_bins + 1)
    hist_3d, _ = np.histogramdd(img_hsv.reshape(-1, 3), bins=(h_edges, s_edges, v_edges))
    hist_3d = hist_3d / (hist_3d.sum() + 1e-10)
    features["color_entropy"] = float(entropy(hist_3d.flatten() + 1e-10))

    # 1.3 色相均值
    features["hue_mean"] = float(img_hsv[:, :, 0].mean())

    # 1.4 饱和度均值与标准差
    features["saturation_mean"] = float(img_hsv[:, :, 1].mean())
    features["saturation_std"] = float(img_hsv[:, :, 1].std())

    # 1.5 明度均值与标准差
    features["value_mean"] = float(img_hsv[:, :, 2].mean())
    features["value_std"] = float(img_hsv[:, :, 2].std())

    # 1.5 明度对比度
    L = img_lab[:, :, 0].astype(np.float64)
    features["lightness_contrast"] = float(np.sqrt(np.mean((L - L.mean()) ** 2)))

    # 1.6 色相对比
    h_channel = img_hsv[:, :, 0].flatten()
    h_hist = np.histogram(h_channel, bins=36, range=(0, 180))[0]
    top_hue_bins = np.argsort(h_hist)[-3:]
    hue_contrasts = []
    for i in range(len(top_hue_bins)):
        for j in range(i + 1, len(top_hue_bins)):
            diff = abs(int(top_hue_bins[i]) - int(top_hue_bins[j])) * 5
            diff = min(diff, 180 - diff) / 90.0
            hue_contrasts.append(diff)
    features["hue_contrast"] = float(np.mean(hue_contrasts)) if hue_contrasts else 0.0

    # 1.7 暖色比例
    h = img_hsv[:, :, 0]
    warm_mask = (h <= 30) | (h >= 150)
    features["warm_color_ratio"] = float(warm_mask.mean())

    # 1.8 色彩和谐度
    features["color_harmony"] = float(_estimate_color_harmony(img_hsv))

    return features


def _estimate_color_harmony(img_hsv):
    h_channel = img_hsv[:, :, 0].flatten()
    h_hist = np.histogram(h_channel, bins=36, range=(0, 180))[0]
    top2 = np.argsort(h_hist)[-2:]
    hue_diff = abs(int(top2[0]) - int(top2[1])) * 5
    hue_diff = min(hue_diff, 180 - hue_diff)
    harmonic_angles = [0, 30, 60, 120, 180]
    best_match = min(harmonic_angles, key=lambda x: abs(hue_diff - x))
    return 1.0 - abs(hue_diff - best_match) / 90.0


# ============================================================
# 2. 纹理与边缘特征 (7 维)
# ============================================================
def extract_texture_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features = {}

    # 2.1 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    features["edge_density"] = float(edges.mean() / 255.0)

    # 2.2 边缘方向熵
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(sobel_y, sobel_x)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mag_threshold = np.percentile(magnitude, 70)
    strong_edges = magnitude > mag_threshold
    if strong_edges.sum() > 0:
        orient_hist = np.histogram(orientation[strong_edges], bins=8, range=(-np.pi, np.pi))[0]
        orient_hist = orient_hist / (orient_hist.sum() + 1e-10)
        features["edge_orientation_entropy"] = float(entropy(orient_hist + 1e-10))
    else:
        features["edge_orientation_entropy"] = 0.0

    # 2.3-2.6 GLCM 特征
    gray_q = (gray // 4).astype(np.uint8)  # 量化到 64 级加速
    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    features["glcm_contrast"] = float(graycoprops(glcm, "contrast").mean())
    features["glcm_energy"] = float(graycoprops(glcm, "energy").mean())
    features["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity").mean())
    features["glcm_correlation"] = float(graycoprops(glcm, "correlation").mean())

    # 2.7 灰度统计与梯度能量
    features["gray_mean"] = float(gray.mean())
    features["gray_std"] = float(gray.std())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features["gradient_energy"] = float(np.sqrt(gx ** 2 + gy ** 2).mean())

    # 2.8 灰度熵
    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    gray_hist = gray_hist / (gray_hist.sum() + 1e-10)
    features["gray_entropy"] = float(entropy(gray_hist.flatten() + 1e-10))

    return features


# ============================================================
# 3. 构图特征 (8 维)
# ============================================================
def extract_composition_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    features = {}

    # 3.1 对称性
    flipped = cv2.flip(gray, 1)
    min_dim = min(h, w)
    if min_dim >= 7:
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        try:
            features["symmetry"] = float(ssim(gray, flipped,
                                               data_range=float(gray.max() - gray.min()),
                                               win_size=win_size))
        except Exception:
            features["symmetry"] = 0.5
    else:
        features["symmetry"] = 0.5

    # 3.2-3.6 构图特征（基于显著图）
    saliency = _compute_saliency(img_bgr)
    cy, cx = _centroid_of_mass(saliency)

    # 三分法则
    third_h, third_w = h / 3, w / 3
    intersections = [(third_h, third_w), (third_h, 2 * third_w),
                     (2 * third_h, third_w), (2 * third_h, 2 * third_w)]
    dists = [np.sqrt((cx - ix) ** 2 + (cy - iy) ** 2) for ix, iy in intersections]
    features["rule_of_thirds"] = float(1.0 - min(dists) / (np.sqrt(h ** 2 + w ** 2) + 1e-10))

    # 视觉重心偏移
    features["center_offset_x"] = float((cx - w / 2) / (w / 2))
    features["center_offset_y"] = float((cy - h / 2) / (h / 2))

    # 留白比例
    edge_block = cv2.Canny(gray, 30, 100).astype(np.float32)
    kernel = np.ones((32, 32), np.float32) / (32 * 32)
    edge_density_map = cv2.filter2D(edge_block, -1, kernel)
    features["whitespace_ratio"] = float((edge_density_map < 0.02).mean())

    # 前景/背景面积比
    saliency_binary = (saliency > saliency.mean()).astype(np.float32)
    features["fg_bg_ratio"] = float(saliency_binary.mean())

    # 对角线能量比
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    hh, ww = gray.shape[:2]
    diag_mask = np.eye(hh, ww, dtype=bool)
    anti_mask = np.fliplr(np.eye(hh, ww, dtype=bool))
    diag_mean = mag[diag_mask].mean()
    anti_mean = mag[anti_mask].mean()
    if diag_mean + anti_mean > 0:
        features["diagonal_energy_ratio"] = float(diag_mean / (anti_mean + 1e-10))
    else:
        features["diagonal_energy_ratio"] = 1.0

    return features


def _compute_saliency(img_bgr):
    """Return a deterministic spectral-residual saliency map.

    This in-repository implementation follows the log-spectrum residual idea
    of Hou and Zhang (CVPR 2007; DOI 10.1109/CVPR.2007.383267).  It deliberately
    depends only on NumPy and OpenCV core.  Earlier code attempted to call the
    optional ``cv2.saliency`` contrib module and silently returned a constant
    0.5 map when that module was absent.  A non-uniform input that degenerates
    to a constant map now raises instead of producing plausible-looking data.
    """
    if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("saliency requires a non-empty three-channel BGR image")
    if not np.isfinite(img_bgr).all():
        raise ValueError("saliency input contains non-finite values")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape
    reduced = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    input_range = float(np.ptp(reduced))
    if input_range <= 1e-12:
        # A truly uniform image has no spatially distinctive location.
        return np.zeros((h, w), dtype=np.float32)

    spectrum = np.fft.fft2(reduced)
    log_amplitude = np.log(np.abs(spectrum) + 1e-12)
    phase = np.angle(spectrum)
    local_average = cv2.boxFilter(
        log_amplitude, -1, (3, 3), normalize=True, borderType=cv2.BORDER_REFLECT
    )
    residual = log_amplitude - local_average
    reconstructed = np.fft.ifft2(np.exp(residual + 1j * phase))
    saliency_small = np.abs(reconstructed) ** 2
    saliency_small = cv2.GaussianBlur(
        saliency_small, (9, 9), 2.5, borderType=cv2.BORDER_REFLECT
    )

    minimum = float(saliency_small.min())
    dynamic_range = float(saliency_small.max() - minimum)
    if not np.isfinite(saliency_small).all() or dynamic_range <= 1e-12:
        raise RuntimeError("spectral-residual saliency degenerated on a non-uniform image")
    saliency_small = (saliency_small - minimum) / dynamic_range
    saliency_map = cv2.resize(
        saliency_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    saliency_map = np.clip(saliency_map, 0.0, 1.0).astype(np.float32)
    if not np.isfinite(saliency_map).all() or float(saliency_map.std()) <= 1e-8:
        raise RuntimeError("saliency semantic-health check failed after resizing")
    return saliency_map


def _centroid_of_mass(saliency_map):
    total = saliency_map.sum()
    if total == 0:
        h, w = saliency_map.shape
        return h / 2, w / 2
    rows = np.arange(saliency_map.shape[0])
    cols = np.arange(saliency_map.shape[1])
    cy = np.sum(saliency_map * rows[:, np.newaxis]) / total
    cx = np.sum(saliency_map * cols[np.newaxis, :]) / total
    return cy, cx


# ============================================================
# 4. 版式特征 (4 维，快速版)
# ============================================================
def extract_typography_features(img_bgr):
    """Return legacy-named projection fields (none is a text detector).

    The storage names are retained so historical tables remain readable.  Their
    exact operational definitions are documented in docs/FEATURE_DATA_DICTIONARY.md.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    features = {}

    # 4.1 Exact duplicate of Canny edge_density; excluded from active formulas.
    edges = cv2.Canny(gray, 50, 150)
    features["text_coverage"] = float(edges.mean() / 255.0)

    # 4.2 Positive transitions in a thresholded horizontal mean projection.
    h_proj = gray.mean(axis=1)
    features["text_block_count"] = int(np.sum(np.diff(h_proj > h_proj.mean()) > 0))

    # 4.3 Coefficient of variation of 64x64 local standard deviations.
    block_size = 64
    h, w = gray.shape
    contrasts = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            contrasts.append(float(block.std()))
    if contrasts and np.mean(contrasts) > 0:
        features["font_size_cv"] = float(np.std(contrasts) / np.mean(contrasts))
    else:
        features["font_size_cv"] = 0.0

    # 4.4 Otsu dark-pixel-range flag; this does not establish text presence.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_ratio = (binary == 0).mean()
    features["has_text"] = 1 if 0.02 < dark_ratio < 0.5 else 0

    return features


# ============================================================
# 5. 显著性特征 (4 维)
# ============================================================
def extract_saliency_features(img_bgr):
    saliency_map = _compute_saliency(img_bgr)
    features = {}

    # 5.1 显著图均值与标准差
    features["saliency_mean"] = float(saliency_map.mean())
    features["saliency_std"] = float(saliency_map.std())

    # 5.2 显著区域数量
    saliency_binary = (saliency_map > saliency_map.mean()).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(saliency_binary)
    features["num_salient_regions"] = int(num_labels - 1)

    # 5.3 显著图熵
    sal_flat = saliency_map.flatten()
    sal_flat = sal_flat / (sal_flat.sum() + 1e-10)
    features["saliency_entropy"] = float(entropy(sal_flat + 1e-10))

    # 5.4 显著图峰值数量
    try:
        local_max = (saliency_map == maximum_filter(saliency_map, size=15))
        features["num_saliency_peaks"] = int(local_max.sum())
    except Exception:
        features["num_saliency_peaks"] = 0

    # 5.5 显著图覆盖率
    features["saliency_coverage"] = float(
        (saliency_map > saliency_map.mean() + saliency_map.std()).mean()
    )

    return features


# ============================================================
# 6. 主函数：批量提取（带多进程加速）
# ============================================================
def process_single_image(args):
    """处理单张图像（用于多进程）"""
    img_path, image_id, category = args
    if not Path(img_path).exists():
        raise FileNotFoundError(f"image does not exist: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {img_path}")
    img = cv2.resize(img, (512, 512))

    feats = {"image_id": image_id, "category": category}
    feats.update(extract_color_features(img))
    feats.update(extract_texture_features(img))
    feats.update(extract_composition_features(img))
    feats.update(extract_typography_features(img))
    feats.update(extract_saliency_features(img))
    if feats["edge_density"] != feats["text_coverage"]:
        raise AssertionError("legacy text_coverage must equal edge_density exactly")
    numeric_values = np.asarray(
        [value for key, value in feats.items() if key not in {"image_id", "category"}],
        dtype=float,
    )
    if not np.isfinite(numeric_values).all():
        raise RuntimeError(f"non-finite feature value for {image_id}")
    return feats


def extract_all_traditional_features(metadata_csv, output_csv, n_workers=4):
    from multiprocessing import Pool

    df = pd.read_csv(metadata_csv)
    tasks = []
    for _, row in df.iterrows():
        img_path = row.get("standardized_path", "")
        if not Path(img_path).exists():
            img_path = str(Path("data/processed") / row["category"] / f"{row['image_id']}.jpg")
        tasks.append((img_path, row["image_id"], row["category"]))

    print(f"Extracting features for {len(tasks)} images with {n_workers} workers...")

    all_features = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_image, tasks),
                           total=len(tasks), desc="Extracting"):
            if result is not None:
                all_features.append(result)

    result_df = pd.DataFrame(all_features)
    if len(result_df) != len(tasks):
        raise RuntimeError(
            f"feature extraction returned {len(result_df)} rows for {len(tasks)} tasks"
        )
    if result_df["image_id"].nunique() != len(result_df):
        raise RuntimeError("feature extraction produced duplicate image identifiers")
    result_df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Total images: {len(result_df)}")
    print(f"Feature dims: {len(result_df.columns) - 2}")
    return result_df


if __name__ == "__main__":
    METADATA_CSV = "data/processed/metadata.csv"
    OUTPUT_CSV = "data/features/traditional_features.csv"
    extract_all_traditional_features(METADATA_CSV, OUTPUT_CSV, n_workers=4)
