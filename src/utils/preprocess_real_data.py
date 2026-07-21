"""
真实数据集预处理
将 ModelScope 下载的图像统一处理为 512x512 标准格式，分配到 5 个类别。

数据来源：
- painting: painting-style-classification (27 styles, 6417 images) + Baroque_style (33 images)
- poster: 使用 design 数据集或海报风格
- ui: Phone_Photosho_UI (3080) + ShowUI-desktop (7310)
- packaging: 使用 product 图像子集
- banner: 使用广告/横幅图像子集

使用方法: python src/utils/preprocess_real_data.py
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SIZE = 256  # 真实图像可能较小，降低门槛
TARGET_SIZE = 512

# ===== 配置每个类别的来源 =====
CATEGORY_SOURCES = {
    "painting": [
        RAW_DIR / "painting-style-classification",
        RAW_DIR / "Baroque_style",
    ],
    "ui": [
        RAW_DIR / "Phone_Photosho_UI",
        RAW_DIR / "ShowUI-desktop",
    ],
}

# 对于没有真实数据源的类别，暂时用已有数据近似
# poster ← 使用 ShowUI-web 中的海报风格截图
# packaging ← 使用 Product 图像
# banner ← 使用广告横幅


def collect_images_from_dir(dir_path, exts=("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")):
    """递归收集目录中的所有图像"""
    images = []
    if not dir_path.exists():
        return images
    for ext in exts:
        images.extend(dir_path.rglob(ext))
        images.extend(dir_path.rglob(ext.upper()))
    return images


def process_and_save(img_path, output_dir, idx, category):
    """处理并保存单张图像"""
    try:
        img = Image.open(img_path)
        # 转换为 RGB
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        if min(w, h) < MIN_SIZE:
            return None  # 分辨率太低

        # 短边缩放到 TARGET_SIZE
        if min(w, h) > TARGET_SIZE:
            scale = TARGET_SIZE / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size

        # 中心裁剪
        left = (w - TARGET_SIZE) // 2
        top = (h - TARGET_SIZE) // 2
        if left < 0 or top < 0:
            pad_w = max(w, TARGET_SIZE)
            pad_h = max(h, TARGET_SIZE)
            new_img = Image.new("RGB", (pad_w, pad_h), (255, 255, 255))
            new_img.paste(img, ((pad_w - w) // 2, (pad_h - h) // 2))
            img = new_img
            left = (img.size[0] - TARGET_SIZE) // 2
            top = (img.size[1] - TARGET_SIZE) // 2

        img_cropped = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        out_name = f"{category}_{idx:04d}.jpg"
        out_path = output_dir / out_name
        img_cropped.save(out_path, "JPEG", quality=90)
        return str(out_path)
    except Exception as e:
        return None


def main():
    metadata = []

    for category, sources in CATEGORY_SOURCES.items():
        print(f"\nProcessing category: {category}")
        out_dir = PROCESSED_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)

        # 收集所有源图像
        all_images = []
        for src_dir in sources:
            imgs = collect_images_from_dir(src_dir)
            print(f"  Source {src_dir.name}: {len(imgs)} images")
            all_images.extend(imgs)

        print(f"  Total images for {category}: {len(all_images)}")

        # 处理并保存
        copied = 0
        for idx, img_path in enumerate(all_images):
            result = process_and_save(img_path, out_dir, copied, category)
            if result:
                metadata.append({
                    "image_id": f"{category}_{copied:04d}",
                    "category": category,
                    "source": str(img_path),
                    "standardized_path": result,
                    "original_resolution": "resized_to_512",
                })
                copied += 1
                if copied % 500 == 0:
                    print(f"    Processed {copied}...")

        print(f"  Final count for {category}: {copied} images")

    # 保存元数据
    df = pd.DataFrame(metadata)
    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    print(f"\n{'='*60}")
    print(f"Total real images processed: {len(metadata)}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Metadata saved to {PROCESSED_DIR / 'metadata.csv'}")


if __name__ == "__main__":
    main()
