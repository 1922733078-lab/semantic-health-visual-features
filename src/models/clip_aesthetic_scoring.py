"""
CLIP 零样本美学评分
使用 CLIP 计算每张图像的美学质量分数，作为预测目标。
无需 IRB，无需真人被试，结果可复现。

方法：比较图像与"高质量图像"vs"低质量图像"文本描述的余弦相似度。
参考：CLIP zero-shot classification (Radford et al., 2021)

使用方法: python src/models/clip_aesthetic_scoring.py
"""

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# 加载 CLIP
print("Loading CLIP...")
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="models/downloaded/clip"
).to(device).eval()

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="models/downloaded/clip"
)

# 美学评分的文本描述（多维度）
AESTHETIC_PROMPTS = {
    "overall": ("a high quality aesthetically pleasing image", "a low quality ugly image"),
    "complexity": ("a highly complex detailed image", "a very simple minimal image"),
    "beauty": ("a beautiful attractive image", "an ugly unattractive image"),
    "order": ("a well organized structured image", "a chaotic disorganized image"),
    "hierarchy": ("an image with clear visual hierarchy", "an image with no clear focus"),
    "emotion": ("an emotionally impactful image", "an emotionally flat boring image"),
}


@torch.no_grad()
def score_image(img_path, positive_prompt, negative_prompt):
    """计算单张图像的零样本美学分数"""
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        return np.nan

    inputs = processor(
        text=[positive_prompt, negative_prompt],
        images=img,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

    # 正类概率作为分数 (0-1)，缩放到 1-7
    score = probs[0, 0].item() * 6 + 1
    return score


def main():
    # 加载图像列表
    meta = pd.read_csv("data/processed/metadata.csv")
    # 仅使用真实数据（排除合成）
    meta = meta[meta["category"].isin(["painting", "ui", "banner", "poster", "packaging"])]
    print(f"Scoring {len(meta)} images...")

    results = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="CLIP scoring"):
        img_path = row["standardized_path"]
        if not Path(img_path).exists():
            continue

        scores = {}
        for dim, (pos, neg) in AESTHETIC_PROMPTS.items():
            scores[dim] = score_image(img_path, pos, neg)

        scores["image_id"] = row["image_id"]
        scores["category"] = row["category"]
        results.append(scores)

    df = pd.DataFrame(results)
    df.to_csv("data/ratings/clip_aesthetic_scores.csv", index=False)
    print(f"\nScored {len(df)} images")
    print(f"Saved to data/ratings/clip_aesthetic_scores.csv")

    # 打印统计
    for dim in ["overall", "complexity", "beauty", "order", "hierarchy", "emotion"]:
        vals = df[dim].dropna()
        print(f"  {dim:12s}: mean={vals.mean():.2f}, std={vals.std():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")

    return df


if __name__ == "__main__":
    main()
