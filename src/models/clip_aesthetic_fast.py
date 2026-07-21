"""
CLIM 零样本美学评分（批处理加速版）
对所有图像只算一次图像嵌入，文本嵌入只算一次，点积得分数。
支持分批次处理，每批次 64 张。

使用方法: python src/models/clip_aesthetic_fast.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# 加载 CLIP 视觉编码器
print("Loading CLIP vision encoder...")
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="models/downloaded/clip"
).vision_model.to(device).eval()

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="models/downloaded/clip"
)

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return transform(img), str(self.image_paths[idx])


@torch.no_grad()
def extract_image_embeddings(image_paths, batch_size=64):
    """批量提取 CLIP 图像嵌入"""
    dataset = ImageDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_embeddings = []
    all_paths = []

    for images, paths in tqdm(loader, desc="Extracting image embeddings"):
        images = images.to(device)
        outputs = clip_model(pixel_values=images)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-10)
        all_embeddings.append(emb.cpu().numpy())
        all_paths.extend(paths)

    return np.vstack(all_embeddings), all_paths


@torch.no_grad()
def get_text_embeddings(prompts):
    """计算文本嵌入"""
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = clip_model.text_projection(inputs["input_ids"] if False else None)
    # CLIP text encoding
    from transformers import CLIPTextModel
    text_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="models/downloaded/clip"
    ).text_model.to(device).eval()
    text_out = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    text_emb = text_out.last_hidden_state[:, 0, :]
    text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)
    return text_emb.cpu().numpy()


def main():
    # 加载图像
    meta = pd.read_csv("data/processed/metadata.csv")
    meta = meta[meta["category"].isin(["painting", "ui", "banner", "poster", "packaging"])]
    image_paths = meta["standardized_path"].tolist()
    print(f"Found {len(image_paths)} images")

    # 只处理前 3000 张（节省时间）
    subset_n = min(3000, len(image_paths))
    image_paths = image_paths[:subset_n]
    print(f"Processing subset: {subset_n} images")

    # 提取图像嵌入
    print("\nStep 1: Extracting image embeddings...")
    img_emb, img_paths = extract_image_embeddings(image_paths, batch_size=32)
    print(f"Image embeddings: {img_emb.shape}")

    # 计算文本嵌入（每个维度正负描述）
    print("\nStep 2: Computing text embeddings...")
    AESTHETIC_PROMPTS = {
        "overall": ("a high quality aesthetically pleasing image", "a low quality ugly image"),
        "complexity": ("a highly complex detailed image", "a very simple minimal image"),
        "beauty": ("a beautiful attractive image", "an ugly unattractive image"),
        "order": ("a well organized structured image", "a chaotic disorganized image"),
        "hierarchy": ("an image with clear visual hierarchy", "an image with no clear focus"),
        "emotion": ("an emotionally impactful image", "an emotionally flat boring image"),
    }

    text_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir="models/downloaded/clip"
    ).text_model.to(device).eval()

    scores = {}
    for dim, (pos_prompt, neg_prompt) in AESTHETIC_PROMPTS.items():
        inputs = processor(text=[pos_prompt, neg_prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_out = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        text_emb = text_out.last_hidden_state[:, 0, :].cpu().numpy()
        # 归一化
        text_emb = text_emb / (np.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-10)

        # 计算分数 = 正类概率 - 负类概率 → 缩放到 1-7
        pos_sim = img_emb @ text_emb[0]  # (N,)
        neg_sim = img_emb @ text_emb[1]  # (N,)
        # sigmoid-like mapping
        prob_pos = 1 / (1 + np.exp(-(pos_sim - neg_sim) * 5))
        scores[dim] = prob_pos * 6 + 1  # scale to 1-7

    # 保存结果
    results = {"image_id": [Path(p).stem for p in img_paths], "category": meta.iloc[:subset_n]["category"].tolist()}
    results.update(scores)
    df = pd.DataFrame(results)
    df.to_csv("data/ratings/clip_aesthetic_scores.csv", index=False)

    print(f"\nStep 3: Saved {len(df)} scores to data/ratings/clip_aesthetic_scores.csv")
    for dim in AESTHETIC_PROMPTS.keys():
        vals = df[dim]
        print(f"  {dim:12s}: mean={vals.mean():.2f}, std={vals.std():.2f}, range=[{vals.min():.2f}, {vals.max():.2f}]")


if __name__ == "__main__":
    main()
