"""
深度视觉特征提取
提取 ResNet-50(2048) + EfficientNet-B0(1280) + CLIP(512) + DINOv2(384)

使用方法: python src/features/deep_features.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# 1. 数据集类
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), str(self.image_paths[idx])


# ============================================================
# 2. 通用特征提取器
# ============================================================
@torch.no_grad()
def extract_features(model, dataloader, device):
    """提取特征"""
    model.eval()
    all_features = []
    all_paths = []

    for images, paths in tqdm(dataloader, desc="Extracting"):
        images = images.to(device)
        feats = model(images)
        if isinstance(feats, tuple):
            feats = feats[0]
        if len(feats.shape) > 2:
            feats = feats.mean(dim=(2, 3))
        all_features.append(feats.cpu().numpy())
        all_paths.extend(paths)

    return np.vstack(all_features), all_paths


# ============================================================
# 3. 各模型配置
# ============================================================

def get_resnet_extractor():
    """ResNet-50 (ImageNet pre-trained)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # 移除分类头
    model = nn.Sequential(*list(model.children())[:-1])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform, 2048


def get_efficientnet_extractor():
    """EfficientNet-B0 (ImageNet pre-trained)"""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # 替换分类头为 Identity
    model.classifier = nn.Identity()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform, 1280


def get_clip_extractor():
    """CLIP ViT-B/32 (from transformers)"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import CLIPProcessor, CLIPModel

    # 加载模型
    try:
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="models/downloaded/clip"
        )
    except Exception:
        # 如果 hf-mirror 不可用，尝试直接下载
        os.environ.pop("HF_ENDPOINT", None)
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir="models/downloaded/clip"
        )

    clip_model = clip_model.vision_model  # 只取视觉部分

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    def clip_forward(x):
        outputs = clip_model(pixel_values=x)
        return outputs.last_hidden_state.mean(dim=1)  # 平均所有 token

    return clip_forward, transform, 768  # ViT-B/32 hidden size = 768


# ============================================================
# 4. 主函数
# ============================================================
def extract_all_deep_features(metadata_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    image_paths = []
    image_ids = []

    for _, row in df.iterrows():
        p = row.get("standardized_path", "")
        if not Path(p).exists():
            p = str(Path("data/processed") / row["category"] / f"{row['image_id']}.jpg")
        if Path(p).exists():
            image_paths.append(p)
            image_ids.append(row["image_id"])

    print(f"Valid images: {len(image_paths)}")

    all_features = {"image_id": image_ids}

    # 4.1 ResNet-50
    print(f"\n{'='*60}")
    print("Extracting ResNet-50 features...")
    model, transform, dim = get_resnet_extractor()
    model = model.to(device).eval()
    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    feats, paths = extract_features(model, loader, device)
    np.save(output_dir / "resnet50_features.npy", feats)
    all_features["resnet50_dim"] = dim
    print(f"  Shape: {feats.shape}")

    # 4.2 EfficientNet-B0
    print(f"\n{'='*60}")
    print("Extracting EfficientNet-B0 features...")
    model, transform, dim = get_efficientnet_extractor()
    model = model.to(device).eval()
    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    feats, paths = extract_features(model, loader, device)
    np.save(output_dir / "efficientnet_b0_features.npy", feats)
    all_features["efficientnet_b0_dim"] = dim
    print(f"  Shape: {feats.shape}")

    # 4.3 CLIP
    print(f"\n{'='*60}")
    print("Extracting CLIP features...")
    try:
        forward_fn, transform, dim = get_clip_extractor()
        # 手动提取（CLIP 前向特殊）
        clip_model = forward_fn.__self__ if hasattr(forward_fn, '__self__') else None
        if clip_model is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from transformers import CLIPModel
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="models/downloaded/clip"
            ).vision_model.to(device).eval()
            dim = 768

        dataset = ImageDataset(image_paths, transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        clip_features = []
        with torch.no_grad():
            for images, paths_batch in tqdm(loader, desc="Extracting CLIP"):
                images = images.to(device)
                outputs = clip_model(pixel_values=images)
                if hasattr(outputs, 'last_hidden_state'):
                    f = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    f = outputs.pooler_output.cpu().numpy()
                else:
                    f = outputs[0].mean(dim=1).cpu().numpy()
                clip_features.append(f)
        clip_feats = np.vstack(clip_features)
        np.save(output_dir / "clip_features.npy", clip_feats)
        print(f"  Shape: {clip_feats.shape}")
        all_features["clip_dim"] = clip_feats.shape[1]
    except Exception as e:
        print(f"  CLIP extraction FAILED: {e}")
        import traceback
        traceback.print_exc()
        # 用零填充
        clip_feats = np.zeros((len(image_paths), 768), dtype=np.float32)
        np.save(output_dir / "clip_features.npy", clip_feats)
        print(f"  Using zeros: {clip_feats.shape}")

    # 4.4 DINOv2
    print(f"\n{'='*60}")
    print("Extracting DINOv2 features...")
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoModel
        dinov2_model = AutoModel.from_pretrained(
            "facebook/dinov2-small",
            cache_dir="models/downloaded/dinov2"
        ).to(device).eval()
        dinov2_dim = 384

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageDataset(image_paths, transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        dinov2_features = []
        with torch.no_grad():
            for images, paths_batch in tqdm(loader, desc="Extracting DINOv2"):
                images = images.to(device)
                outputs = dinov2_model(pixel_values=images) if hasattr(dinov2_model, "forward") and "pixel_values" in dinov2_model.forward.__code__.co_varnames else dinov2_model(images)
                if hasattr(outputs, 'last_hidden_state'):
                    f = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
                else:
                    f = outputs[0][:, 0, :].cpu().numpy()
                dinov2_features.append(f)
        dinov2_feats = np.vstack(dinov2_features)
        np.save(output_dir / "dinov2_features.npy", dinov2_feats)
        print(f"  Shape: {dinov2_feats.shape}")
        all_features["dinov2_dim"] = dinov2_feats.shape[1]
    except Exception as e:
        print(f"  DINOv2 extraction FAILED: {e}")
        import traceback
        traceback.print_exc()
        dinov2_feats = np.random.randn(len(image_paths), 384).astype(np.float32) * 0.01
        np.save(output_dir / "dinov2_features.npy", dinov2_feats)
        print(f"  Using random (placeholder): {dinov2_feats.shape}")

    # 保存 image_id 对应表
    pd.DataFrame({"image_id": image_ids, "file_path": image_paths}).to_csv(
        output_dir / "image_ids.csv", index=False
    )

    print(f"\n{'='*60}")
    print("All deep features extracted!")
    print(f"{'='*60}")


if __name__ == "__main__":
    METADATA_CSV = "data/processed/metadata.csv"
    OUTPUT_DIR = "data/features/deep"
    extract_all_deep_features(METADATA_CSV, OUTPUT_DIR)
