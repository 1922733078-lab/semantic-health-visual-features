"""
合成数据集生成器
为 5 个图像类别生成具有不同视觉特性的合成图像（512x512），
使得特征提取、建模和统计检验均有可分析的真实差异。

类别: painting, poster, ui, packaging, banner
每类 300 张，共 1500 张。

使用方法: python src/utils/generate_dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import json
import math

random.seed(42)
np.random.seed(42)

# ===== 配置 =====
BASE_DIR = Path("data/processed")
RAW_DIR = Path("data/raw/synthetic")
RAW_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "painting": {"n": 300, "complexity_range": (2.0, 6.5), "beauty_range": (2.5, 6.5),
                 "order_range": (2.0, 6.0), "colorfulness": (0.5, 1.0),
                 "texture_density": (0.3, 0.9), "symmetry": (0.1, 0.7)},
    "poster":   {"n": 300, "complexity_range": (3.0, 6.0), "beauty_range": (3.5, 6.5),
                 "order_range": (4.0, 6.5), "colorfulness": (0.6, 1.0),
                 "texture_density": (0.2, 0.6), "symmetry": (0.4, 0.8)},
    "ui":       {"n": 300, "complexity_range": (2.0, 5.0), "beauty_range": (3.0, 6.0),
                 "order_range": (5.0, 7.0), "colorfulness": (0.2, 0.6),
                 "texture_density": (0.1, 0.4), "symmetry": (0.6, 0.9)},
    "packaging":{"n": 300, "complexity_range": (2.5, 5.5), "beauty_range": (3.0, 6.0),
                 "order_range": (3.5, 6.0), "colorfulness": (0.4, 0.9),
                 "texture_density": (0.2, 0.7), "symmetry": (0.3, 0.7)},
    "banner":   {"n": 300, "complexity_range": (2.5, 5.5), "beauty_range": (3.5, 6.5),
                 "order_range": (4.0, 6.5), "colorfulness": (0.5, 0.95),
                 "texture_density": (0.15, 0.5), "symmetry": (0.5, 0.85)},
}


def lerp(a, b, t):
    return a + (b - a) * t


def hsv_to_rgb(h, s, v):
    """h in [0,360), s in [0,1], v in [0,1] -> (r,g,b) in [0,255]"""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def generate_painting(params, idx):
    """生成绘画风格图像"""
    size = 512
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    complexity = np.random.uniform(*params["complexity_range"])
    colorfulness = np.random.uniform(*params["colorfulness"])
    texture_d = np.random.uniform(*params["texture_density"])
    symmetry = np.random.uniform(*params["symmetry"])

    # 基础背景色（暖色调）
    bg_hue = np.random.uniform(0, 360)
    bg_sat = np.random.uniform(0.1, 0.3) * colorfulness
    bg_val = np.random.uniform(0.7, 0.95)
    bg_color = hsv_to_rgb(bg_hue, bg_sat, bg_val)
    draw.rectangle([0, 0, size, size], fill=bg_color)

    # 笔触/色块
    n_strokes = int(complexity * 30 + 20)
    base_hue = np.random.uniform(0, 360)

    for _ in range(n_strokes):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        stroke_w = int(np.random.uniform(10, 80) * texture_d)
        stroke_h = int(np.random.uniform(5, 40))
        angle = np.random.uniform(0, 180)
        hue_offset = np.random.uniform(-40, 40)
        h = base_hue + hue_offset
        s = np.random.uniform(0.4, 0.9) * colorfulness
        v = np.random.uniform(0.5, 0.95)
        alpha = int(np.random.uniform(80, 200))
        color = hsv_to_rgb(h, s, v)

        # 绘制椭圆笔触
        overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.ellipse(
            [x - stroke_w // 2, y - stroke_h // 2,
             x + stroke_w // 2, y + stroke_h // 2],
            fill=color + (alpha,)
        )
        overlay = overlay.rotate(angle, expand=False)
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)

    # 添加一些随机线条（模拟绘画笔触感）
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    n_lines = int(complexity * 15)
    for _ in range(n_lines):
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = x1 + np.random.randint(-100, 100), y1 + np.random.randint(-100, 100)
        color = hsv_to_rgb(base_hue + np.random.uniform(-30, 30),
                          0.5 * colorfulness, 0.5)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=int(1 + 4 * texture_d))

    return img


def generate_poster(params, idx):
    """生成海报设计图像"""
    size = 512
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    complexity = np.random.uniform(*params["complexity_range"])
    colorfulness = np.random.uniform(*params["colorfulness"])
    symmetry = np.random.uniform(*params["symmetry"])

    # 大色块背景
    bg_hue = np.random.uniform(0, 360)
    bg_sat = np.random.uniform(0.3, 0.8) * colorfulness
    bg_color = hsv_to_rgb(bg_hue, bg_sat, np.random.uniform(0.4, 0.8))
    draw.rectangle([0, 0, size, size], fill=bg_color)

    # 几何形状（海报特征）
    n_shapes = int(complexity * 5 + 3)
    contrast_hue = (bg_hue + 180 + np.random.uniform(-30, 30)) % 360

    for _ in range(n_shapes):
        x = np.random.randint(50, size - 50)
        y = np.random.randint(50, size - 50)
        shape_size = int(np.random.uniform(40, 150))
        shape_type = np.random.choice(["rect", "circle"])

        h = contrast_hue + np.random.uniform(-20, 20)
        s = np.random.uniform(0.5, 1.0) * colorfulness
        v = np.random.uniform(0.5, 0.95)
        color = hsv_to_rgb(h, s, v)

        if shape_type == "rect":
            draw.rectangle([x, y, x + shape_size, y + shape_size], fill=color)
        else:
            draw.ellipse([x, y, x + shape_size, y + shape_size], fill=color)

    # 模拟文字区域（水平条带）
    n_text_blocks = max(1, int(np.random.uniform(1, 4)))
    for _ in range(n_text_blocks):
        y_pos = np.random.randint(30, size - 60)
        h_text = int(np.random.uniform(15, 40))
        draw.rectangle([30, y_pos, size - 30, y_pos + h_text],
                       fill=(255, 255, 255, 200) if np.random.random() > 0.5 else (30, 30, 30))

    return img


def generate_ui(params, idx):
    """生成 UI 界面截图风格图像"""
    size = 512
    img = Image.new("RGB", (size, size), (240, 240, 245))
    draw = ImageDraw.Draw(img)

    complexity = np.random.uniform(*params["complexity_range"])
    colorfulness = np.random.uniform(*params["colorfulness"])

    # 顶部导航栏
    nav_color = hsv_to_rgb(np.random.uniform(200, 240), 0.3 * colorfulness, 0.8)
    draw.rectangle([0, 0, size, 40], fill=nav_color)

    # 卡片布局
    n_cards = int(complexity * 2 + 2)
    card_w = size - 40
    card_h = int(np.random.uniform(60, 100))
    margin = 15
    y_start = 55

    for i in range(n_cards):
        if y_start + card_h > size - 20:
            break
        y = y_start + i * (card_h + margin)
        # 卡片背景（白色）
        draw.rectangle([20, y, 20 + card_w, y + card_h],
                       fill=(255, 255, 255),
                       outline=(200, 200, 200))
        # 模拟图片区域
        img_area = int(np.random.uniform(40, 70))
        accent_hue = np.random.uniform(0, 360)
        accent_color = hsv_to_rgb(accent_hue, 0.5 * colorfulness, 0.8)
        draw.rectangle([30, y + 10, 30 + img_area, y + card_h - 10],
                       fill=accent_color)
        # 模拟文字行
        for j in range(3):
            line_w = int(card_w - img_area - 60 - np.random.uniform(0, 100))
            draw.rectangle([img_area + 45, y + 15 + j * 20,
                           img_area + 45 + max(line_w, 30), y + 28 + j * 20],
                           fill=(180, 180, 180))

    # 底部 tab bar
    n_tabs = 4
    tab_w = size // n_tabs
    tab_colors = [hsv_to_rgb(210, 0.6 * colorfulness, 0.9)] * n_tabs
    tab_colors[np.random.randint(0, 4)] = (60, 60, 60)  # active tab
    for i in range(n_tabs):
        draw.rectangle([i * tab_w, size - 45, (i + 1) * tab_w, size],
                       fill=(248, 248, 248), outline=(200, 200, 200))
        # tab icon placeholder
        draw.ellipse([i * tab_w + tab_w // 2 - 8, size - 38,
                      i * tab_w + tab_w // 2 + 8, size - 25],
                     fill=tab_colors[i])

    return img


def generate_packaging(params, idx):
    """生成包装设计图像"""
    size = 512
    img = Image.new("RGB", (size, size), (250, 250, 248))
    draw = ImageDraw.Draw(img)

    complexity = np.random.uniform(*params["complexity_range"])
    colorfulness = np.random.uniform(*params["colorfulness"])

    # 产品主体区域（中央偏上）
    center_x, center_y = size // 2, size // 2 - 30
    product_w = int(size * 0.5)
    product_h = int(size * 0.45)

    # 产品背景色
    bg_hue = np.random.uniform(0, 360)
    bg_color = hsv_to_rgb(bg_hue, 0.2 * colorfulness, 0.85)
    draw.rectangle([center_x - product_w // 2, center_y - product_h // 2,
                    center_x + product_w // 2, center_y + product_h // 2],
                   fill=bg_color, outline=(180, 180, 180))

    # 装饰元素
    n_elements = int(complexity * 4)
    accent_hue = (bg_hue + np.random.uniform(120, 240)) % 360
    for _ in range(n_elements):
        ex = center_x + np.random.randint(-product_w // 2 + 20, product_w // 2 - 20)
        ey = center_y + np.random.randint(-product_h // 2 + 20, product_h // 2 - 20)
        e_size = int(np.random.uniform(10, 35))
        color = hsv_to_rgb(accent_hue + np.random.uniform(-15, 15),
                          0.7 * colorfulness, 0.8)
        shape = np.random.choice(["circle", "rect", "diamond"])
        if shape == "circle":
            draw.ellipse([ex - e_size, ey - e_size, ex + e_size, ey + e_size], fill=color)
        elif shape == "rect":
            draw.rectangle([ex - e_size, ey - e_size, ex + e_size, ey + e_size], fill=color)

    # 底部文字区域
    draw.rectangle([30, size - 80, size - 30, size - 50], fill=(60, 60, 60))
    if np.random.random() > 0.3:
        draw.rectangle([60, size - 45, size - 60, size - 30], fill=(180, 180, 180))

    return img


def generate_banner(params, idx):
    """生成广告 Banner 图像"""
    size = 512
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    complexity = np.random.uniform(*params["complexity_range"])
    colorfulness = np.random.uniform(*params["colorfulness"])

    # 渐变背景（模拟 ad banner 常见设计）
    base_hue = np.random.uniform(0, 360)
    for y_step in range(size):
        t = y_step / size
        h = base_hue + t * np.random.uniform(-30, 30)
        s = np.random.uniform(0.5, 0.9) * colorfulness
        v = 0.9 - t * 0.3
        color = hsv_to_rgb(h, s, v)
        draw.line([(0, y_step), (size, y_step)], fill=color)

    # 中央主体（产品/人物剪影区域）
    center_x = size // 2 + np.random.randint(-30, 30)
    element_w = int(size * 0.35)
    element_h = int(size * 0.4)
    element_color = hsv_to_rgb((base_hue + 180) % 360, 0.6 * colorfulness, 0.9)
    draw.ellipse([center_x - element_w // 2, size // 2 - element_h // 2 - 20,
                  center_x + element_w // 2, size // 2 + element_h // 2 - 20],
                 fill=element_color)

    # CTA 按钮
    btn_w, btn_h = 120, 40
    btn_x = np.random.randint(30, size - btn_w - 30)
    btn_y = size - 100
    cta_color = hsv_to_rgb((base_hue + np.random.uniform(120, 240)) % 360,
                           0.8 * colorfulness, 0.9)
    draw.rounded_rectangle([btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
                           radius=8, fill=cta_color)

    # 顶部/底部文字条
    if np.random.random() > 0.3:
        draw.rectangle([20, 20, size - 20, 50], fill=(255, 255, 255))
    if np.random.random() > 0.3:
        draw.rectangle([20, size - 55, size - 20, size - 25], fill=(40, 40, 40))

    return img


def main():
    generators = {
        "painting": generate_painting,
        "poster": generate_poster,
        "ui": generate_ui,
        "packaging": generate_packaging,
        "banner": generate_banner,
    }

    metadata = []
    total_images = 0

    for cat, params in CATEGORIES.items():
        print(f"\nGenerating {params['n']} images for category: {cat}")
        out_dir = BASE_DIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        gen_fn = generators[cat]

        for idx in range(params["n"]):
            img = gen_fn(params, idx)
            img_name = f"{cat}_{idx:04d}.jpg"
            img_path = out_dir / img_name
            img.save(img_path, "JPEG", quality=92)

            metadata.append({
                "image_id": f"{cat}_{idx:04d}",
                "category": cat,
                "source": "synthetic_generation",
                "standardized_path": str(img_path),
                "original_resolution": "512x512",
            })

            total_images += 1
            if (idx + 1) % 60 == 0:
                print(f"  {idx + 1}/{params['n']}")

    # 保存元数据
    df = pd.DataFrame(metadata)
    df.to_csv(BASE_DIR / "metadata.csv", index=False)
    print(f"\nTotal images generated: {total_images}")
    print(f"Metadata saved to data/processed/metadata.csv")
    print(f"Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
