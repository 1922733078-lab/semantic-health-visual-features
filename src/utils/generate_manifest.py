"""
生成 image_manifest.js - 供评分实验网页加载图像列表

扫描 data/processed/ 中的所有图像，生成_JS_MANIFEST 格式。
该文件被 subjective_experiment/rating_tool.html 引用。

使用方法: python src/utils/generate_manifest.py
"""

import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT = Path("subjective_experiment/image_manifest.js")


def main():
    images = []
    for cat_dir in sorted(PROCESSED_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        if cat_name.startswith('.'):
            continue
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                images.append({
                    "image_id": img_path.stem,
                    "category": cat_name,
                    "path": f"../{img_path}"  # 相对路径，从 subjective_experiment/ 指向 data/processed/
                })

    manifest_json = json.dumps({"images": images}, indent=2, ensure_ascii=False)

    js_content = f"""// 自动生成 - 请勿手动修改
// 生成时间: {__import__('datetime').datetime.now().isoformat()}
var IMAGE_MANIFEST = {manifest_json};
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(js_content, encoding='utf-8')
    print(f"Generated manifest with {len(images)} images -> {OUTPUT}")

    # 按类别统计
    from collections import Counter
    cats = Counter(img["category"] for img in images)
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
