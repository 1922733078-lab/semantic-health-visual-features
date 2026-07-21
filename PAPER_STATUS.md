# 论文实验状态

## 最终决策
- **路径**: 2（使用多源公开图像数据集替代 AVA）
- **标题**: 面向艺术与设计图像的轻量级视觉复杂度与审美感知可解释预测模型
- **数据集**: 多源公开图像数据集（ModelScope 国内镜像）

## 数据概况
- 总图像: 17,337 张
- 类别: painting (6450), ui (10386), packaging (324), poster (100), banner (77)
- 来源: ModelScope 国内镜像（WikiArt, Phone_Photosho_UI, ShowUI-desktop, Korean Product Labels, poster-design, Korean Billboards 等）

## 核心实验结果

### 建模性能（5-fold CV）
| 目标 | Ridge | RF | GBR |
|------|-------|-----|-----|
| overall | 0.818 | 0.810 | 0.801 |
| emotion | 0.774 | 0.762 | 0.754 |
| complexity | 0.333 | 0.290 | 0.252 |
| beauty | 0.253 | 0.208 | 0.166 |
| order | 0.070 | 0.068 | 0.028 |

### SHAP Top-5 特征
1. saturation_mean (0.258)
2. symmetry (0.007)
3. color_harmony (0.007)
4. saturation_std (0.007)
5. center_offset_x (0.007)

### 统计检验
全部 5 个感知维度在 5 类图像间存在极显著差异（Kruskal-Wallis p < 1e-66）

## 论文可用文件
- results/tables/: 15 个 CSV 结果表
- results/figures/: 38 张论文图表
- src/: 完整可复现代码
- subjective_experiment/: 浏览器评分工具（可用于补充实验）

## IRB 状态
无需 IRB。评分基于计算美学模型（文献公认方法），无新增人类被试。

## 下一步
选择目标期刊（JVCIR / IVC / TVC / PRL），开始撰写论文。
