#!/usr/bin/env python3
"""
Step 4 — Reclassify and Redesign the Proxy Labels
Document the theory-derived proxy targets and quantify overlap with student features.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data" / "documentation"
QC = ROOT / "results" / "redesign" / "quality_control"
DOC.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

spec_md = """# Proxy Target Specification

> Human-Light redesign — Step 4
> Status: **Transparent theory index**

## 1. Redesign decision

The original Phase 1 targets in `data/ratings/aesthetic_scores.csv` were theory-derived scores computed from the same handcrafted features used as model inputs. This creates a circularity risk: a high in-sample \(R^2\) mainly reflects reconstruction of the scoring formula rather than prediction of independent human perception.

Following the redesign plan, **Option 3 — Transparent theory index** is adopted:

- The proxy targets are retained as a **design-theory-grounded index approximation**.
- The Phase 1 task is renamed **proxy-task performance / theory-index approximation**.
- All human-accuracy interpretations are removed from Phase 1.
- Claims about human perception are supported only by the frozen external validation on `D-human-mean`.

## 2. Target definitions

| Dimension | Definition | Formula source | Input features used by target generator |
|---|---|---|---|
| `overall` | Aggregated proxy for overall aesthetic quality | Linear combination of the five dimension proxies | Same feature set as the dimension proxies |
| `beauty` | Color-driven attractiveness proxy | `saturation_mean`, `value_mean`, `color_harmony` | Color features from `traditional_features.csv` |
| `complexity` | Visual complexity proxy | `edge_density`, `edge_orientation_entropy`, `gray_entropy`, `gradient_energy`, `color_entropy`, `num_dominant_colors`, `lightness_contrast` | Edge, texture, color features from `traditional_features.csv` |
| `order` | Perceived orderliness proxy | `symmetry`, `rule_of_thirds`, `edge_orientation_entropy` (negative) | Composition and edge features from `traditional_features.csv` |
| `hierarchy` | Visual hierarchy clarity proxy | `symmetry`, `rule_of_thirds`, `saliency_mean`, `lightness_contrast` | Composition, saliency, contrast features from `traditional_features.csv` |
| `emotion` | Emotional intensity proxy | `saturation_mean`, `saturation_std`, `warm_color_ratio`, `saliency_mean` | Color and saliency features from `traditional_features.csv` |

## 3. Source publications / rationale

The weights were chosen heuristically based on common design-theory and empirical aesthetics literature:

- Complexity: edge density and color variety are standard correlates of visual complexity (Forsythe et al., 2011; Oliva & Torralba, 2001).
- Beauty / preference: saturation, brightness, and color harmony are widely reported predictors (Palmer & Schloss, 2010; Valdez & Mehrabian, 1994).
- Order / hierarchy: symmetry and rule-of-thirds composition are classical order and hierarchy cues (Arnheim, 1954; Locher et al., 1998).
- Emotion: warm colors and high saturation are associated with higher arousal (Russell & Pratt, 1980).

**Important**: These are simplified theory-derived indices, not validated human ground-truth scales.

## 4. Known limitations

1. **Circularity**: The target generator uses a subset of the same features that the student model receives. High proxy-task \(R^2\) is therefore expected and does not validate human-perception prediction.
2. **Simplified weights**: The weights are fixed heuristics, not learned from human judgments.
3. **Domain specificity**: The indices were derived for art and design images and may not generalize to other domains.
4. **No inter-rater validation**: The proxy targets were not compared against human ratings during construction.

## 5. Naming convention

In all redesigned outputs:

- Use "proxy score" or "theory-derived index", never "human rating" or "ground truth".
- Phase 1 metrics are labeled "proxy-task performance".
- Phase 2 / frozen validation metrics are labeled "human alignment".
"""

(DOC / "proxy_target_specification.md").write_text(spec_md, encoding="utf-8")
print("Wrote data/documentation/proxy_target_specification.md")

overlap_csv = [
    ["dimension", "target_generator_features", "student_feature_set", "overlap_count", "overlap_feature_names", "circularity_risk"],
    ["beauty", "saturation_mean,value_mean,color_harmony", "traditional 30 + enhanced 14", "3", "saturation_mean,value_mean,color_harmony", "High"],
    ["complexity", "edge_density,edge_orientation_entropy,gray_entropy,gradient_energy,color_entropy,num_dominant_colors,lightness_contrast", "traditional 30 + enhanced 14", "7", "edge_density,edge_orientation_entropy,gray_entropy,gradient_energy,color_entropy,num_dominant_colors,lightness_contrast", "High"],
    ["order", "symmetry,rule_of_thirds,edge_orientation_entropy", "traditional 30 + enhanced 14", "3", "symmetry,rule_of_thirds,edge_orientation_entropy", "High"],
    ["hierarchy", "symmetry,rule_of_thirds,saliency_mean,lightness_contrast", "traditional 30 + enhanced 14", "4", "symmetry,rule_of_thirds,saliency_mean,lightness_contrast", "High"],
    ["emotion", "saturation_mean,saturation_std,warm_color_ratio,saliency_mean", "traditional 30 + enhanced 14", "4", "saturation_mean,saturation_std,warm_color_ratio,saliency_mean", "High"],
    ["overall", "linear combination of dimension proxies", "traditional 30 + enhanced 14", "all dimension features indirectly", "(inherited)", "High"],
]

with open(QC / "proxy_feature_overlap_audit.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(overlap_csv)
print("Wrote results/redesign/quality_control/proxy_feature_overlap_audit.csv")

print("Step 4 complete.")
