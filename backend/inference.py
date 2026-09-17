"""
inference.py — Cervical Cell Hybrid Cross-Attention Classifier

Loads the four packaged artifacts (segmentation U-Net, ViT backbone,
cross-attention model, M2 morphology scaler) from a local export directory
or by downloading them from the Hugging Face model repo, and exposes a
single CervicalCellPredictor.predict(image) function.

Model weights live on Hugging Face:
https://huggingface.co/NamrahDurrani/Huggingface-deployement

Usage:
    from inference import CervicalCellPredictor
    predictor = CervicalCellPredictor.from_pretrained("NamrahDurrani/Huggingface-deployement")
    result = predictor.predict(pil_image)
"""

import os
import json

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms, models

CLASSES = ["NILM", "LSIL", "HSIL", "SCC"]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
SEG_TFM = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), NORM])
CLS_TFM = transforms.Compose([transforms.ToTensor(), NORM])
PADDING_FRAC = 0.0


# ---- Model classes ----

class FineTunedViT(nn.Module):
    def __init__(self, n_classes=4, unfreeze_blocks=2):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        self.backbone.heads = nn.Identity()
        self.head = nn.Linear(768, n_classes)


def get_full_tokens(vit_backbone, images):
    x = vit_backbone._process_input(images)
    n = x.shape[0]
    cls = vit_backbone.class_token.expand(n, -1, -1)
    x = torch.cat([cls, x], dim=1)
    return vit_backbone.encoder(x)


class MorphologyBranch(nn.Module):
    def __init__(self, in_dim=22, emb_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, emb_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MorphologyGuidedCrossAttention(nn.Module):
    def __init__(self, morph_dim=32, token_dim=768, attn_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(morph_dim, attn_dim)
        self.k_proj = nn.Linear(token_dim, attn_dim)
        self.v_proj = nn.Linear(token_dim, attn_dim)
        self.mha = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.ffn = nn.Sequential(nn.Linear(attn_dim, attn_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(attn_dim, attn_dim))
        self.norm2 = nn.LayerNorm(attn_dim)

    def forward(self, morph_embedding, patch_tokens, need_weights=False):
        q = self.q_proj(morph_embedding).unsqueeze(1)
        k = self.k_proj(patch_tokens)
        v = self.v_proj(patch_tokens)
        attn_out, attn_weights = self.mha(q, k, v, need_weights=need_weights, average_attn_weights=True)
        x = self.norm1(attn_out + q)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(1), (attn_weights.squeeze(1) if need_weights else None)


class FusionHead(nn.Module):
    def __init__(self, in_dim, n_classes=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


class CrossAttentionHybrid(nn.Module):
    def __init__(self, morph_in_dim=22):
        super().__init__()
        self.morphology_branch = MorphologyBranch(morph_in_dim)
        self.cross_attention = MorphologyGuidedCrossAttention()
        self.fusion_head = FusionHead(in_dim=768 + 128 + 32)

    def forward(self, vit_tokens, morphology, need_weights=False):
        cls_token = vit_tokens[:, 0, :]
        patch_tokens = vit_tokens[:, 1:, :]
        morph = self.morphology_branch(morphology)
        attended, attn_weights = self.cross_attention(morph, patch_tokens, need_weights)
        fused = torch.cat([cls_token, attended, morph], dim=1)
        return self.fusion_head(fused), attn_weights


# ---- Out-of-distribution pre-filter ----
# Real Pap-stained LBC images are dominated by blue/purple (hematoxylin) and
# pink/magenta (counterstain) hues. This rejects images whose color profile
# doesn't resemble stained cytology at all (cartoons, text graphics, photos,
# etc.) BEFORE running segmentation, catching cases the "no cell detected"
# check alone misses — e.g. an out-of-distribution image where the
# segmentation model still finds *some* blob-shaped region.
#
# This is a heuristic, not a learned check — it will not catch every
# out-of-distribution image, and should be documented as such.

def looks_like_stained_cytology(pil_image: Image.Image, min_stain_fraction: float = 0.15) -> bool:
    hsv = np.array(pil_image.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Hue ranges (0-255 scale from PIL's HSV) roughly covering hematoxylin
    # blue-purple and eosin/counterstain pink-magenta tones.
    blue_purple = (h >= 120) & (h <= 200)
    pink_magenta = ((h >= 200) | (h <= 15)) & (s > 40)

    stain_like = (blue_purple | pink_magenta) & (s > 25) & (v > 20) & (v < 245)
    stain_fraction = stain_like.mean()

    return stain_fraction >= min_stain_fraction


# ---- Segmentation-guided crop ----

def predict_mask_and_bbox(seg_model, device, img_pil, padding_frac=0.0):
    img_np = np.array(img_pil.resize((256, 256)))
    with torch.no_grad():
        t = SEG_TFM(img_pil).unsqueeze(0).to(device)
        mask = (torch.sigmoid(seg_model(t)) > 0.5).float()[0, 0].cpu().numpy()
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img_np, mask, (0, 0, 256, 256)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0, x1 - x0
    pad_y, pad_x = int(h * padding_frac), int(w * padding_frac)
    y0, y1 = max(0, y0 - pad_y), min(256, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(256, x1 + pad_x)
    return img_np, mask, (y0, y1, x0, x1)


def crop_and_resize(img_np, bbox, out_size=224):
    y0, y1, x0, x1 = bbox
    crop = img_np[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img_np
    return np.array(Image.fromarray(crop).resize((out_size, out_size)))


# ---- Morphology extraction (M2, 22 features) ----

def get_clean_contours(mask, min_area=30):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def extract_morph_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if len(contour) >= 5:
        (_, _), (maj, minr), _ = cv2.fitEllipse(contour)
        major_axis, minor_axis = max(maj, minr), min(maj, minr)
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2)) if major_axis > 0 else 0
    else:
        major_axis = minor_axis = eccentricity = 0
    equiv_diameter = np.sqrt(4 * area / np.pi)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
    return {
        "Area": area, "Perimeter": perimeter, "Circularity": circularity,
        "Solidity": solidity, "Extent": extent, "Aspect_Ratio": aspect_ratio,
        "Equivalent_Diameter": equiv_diameter, "Convex_Area": hull_area,
        "Major_Axis_Length": major_axis, "Minor_Axis_Length": minor_axis,
        "Eccentricity": eccentricity, "Bounding_Box_Width": w, "Bounding_Box_Height": h,
    }


def extract_m1_aggregated(mask, min_area=30):
    contours = get_clean_contours(mask, min_area)
    if not contours:
        raise ValueError("No cell region found — input-quality gate should catch this upstream.")
    rows = [extract_morph_features(c) for c in contours]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["Cell_Count"] = len(contours)
    return agg


def extract_m2_features(mask, min_area=30):
    m1 = extract_m1_aggregated(mask, min_area)
    eps = 1e-6
    m2 = dict(m1)
    m2["Elongation"] = 1 - (m1["Minor_Axis_Length"] / (m1["Major_Axis_Length"] + eps))
    m2["BBox_Aspect_Ratio"] = m1["Bounding_Box_Width"] / (m1["Bounding_Box_Height"] + eps)
    m2["Convex_Deficiency"] = m1["Convex_Area"] - m1["Area"]
    m2["Diameter_Convex_Ratio"] = m1["Equivalent_Diameter"] / (np.sqrt(m1["Convex_Area"]) + eps)
    m2["Log_Area"] = np.log1p(m1["Area"])
    m2["Log_Convex_Area"] = np.log1p(m1["Convex_Area"])
    m2["Cell_Density_Proxy"] = m1["Cell_Count"] / (m1["Area"] + eps)
    m2["Diameter_MajorAxis_Ratio"] = m1["Equivalent_Diameter"] / (m1["Major_Axis_Length"] + eps)
    order = [
        "Area", "Perimeter", "Circularity", "Solidity", "Extent", "Aspect_Ratio",
        "Equivalent_Diameter", "Convex_Area", "Major_Axis_Length", "Minor_Axis_Length",
        "Eccentricity", "Bounding_Box_Width", "Bounding_Box_Height", "Cell_Count",
        "Elongation", "BBox_Aspect_Ratio", "Convex_Deficiency", "Diameter_Convex_Ratio",
        "Log_Area", "Log_Convex_Area", "Cell_Density_Proxy", "Diameter_MajorAxis_Ratio",
    ]
    return np.array([m2[k] for k in order], dtype=np.float32)


# ---- Full pipeline ----

class CervicalCellPredictor:
    def __init__(self, export_dir, device="cpu"):
        self.device = device
        with open(os.path.join(export_dir, "config.json")) as f:
            self.config = json.load(f)

        self.seg_model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
        self.seg_model.load_state_dict(load_file(os.path.join(export_dir, "segmentation_unet_resnet34.safetensors")))
        self.seg_model.to(device).eval()

        self.vit = FineTunedViT()
        self.vit.load_state_dict(load_file(os.path.join(export_dir, "vit_backbone_seed2.safetensors")))
        self.vit.to(device).eval()

        self.ca_model = CrossAttentionHybrid(morph_in_dim=22)
        self.ca_model.load_state_dict(load_file(os.path.join(export_dir, "cross_attention_hybrid_seed2.safetensors")))
        self.ca_model.to(device).eval()

        self.scaler = joblib.load(os.path.join(export_dir, "morphology_scaler_m2_train_only.pkl"))

    @classmethod
    def from_pretrained(cls, repo_id="NamrahDurrani/Huggingface-deployement", device="cpu", cache_dir=None):
        """Downloads all artifacts from the Hugging Face Hub and returns a ready-to-use predictor."""
        from huggingface_hub import snapshot_download
        export_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        return cls(export_dir, device=device)

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> dict:
        pil_image = pil_image.convert("RGB")

        if not looks_like_stained_cytology(pil_image):
            return {
                "predicted_class": None,
                "class_scores": None,
                "confidence": None,
                "status": "REJECTED_NOT_CYTOLOGY_IMAGE",
                "message": (
                    "This image's color profile does not resemble a stained "
                    "cytology slide. Rejected before classification."
                ),
            }

        img_np, mask, bbox = predict_mask_and_bbox(self.seg_model, self.device, pil_image, PADDING_FRAC)
        crop = crop_and_resize(img_np, bbox, out_size=224)

        try:
            raw_features = extract_m2_features(mask).reshape(1, -1)
        except ValueError:
            return {
                "predicted_class": None,
                "class_scores": None,
                "confidence": None,
                "status": "REJECTED_NO_CELL_DETECTED",
                "message": "No usable cell region found in this image. Manual review required.",
            }

        scaled_features = torch.tensor(self.scaler.transform(raw_features), dtype=torch.float32).to(self.device)
        crop_tensor = CLS_TFM(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        tokens = get_full_tokens(self.vit.backbone, crop_tensor)
        logits, attn_weights = self.ca_model(tokens, scaled_features, need_weights=True)
        probs = torch.softmax(logits, dim=1)[0]

        return {
            "predicted_class": CLASSES[probs.argmax().item()],
            "class_scores": {c: float(p) for c, p in zip(CLASSES, probs)},
            "confidence": float(probs.max()),
            "status": "OK",
            "attention_weights": attn_weights[0].tolist() if attn_weights is not None else None,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    predictor = CervicalCellPredictor.from_pretrained()
    result = predictor.predict(Image.open(sys.argv[1]))
    print(json.dumps(result, indent=2))