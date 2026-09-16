# Cervical Cell Hybrid Cross-Attention Classifier — Inference Code

Code repository for the thesis **"Hybrid U-Net and Transformer Architecture for Multi-Class Classification of Cervical Cells in LBC Images."**

Model weights are hosted separately on Hugging Face (not in this repo — see below):
**https://huggingface.co/NamrahDurrani/Huggingface-deployement**

## What this repo contains

- `inference.py` — the full inference pipeline: segmentation-guided crop → M2 morphology extraction → ViT patch tokens → morphology-guided cross-attention → classification.
- `requirements.txt` — minimal dependencies for inference.
- The deployment/validation notebook used to package and test the model.

This repo does **not** contain the model weights (`.safetensors`, `.pkl`) — those live on the Hugging Face Hub and are downloaded automatically at runtime.

## Usage

```bash
pip install -r requirements.txt
python inference.py path/to/image.jpg
```

or in Python:

```python
from inference import CervicalCellPredictor
from PIL import Image

predictor = CervicalCellPredictor.from_pretrained("NamrahDurrani/Huggingface-deployement")
result = predictor.predict(Image.open("path/to/image.jpg"))
print(result)
```

## Output

```json
{
  "predicted_class": "HSIL",
  "class_scores": {"NILM": 0.03, "LSIL": 0.08, "HSIL": 0.84, "SCC": 0.05},
  "confidence": 0.84,
  "status": "OK",
  "attention_weights": [...]
}
```

## Model summary

- **Task:** 4-class classification of cervical cells (NILM / LSIL / HSIL / SCC) from LBC RGB images.
- **Architecture:** frozen U-Net (ResNet34) segmentation → M2 morphology features (22-dim) + ViT-B/16 patch tokens → 4-head morphology-guided cross-attention (morphology = Query, patch tokens = Key/Value) → classification head.
- **Test performance:** 97.65% accuracy, 97.04% macro-F1, 97.76% balanced accuracy on a held-out 170-image test set.

## Important limitations

This is a research prototype and **not a standalone medical diagnostic system**. It performs computational classification of individual cell images and does not constitute a clinical diagnosis. No formal calibration (ECE) has been performed on the confidence scores yet, and no confidence-threshold gate is implemented beyond basic no-cell-detected rejection. See the full deployment report for details.
