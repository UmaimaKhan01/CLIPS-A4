# CLIP vs A-CLIP vs SigLIP on Human Action Recognition

This repository contains the implementation and experiments for comparing **CLIP**, **Attentive Mask CLIP (A-CLIP)**, and **Sigmoid Loss CLIP (SigLIP)** on the **Human Action Recognition (HAR)** dataset.

##  Overview
I fine-tuned three vision–language models on the same HAR dataset split:
- **CLIP (ViT-B/32)**: Standard baseline with contrastive loss.
- **A-CLIP**: Adds attentive masking to focus on salient tokens.
- **SigLIP**: Replaces softmax with sigmoid loss and supports very large batch sizes.

---

##  Dataset
- **Dataset**: Human Action Recognition (HAR)
- **Classes**: 15 (e.g., calling, cycling, eating, running, etc.)
- **Split**: Provided fixed train/test split
- **Image size**: 224×224

---

##  Training Setup
- **Loss**: Cross-entropy
- **Optimizer**: AdamW + cosine schedule
- **Precision**: Mixed precision (AMP)
- **Epochs**: 10 for each model
- **Effective Batch Size**:
  - CLIP, A-CLIP → 4096  
  - SigLIP → 32k (via gradient accumulation)

---

## Results

### Final Performance
| Model   | Top-1 Acc (%) | Params   | Peak Mem (MB) | Epoch Time (s) |
|---------|--------------|----------|---------------|----------------|
| CLIP    | **45.8**     | 151.3M   | 3608          | ~95            |
| A-CLIP  | **48.8**     | 0.01M\*  | 2970          | ~100           |
| SigLIP  | **69.2**     | 92.9M    | 6188          | ~320           |

\*Trainable params only (backbone mostly frozen).

---

### Training Progression (Val Accuracy %)
| Epoch | CLIP | A-CLIP | SigLIP |
|-------|------|--------|--------|
| 1     | 17.1 | 13.9   | 29.1   |
| 5     | 31.1 | 30.8   | 68.5   |
| 10    | 45.8 | 48.8   | 69.2   |

---

##  Key Insights
- **SigLIP** is the best performing model by a wide margin, thanks to sigmoid loss and large batch scaling.
- **A-CLIP** improves slightly over CLIP, especially in later epochs, but gains are modest.
- **CLIP** remains a strong baseline with fast training but limited fine-grained accuracy.

##  How to Run
```bash
# Clone repo
git clone https://github.com/UmaimaKhan01/CLIPS-A4.git
cd CLIPS-A4

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Train CLIP
python models/train_clip.py

# Train A-CLIP
python models/train_aclip_mask.py

# Train SigLIP
python models/train_siglip.py

📎 Reference
Radford et al., Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)

Xu et al., Masked Vision-Language Models for Robust Representation Learning (NeurIPS 2022)

Chen et al., Improving Contrastive Learning by Sigmoid Loss (CVPR 2023)
