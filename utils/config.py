# utils/config.py
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\umaim\Downloads\CLIPS_A4\Data")

# Logging/output
RUNS_DIR = Path("./runs")

# Common training settings
NUM_CLASSES = 15
EPOCHS = 10
SEED = 42

# Image size used across models (CLIP ViT-B/32, SigLIP-base use 224)
IMAGE_SIZE = 224

# Effective batch sizes per assignment
EFFECTIVE_BS_CLIP = 4096
EFFECTIVE_BS_ACLIP = 4096
EFFECTIVE_BS_SIGLIP = 32768

# Per-device micro-batch sizes (adjust if you OOM; increase if you have big GPU)
MICRO_BS_CLIP = 64
MICRO_BS_ACLIP = 64
MICRO_BS_SIGLIP = 128  # SigLIP favors large batches; we simulate via accumulation

# Optimizer, scheduler
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 1
