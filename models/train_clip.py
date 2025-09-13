import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import open_clip

from utils.config import (RUNS_DIR, NUM_CLASSES, EPOCHS, SEED,
                          EFFECTIVE_BS_CLIP, MICRO_BS_CLIP, LR, WEIGHT_DECAY, WARMUP_EPOCHS)
from utils.har_data import get_loaders
from utils.train_utils import (set_seed, ensure_dir, train_epoch, evaluate,
                               write_csv_header, log_row, compute_grad_accum,
                               cuda_memory_peak_mb, reset_cuda_peak, count_params)

"""
CLIP backbone: ViT-B/32 from open_clip.
We do a linear probe by default (freeze encoder; train a classifier on pooled vision features).
You can set FULL_FINETUNE=True to unfreeze.
"""

FULL_FINETUNE = True

def build_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    # We'll use the visual trunk outputs (global pooled features)
    # Add a linear head to NUM_CLASSES
    dim = model.visual.output_dim
    head = nn.Linear(dim, NUM_CLASSES)

    if not FULL_FINETUNE:
        for p in model.parameters():
            p.requires_grad = False
        params = head.parameters()
    else:
        params = list(model.parameters()) + list(head.parameters())

    return model.to(device), head.to(device), params

def get_logits_fn(model, head, images):
    # images already preprocessed in our dataloader (CLIP-normalized + 224x224)
    with torch.no_grad() if all(not p.requires_grad for p in model.parameters()) else torch.enable_grad():
        feats = model.encode_image(images)  # (B, dim)
    logits = head(feats)  # (B, NUM_CLASSES)
    return logits

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RUNS_DIR / "clip"
    ensure_dir(out_dir)
    csv_path = out_dir / "metrics.csv"
    write_csv_header(csv_path, ["epoch","train_loss","train_acc","val_loss","val_acc","epoch_time_s","peak_mem_mb","trainable_params"])

    # Data
    train_loader, test_loader, classes = get_loaders(MICRO_BS_CLIP, num_workers=0)

    # Model
    model, head, params = build_model(device)
    trainable_params = count_params(model) + count_params(head)

    # Optimizer & loss
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    grad_accum = compute_grad_accum(EFFECTIVE_BS_CLIP, MICRO_BS_CLIP)

    print(f"Classes: {classes}")
    print(f"Effective BS: {EFFECTIVE_BS_CLIP} (micro {MICRO_BS_CLIP} x accum {grad_accum})")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    best_acc = 0.0
    for epoch in range(EPOCHS):
        reset_cuda_peak()
        train_loss, train_acc, epoch_time = train_epoch(
            model=(model, head), train_loader=train_loader, optimizer=optimizer, device=device,
            loss_fn=loss_fn, scaler=scaler, grad_accum_steps=grad_accum,
            get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs),
            epoch=epoch, total_epochs=EPOCHS
        )
        val_loss, val_acc = evaluate(
            model=(model, head), test_loader=test_loader, device=device, loss_fn=loss_fn,
            get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs)
        )
        peak_mem = cuda_memory_peak_mb()
        log_row(csv_path, [epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                           f"{val_loss:.4f}", f"{val_acc:.4f}",
                           f"{epoch_time:.2f}", f"{peak_mem:.1f}", int(trainable_params)])
        print(f"[CLIP] E{epoch+1}/{EPOCHS} | trn {train_loss:.4f}/{train_acc:.3f} "
              f"| val {val_loss:.4f}/{val_acc:.3f} | {epoch_time:.1f}s | peak {peak_mem:.1f}MB")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": (model.state_dict(), head.state_dict()),
                        "classes": classes},
                       (out_dir / "best.pt").as_posix())

if __name__ == "__main__":
    main()
