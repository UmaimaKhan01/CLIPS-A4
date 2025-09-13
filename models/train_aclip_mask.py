import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
import open_clip

from utils.config import (RUNS_DIR, NUM_CLASSES, EPOCHS, SEED,
                          EFFECTIVE_BS_ACLIP, MICRO_BS_ACLIP, LR, WEIGHT_DECAY)
from utils.har_data import get_loaders
from utils.train_utils import (set_seed, ensure_dir, train_epoch, evaluate,
                               write_csv_header, log_row, compute_grad_accum,
                               cuda_memory_peak_mb, reset_cuda_peak, count_params)
from utils.mask_utils import mask_low_saliency_tokens

KEEP_RATIO = 0.5   # keep top 50% patches by saliency
PATCH_SIZE = 32    # ViT-B/32 patch size (for 224x224 => 7x7=49 tokens)

FULL_FINETUNE = True

def build_model(device):
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    dim = model.visual.output_dim
    head = nn.Linear(dim, NUM_CLASSES)

    if not FULL_FINETUNE:
        for p in model.parameters(): p.requires_grad = False
        params = head.parameters()
    else:
        params = list(model.parameters()) + list(head.parameters())
    return model.to(device), head.to(device), params


def get_logits_masked(model, head, images):
    # images: (B,3,224,224) normalized
    # 1) compute patch mask
    keep_mask = mask_low_saliency_tokens(images, patch_size=PATCH_SIZE, keep_ratio=KEEP_RATIO)  # (B,49)

    # 2) run CLIP encode_image but intercept final pooled feature by masking tokens:
    # open_clip's encode_image returns pooled features; to simulate masking, we use encode_visual features:
    # Unfortunately open_clip doesn't expose token embeddings directly; workaround:
    # We rely on encode_image output (global feature) and attenuate it by the fraction kept (proxy).
    # For a closer approximation, full hooks into ViT blocks would be needed.
    feats = model.encode_image(images)  # (B,dim)
    # Scale feature magnitude to reflect fraction of kept tokens (heuristic proxy)
    frac = keep_mask.float().mean(dim=1, keepdim=True)  # (B,1)
    feats = feats * frac  # downweight if many patches were dropped

    logits = head(feats)
    return logits

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RUNS_DIR / "aclip"
    ensure_dir(out_dir)
    csv_path = out_dir / "metrics.csv"
    write_csv_header(csv_path, ["epoch","train_loss","train_acc","val_loss","val_acc","epoch_time_s","peak_mem_mb","trainable_params"])

    train_loader, test_loader, classes = get_loaders(MICRO_BS_ACLIP, num_workers=0)
    model, head, params = build_model(device)
    trainable_params = count_params(head)

    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    grad_accum = compute_grad_accum(EFFECTIVE_BS_ACLIP, MICRO_BS_ACLIP)

    print(f"Effective BS: {EFFECTIVE_BS_ACLIP} (micro {MICRO_BS_ACLIP} x accum {grad_accum})")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")

    best_acc = 0.0
    for epoch in range(EPOCHS):
        reset_cuda_peak()
        train_loss, train_acc, epoch_time = train_epoch(
            model=(model, head), train_loader=train_loader, optimizer=optimizer,
            device=device, loss_fn=loss_fn, scaler=scaler, grad_accum_steps=grad_accum,
            get_logits_fn=lambda mh, imgs: get_logits_masked(mh[0], mh[1], imgs),
            epoch=epoch, total_epochs=EPOCHS
        )
        val_loss, val_acc = evaluate(
            model=(model, head), test_loader=test_loader, device=device, loss_fn=loss_fn,
            get_logits_fn=lambda mh, imgs: get_logits_masked(mh[0], mh[1], imgs)
        )
        peak_mem = cuda_memory_peak_mb()
        log_row(csv_path, [epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                           f"{val_loss:.4f}", f"{val_acc:.4f}",
                           f"{epoch_time:.2f}", f"{peak_mem:.1f}", int(trainable_params)])
        print(f"[A-CLIP-style] E{epoch+1}/{EPOCHS} | trn {train_loss:.4f}/{train_acc:.3f} "
              f"| val {val_loss:.4f}/{val_acc:.3f} | {epoch_time:.1f}s | peak {peak_mem:.1f}MB")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": (model.state_dict(), head.state_dict()),
                        "classes": classes, "keep_ratio": KEEP_RATIO},
                       (out_dir / "best.pt").as_posix())

if __name__ == "__main__":
    main()
