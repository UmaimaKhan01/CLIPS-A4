# import os, time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from pathlib import Path
# from transformers import SiglipProcessor, SiglipVisionModel

# from utils.config import (RUNS_DIR, NUM_CLASSES, EPOCHS, SEED,
#                           EFFECTIVE_BS_SIGLIP, MICRO_BS_SIGLIP, LR, WEIGHT_DECAY, WARMUP_EPOCHS)
# from utils.har_data import get_loaders
# from utils.train_utils import (set_seed, ensure_dir, train_epoch, evaluate,
#                                write_csv_header, log_row, compute_grad_accum,
#                                cuda_memory_peak_mb, reset_cuda_peak, count_params)

# """
# SigLIP backbone: Base patch16-224 from HuggingFace
# We fine-tune the whole vision encoder + train a linear head for NUM_CLASSES
# """

# MODEL_NAME = "google/siglip-base-patch16-224"
# FULL_FINETUNE = True   # <--- ensure full fine-tuning

# def build_model(device):
#     # Load pretrained vision model
#     backbone = SiglipVisionModel.from_pretrained(MODEL_NAME)
#     dim = backbone.config.hidden_size
#     head = nn.Linear(dim, NUM_CLASSES)

#     if not FULL_FINETUNE:
#         for p in backbone.parameters():
#             p.requires_grad = False
#         params = head.parameters()
#     else:
#         for p in backbone.parameters():
#             p.requires_grad = True
#         params = list(backbone.parameters()) + list(head.parameters())

#     return backbone.to(device), head.to(device), params

# def get_logits_fn(backbone, head, images):
#     outputs = backbone(pixel_values=images)
#     feats = outputs.last_hidden_state[:, 0]  # CLS token, (B, dim)
#     logits = head(feats)
#     return logits


# def main():
#     set_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     out_dir = RUNS_DIR / "siglip"
#     ensure_dir(out_dir)
#     csv_path = out_dir / "metrics.csv"
#     write_csv_header(csv_path, ["epoch","train_loss","train_acc","val_loss","val_acc",
#                                 "epoch_time_s","peak_mem_mb","trainable_params"])

#     # Data
#     train_loader, test_loader, classes = get_loaders(MICRO_BS_SIGLIP, num_workers=0)

#     # Model
#     backbone, head, params = build_model(device)
#     trainable_params = count_params([backbone, head])

#     # Optimizer & loss
#     optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
#     loss_fn = nn.CrossEntropyLoss()
#     scaler = torch.cuda.amp.GradScaler()
#     grad_accum = compute_grad_accum(EFFECTIVE_BS_SIGLIP, MICRO_BS_SIGLIP)

#     print(f"Classes: {classes}")
#     print(f"Effective BS: {EFFECTIVE_BS_SIGLIP} (micro {MICRO_BS_SIGLIP} x accum {grad_accum})")
#     print(f"Trainable params: {trainable_params/1e6:.2f}M")

#     best_acc = 0.0
#     for epoch in range(EPOCHS):
#         reset_cuda_peak()
#         train_loss, train_acc, epoch_time = train_epoch(
#             model=(backbone, head), train_loader=train_loader, optimizer=optimizer, device=device,
#             loss_fn=loss_fn, scaler=scaler, grad_accum_steps=grad_accum,
#             get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs),
#             epoch=epoch, total_epochs=EPOCHS
#         )
#         val_loss, val_acc = evaluate(
#             model=(backbone, head), test_loader=test_loader, device=device, loss_fn=loss_fn,
#             get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs)
#         )
#         peak_mem = cuda_memory_peak_mb()
#         log_row(csv_path, [epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
#                            f"{val_loss:.4f}", f"{val_acc:.4f}",
#                            f"{epoch_time:.2f}", f"{peak_mem:.1f}", int(trainable_params)])
#         print(f"[SigLIP] E{epoch+1}/{EPOCHS} | trn {train_loss:.4f}/{train_acc:.3f} "
#               f"| val {val_loss:.4f}/{val_acc:.3f} | {epoch_time:.1f}s | peak {peak_mem:.1f}MB")

#         # Save best
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save({"state_dict": (backbone.state_dict(), head.state_dict()),
#                         "classes": classes},
#                        (out_dir / "best.pt").as_posix())

# if __name__ == "__main__":
#     main()
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from transformers import SiglipProcessor, SiglipVisionModel

from utils.config import (RUNS_DIR, NUM_CLASSES, EPOCHS, SEED,
                          LR, WEIGHT_DECAY, WARMUP_EPOCHS)
from utils.har_data import get_loaders
from utils.train_utils import (set_seed, ensure_dir, train_epoch, evaluate,
                               write_csv_header, log_row, compute_grad_accum,
                               cuda_memory_peak_mb, reset_cuda_peak, count_params)

"""
OPTION 1: Quick Salvage for SigLIP
- Drastically reduce batch size for faster epochs
- Keep simple CE loss approach
- Should get 15-20% accuracy for comparative results
"""

MODEL_NAME = "google/siglip-base-patch16-224"
FULL_FINETUNE = True

# FIXED: Much smaller batch sizes for speed
EFFECTIVE_BS_SIGLIP = 512   # Down from 32768!
MICRO_BS_SIGLIP = 64        # Down from 128

def build_model(device):
    backbone = SiglipVisionModel.from_pretrained(MODEL_NAME)
    dim = backbone.config.hidden_size
    
    # Better head initialization
    head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(dim, NUM_CLASSES)
    )
    
    # Xavier init for the linear layer
    nn.init.xavier_uniform_(head[1].weight)
    nn.init.zeros_(head[1].bias)

    if FULL_FINETUNE:
        for p in backbone.parameters():
            p.requires_grad = True
        params = list(backbone.parameters()) + list(head.parameters())
    else:
        for p in backbone.parameters():
            p.requires_grad = False
        params = head.parameters()

    return backbone.to(device), head.to(device), params

def get_logits_fn(backbone, head, images):
    outputs = backbone(pixel_values=images)
    feats = outputs.last_hidden_state[:, 0]  # CLS token
    logits = head(feats)
    return logits

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RUNS_DIR / "siglip_quick_fix"
    ensure_dir(out_dir)
    csv_path = out_dir / "metrics.csv"
    write_csv_header(csv_path, ["epoch","train_loss","train_acc","val_loss","val_acc",
                                "epoch_time_s","peak_mem_mb","trainable_params"])

    # Data with smaller batch size
    train_loader, test_loader, classes = get_loaders(MICRO_BS_SIGLIP, num_workers=2)
    
    backbone, head, params = build_model(device)
    trainable_params = count_params([backbone, head])
    grad_accum = compute_grad_accum(EFFECTIVE_BS_SIGLIP, MICRO_BS_SIGLIP)

    # Lower learning rate for fine-tuning
    FIXED_LR = 5e-5
    optimizer = optim.AdamW(params, lr=FIXED_LR, weight_decay=WEIGHT_DECAY)
    
    # Add scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"Classes ({len(classes)}): {classes}")
    print(f"FIXED - Effective BS: {EFFECTIVE_BS_SIGLIP} (micro {MICRO_BS_SIGLIP} x accum {grad_accum})")
    print(f"FIXED - Learning rate: {FIXED_LR}")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    print("⚠️  Using CE loss on SigLIP (suboptimal but fast for comparison)")

    best_acc = 0.0
    for epoch in range(EPOCHS):
        reset_cuda_peak()
        
        train_loss, train_acc, epoch_time = train_epoch(
            model=(backbone, head), train_loader=train_loader, optimizer=optimizer, device=device,
            loss_fn=loss_fn, scaler=scaler, grad_accum_steps=grad_accum,
            get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs),
            epoch=epoch, total_epochs=EPOCHS
        )
        
        val_loss, val_acc = evaluate(
            model=(backbone, head), test_loader=test_loader, device=device, loss_fn=loss_fn,
            get_logits_fn=lambda mh, imgs: get_logits_fn(mh[0], mh[1], imgs)
        )
        
        scheduler.step()
        peak_mem = cuda_memory_peak_mb()
        
        log_row(csv_path, [epoch+1, f"{train_loss:.4f}", f"{train_acc:.4f}",
                           f"{val_loss:.4f}", f"{val_acc:.4f}",
                           f"{epoch_time:.2f}", f"{peak_mem:.1f}", int(trainable_params)])
        
        print(f"[SigLIP-Quick] E{epoch+1}/{EPOCHS} | trn {train_loss:.4f}/{train_acc:.3f} "
              f"| val {val_loss:.4f}/{val_acc:.3f} | {epoch_time:.1f}s | peak {peak_mem:.1f}MB")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "backbone": backbone.state_dict(), 
                "head": head.state_dict(),
                "classes": classes, 
                "best_acc": best_acc
            }, out_dir / "best.pt")

    print(f"✅ Quick salvage completed. Best val acc: {best_acc:.3f}")
    print("Expected: 15-20% accuracy (enough for comparative table)")

if __name__ == "__main__":
    main()