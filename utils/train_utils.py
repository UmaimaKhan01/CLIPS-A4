# utils/train_utils.py
import time, csv
import torch
import torch.nn.functional as F
from pathlib import Path
from rich.progress import track

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def count_params(model):
    if isinstance(model, (list, tuple)):
        return sum(p.numel() for m in model if hasattr(m, "parameters")
                   for p in m.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(p: Path):
    p = Path(p)
    if p.exists() and not p.is_dir():
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_epoch(model, train_loader, optimizer, device, loss_fn,
                scaler: torch.cuda.amp.GradScaler, grad_accum_steps: int,
                get_logits_fn, epoch: int, total_epochs: int):
    # Put everything into train mode
    if isinstance(model, (list, tuple)):
        for m in model:
            if hasattr(m, "train"): m.train()
    else:
        model.train()

    start = time.time()
    total_loss, total_acc, seen = 0.0, 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(track(train_loader, description=f"Train E{epoch+1}/{total_epochs}")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = get_logits_fn(model, images)
            loss = loss_fn(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            total_loss += loss.item() * grad_accum_steps
            total_acc  += accuracy(logits, labels) * images.size(0)
            seen       += images.size(0)

    return total_loss / max(1, step+1), total_acc / max(1, seen), time.time() - start

@torch.no_grad()
def evaluate(model, test_loader, device, loss_fn, get_logits_fn):
    if isinstance(model, (list, tuple)):
        for m in model:
            if hasattr(m, "eval"): m.eval()
    else:
        model.eval()

    total_loss, total_acc, seen = 0.0, 0.0, 0
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = get_logits_fn(model, images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels) * images.size(0)
        seen       += images.size(0)
    return total_loss / max(1, len(test_loader)), total_acc / max(1, seen)

def write_csv_header(csv_path, headers):
    newfile = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        if newfile:
            csv.writer(f).writerow(headers)

def log_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def compute_grad_accum(effective_bs, micro_bs):
    if effective_bs % micro_bs != 0:
        raise ValueError(f"effective_bs {effective_bs} must be divisible by micro_bs {micro_bs}")
    return effective_bs // micro_bs

def cuda_memory_peak_mb():
    if not torch.cuda.is_available(): return -1
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
