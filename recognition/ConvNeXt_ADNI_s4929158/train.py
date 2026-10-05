"""Train, validate and test one model (MLP, CNN or ConvNeXt) on ADNI AD-vs-NC.

All three models share exactly the same data splits, augmentation, optimiser,
schedule and epoch budget, so differences in the results come from the
architecture and not from the training recipe.

Protocol
--------
* Train on the patient-level training split; select the best epoch by
  *validation loss* (the test set is never used for any decision).
* After training, reload the best checkpoint and evaluate on the held-out test
  set exactly once. Per-slice test logits are saved so that calibration,
  reject-option and failure-case analysis can be done without re-running the model.
* Evaluation is per-slice (each image counted separately), which is the
  recommended protocol for a 2D model.

Outputs (in ``--out-dir``, default ``runs/<model>_seed<seed>``):
    best.pt                checkpoint (NOT to be committed)
    history.csv            per-epoch loss / accuracy / AUROC
    curves.png             loss and metric curves
    test_predictions.csv   per-slice test logits, one row per image
    results.json           test metrics, parameter count, peak VRAM, latency, args

Example:
    python train.py --model convnext --epochs 40
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # no display on the cluster
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import IDX_TO_CLASS, get_datasets
from modules import build_model, count_parameters


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs. Some GPU ops stay slightly non-deterministic."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """AdamW groups: no weight decay on biases, norm parameters or layer scale.

    All of these are 1-D tensors, so a dimensionality test is enough. This is
    the usual convention for ConvNeXt/transformer training and is applied to
    every model so the recipe stays identical.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Run the model over a loader and return metrics plus raw logits.

    Evaluation runs in full fp32 (no autocast) so saved logits are precise,
    which matters for the calibration analysis.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, n = 0.0, 0
    all_logits, all_labels, all_idx = [], [], []
    for images, labels, idx in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        n += labels.numel()
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
        all_idx.append(idx)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    idx = torch.cat(all_idx)
    probs_ad = torch.softmax(logits, dim=1)[:, 1]
    return {
        "loss": total_loss / n,
        "acc": (logits.argmax(1) == labels).float().mean().item(),
        "auroc": float(roc_auc_score(labels.numpy(), probs_ad.numpy())),
        "logits": logits,
        "labels": labels,
        "idx": idx,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, step, total_steps, warmup_steps, base_lr, amp, clip):
    """One pass over the training data. Returns (loss, accuracy, new_global_step)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        lr = base_lr * lr_multiplier(step, total_steps, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.numel()
        step += 1
    return loss_sum / n, correct / n, step


@torch.no_grad()
def measure_inference(model: nn.Module, device: torch.device, img_size: int) -> Dict[str, float]:
    """Latency (batch 1, ms/image) and throughput (batch 64, images/s)."""
    model.eval()

    def timed(batch: int, iters: int) -> float:
        x = torch.randn(batch, 1, img_size, img_size, device=device)
        for _ in range(5):  # warm-up
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters

    return {
        "latency_ms_batch1": timed(1, 50) * 1000.0,
        "throughput_img_per_s_batch64": 64.0 / timed(64, 10),
    }


def save_history(history: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: List[Dict], path: Path, title: str) -> None:
    """Plot loss, accuracy and validation AUROC against epoch."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axes[2].plot(epochs, [h["val_auroc"] for h in history], label="val AUROC", color="tab:green")
    axes[2].set(title="Validation AUROC", xlabel="epoch", ylabel="AUROC")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_test_predictions(result: Dict, samples, path: Path) -> None:
    """Write one CSV row per test slice so analysis can reuse the logits."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "patient", "slice_idx", "label", "logit_nc", "logit_ad"])
        for logit, label, i in zip(result["logits"], result["labels"], result["idx"]):
            s = samples[int(i)]
            writer.writerow([s.path, s.patient, s.slice_idx, int(label), f"{logit[0]:.6f}", f"{logit[1]:.6f}"])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one model on ADNI AD vs NC")
    p.add_argument("--model", choices=["mlp", "cnn", "convnext"], required=True)
    p.add_argument("--root", default="/home/groups/comp3710/ADNI/AD_NC")
    p.add_argument("--out-dir", default=None, help="default: runs/<model>_seed<seed>")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm, 0 disables")
    p.add_argument("--patience", type=int, default=10, help="early stopping on the selection metric, 0 disables")
    p.add_argument("--select-metric", choices=["loss", "auroc", "acc"], default="loss",
                   help="validation metric used to pick the best epoch and for early stopping")
    p.add_argument("--drop-path", type=float, default=None,
                   help="ConvNeXt stochastic-depth rate (default: model default, 0.1)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--group-by", choices=["subject", "scan"], default="subject",
                   help="unit kept intact when splitting train/val (subject needs the metadata JSON)")
    p.add_argument("--meta", default=None, help="metadata JSON (default: next to the dataset folder)")
    p.add_argument("--drop-test-overlap", action="store_true",
                   help="remove subjects that also appear in the test folder from the training data")
    p.add_argument("--split-seed", type=int, default=42, help="fixes the patient split across all runs")
    p.add_argument("--seed", type=int, default=0, help="training seed (init, shuffling, augmentation)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision training (GPU only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and device.type == "cuda"
    set_seed(args.seed)
    out_dir = Path(args.out_dir or f"runs/{args.model}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data: the split depends only on --split-seed, so every model sees identical splits.
    train_ds, val_ds, test_ds = get_datasets(
        args.root, args.val_fraction, args.split_seed, args.img_size,
        group_by=args.group_by, meta_path=args.meta, drop_test_overlap=args.drop_test_overlap,
    )
    gen = torch.Generator().manual_seed(args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, generator=gen, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    print(f"device={device} amp={amp} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} slices")

    model_kwargs = {"drop_path_rate": args.drop_path} if args.drop_path is not None and args.model == "convnext" else {}
    model = build_model(args.model, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"model={args.model} trainable parameters={n_params:,}")

    optimizer = torch.optim.AdamW(make_param_groups(model, args.weight_decay), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history: List[Dict] = []
    best_val_loss, best_epoch, step = float("inf"), -1, 0
    best_score = float("-inf")  # higher is better; loss is negated so one comparison works for all metrics
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, step, total_steps,
            warmup_steps, args.lr, amp, args.clip_grad,
        )
        val = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val["loss"], "val_acc": val["acc"], "val_auroc": val["auroc"],
            "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - t0,
        })
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {val['loss']:.4f} acc {val['acc']:.4f} auroc {val['auroc']:.4f} | "
              f"{history[-1]['epoch_seconds']:.1f}s")

        score = -val["loss"] if args.select_metric == "loss" else val[args.select_metric]
        if score > best_score:
            best_score, best_val_loss, best_epoch = score, val["loss"], epoch
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "epoch": epoch, "args": vars(args)}, out_dir / "best.pt")
        elif args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"early stopping: no val-{args.select_metric} improvement for {args.patience} epochs")
            break
    train_seconds = time.perf_counter() - train_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

    save_history(history, out_dir / "history.csv")
    plot_history(history, out_dir / "curves.png", f"{args.model} (best epoch {best_epoch})")

    # Final test: best checkpoint, evaluated once.
    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, device)
    save_test_predictions(test, test_ds.samples, out_dir / "test_predictions.csv")
    # Validation logits too: evaluate.py fits temperature scaling and the reject
    # threshold on these, so nothing is ever tuned on the test set.
    val_best = evaluate(model, val_loader, device)
    save_test_predictions(val_best, val_ds.samples, out_dir / "val_predictions.csv")
    inference = measure_inference(model, device, args.img_size)

    results = {
        "model": args.model, "best_epoch": best_epoch, "epochs_run": len(history),
        "test_acc": test["acc"], "test_loss": test["loss"], "test_auroc": test["auroc"],
        "val_loss_at_best": best_val_loss, "trainable_params": n_params,
        "peak_train_vram_mb": peak_vram_mb, "train_seconds": train_seconds,
        "class_names": IDX_TO_CLASS, **inference, "args": vars(args),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTEST (best epoch {best_epoch}): acc {test['acc']:.4f} auroc {test['auroc']:.4f} "
          f"loss {test['loss']:.4f} | latency {inference['latency_ms_batch1']:.2f} ms/img")
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
