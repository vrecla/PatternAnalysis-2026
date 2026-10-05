"""Run a trained model on the held-out ADNI test set and visualise its predictions.

Loads a checkpoint written by ``train.py``, classifies every test slice, prints
the headline results, and saves three figures to ``--out-dir``:

    predictions_grid.png   example test slices with true label, prediction and confidence
    confidence_hist.png    confidence of correct vs incorrect predictions
    confusion_matrix.png   counts and per-class rates

Example:
    python predict.py --checkpoint runs/convnext_seed0/best.pt
    python predict.py --checkpoint runs/convnext_seed0/best.pt --temperature 1.7

``--temperature`` (optional) applies the temperature fitted by ``evaluate.py`` so the
shown confidences are calibrated; it never changes which class is predicted.
Reported accuracy is per slice (each image counted separately).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are saved to files; no display needed on the cluster
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torch.utils.data import DataLoader

from dataset import IDX_TO_CLASS, get_datasets
from evaluate import INK, INK2, SERIES, SURFACE, softmax, style_axes
from modules import build_model
from train import evaluate


def plot_grid(samples, idx, labels, preds, probs, n_examples: int, seed: int, path: Path) -> None:
    """Grid of test slices: mostly correct predictions plus a few errors."""
    rng = np.random.default_rng(seed)
    wrong = np.where(preds != labels)[0]
    right = np.where(preds == labels)[0]
    n_wrong = min(len(wrong), max(1, n_examples // 3))
    chosen = list(rng.choice(wrong, size=n_wrong, replace=False)) if n_wrong else []
    n_right = min(len(right), n_examples - len(chosen))
    chosen += list(rng.choice(right, size=n_right, replace=False)) if n_right else []
    rng.shuffle(chosen)

    cols = 4
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), squeeze=False, facecolor=SURFACE)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), chosen):
        s = samples[int(idx[i])]
        ok = preds[i] == labels[i]
        ax.imshow(Image.open(s.path).convert("L"), cmap="gray")
        ax.set_title(
            f"true {IDX_TO_CLASS[int(labels[i])]}, predicted {IDX_TO_CLASS[int(preds[i])]}  {'(correct)' if ok else '(WRONG)'}\n"
            f"confidence {probs[i].max():.2f}, p(AD) {probs[i, 1]:.2f}\npatient {s.patient}, slice {s.slice_idx}",
            fontsize=8, color=INK,
        )
        for spine in ax.spines.values():  # a frame marks errors in addition to the text
            spine.set_visible(not ok)
            spine.set_edgecolor(SERIES[1])
            spine.set_linewidth(3)
        ax.axis("on" if not ok else "off")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Test-set examples (a random mix, including some errors)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confidence(conf: np.ndarray, correct: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style_axes(ax)
    bins = np.linspace(0.5, 1.0, 11)
    ax.hist([conf[correct], conf[~correct]], bins=bins, color=[SERIES[0], SERIES[1]], edgecolor=SURFACE,
            linewidth=1.5, label=[f"correct ({correct.sum()})", f"incorrect ({(~correct).sum()})"])
    ax.set(title="Confidence of correct vs incorrect predictions", xlabel="confidence (probability of predicted class)",
           ylabel="number of test slices")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, SERIES[0]])  # one hue, light to dark
    fig, ax = plt.subplots(figsize=(4.8, 4.4), facecolor=SURFACE)
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            rate = cm[r, c] / cm[r].sum()
            ax.text(c, r, f"{cm[r, c]}\n{rate:.1%} of true {IDX_TO_CLASS[r]}", ha="center", va="center",
                    color="white" if rate > 0.55 else INK, fontsize=10)
    ax.set_xticks([0, 1], [f"predicted {IDX_TO_CLASS[0]}", f"predicted {IDX_TO_CLASS[1]}"], color=INK2)
    ax.set_yticks([0, 1], [f"true {IDX_TO_CLASS[0]}", f"true {IDX_TO_CLASS[1]}"], color=INK2)
    ax.set_title("Confusion matrix (test, per slice)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify the ADNI test set with a trained model and visualise results")
    ap.add_argument("--checkpoint", required=True, help="best.pt written by train.py")
    ap.add_argument("--root", default=None, help="dataset root (default: the one stored in the checkpoint)")
    ap.add_argument("--out-dir", default=None, help="default: predictions/<run directory name>")
    ap.add_argument("--temperature", type=float, default=1.0, help="temperature from evaluate.py for calibrated confidences")
    ap.add_argument("--num-examples", type=int, default=12, help="slices shown in the example grid")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="which examples are drawn for the grid")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    a = ckpt["args"]
    out = Path(args.out_dir or f"predictions/{ckpt_path.parent.name}")
    out.mkdir(parents=True, exist_ok=True)

    _, _, test_ds = get_datasets(
        args.root or a["root"], a["val_fraction"], a["split_seed"], a["img_size"],
        group_by=a.get("group_by", "scan"), meta_path=a.get("meta"),
        drop_test_overlap=a.get("drop_test_overlap", False),
    )
    model = build_model(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=args.num_workers)
    res = evaluate(model, loader, device)

    labels, idx = res["labels"].numpy(), res["idx"].numpy()
    probs = softmax(res["logits"].numpy(), args.temperature)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = preds == labels
    cm = np.array([[int(((labels == t) & (preds == p)).sum()) for p in (0, 1)] for t in (0, 1)])

    print(f"model: {ckpt['model']} | best epoch {ckpt['epoch']} | {len(labels)} test slices | temperature {args.temperature:.2f}")
    print(f"accuracy {correct.mean():.4f} | AUROC {res['auroc']:.4f} | test loss {res['loss']:.4f}")
    for c in (0, 1):
        print(f"  recall {IDX_TO_CLASS[c]}: {cm[c, c] / cm[c].sum():.4f} ({cm[c, c]}/{cm[c].sum()})")
    print(f"mean confidence: correct {conf[correct].mean():.3f}"
          + (f", incorrect {conf[~correct].mean():.3f}" if (~correct).any() else ""))

    plot_grid(test_ds.samples, idx, labels, preds, probs, args.num_examples, args.seed, out / "predictions_grid.png")
    plot_confidence(conf, correct, out / "confidence_hist.png")
    plot_confusion(cm, out / "confusion_matrix.png")
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
