"""Compare trained models: metrics, calibration, reject option and failure-case evidence.

Reads the per-slice logits written by ``train.py`` (``test_predictions.csv`` and
``val_predictions.csv`` in each run directory) and produces, in ``--out-dir``:

    summary.md             all tables (paste into the README)
    metrics.json           the same numbers, machine readable
    reliability.png        reliability diagrams, raw vs temperature-scaled
    risk_coverage.png      selective risk vs coverage (reject option)
    slice_error.png        error rate by slice index for the focus model
    failure_cases.png      the selected failure cases, with images
    failure_cases.csv      the selected failure cases with their statistics

Optional extras (all off unless requested):

    --ensemble NAME RUN [RUN ...]   average the logits of several runs (e.g. three seeds)
                                    and score the average as one more model; also reports
                                    the mean +/- std of the individual runs. Repeatable.
    --intensity-baseline            logistic regression on three global image statistics,
                                    as a sanity baseline for shortcut cues (needs the dataset)
    --bootstrap B                   patient-level bootstrap confidence intervals for
                                    accuracy and AUROC, plus paired differences against
                                    --reference (default: the focus model)

Protocol notes
--------------
* Everything is per slice (each image counted separately), as recommended for a
  2D model. Patient-level accuracy is reported as a supplementary figure only.
* Anything that is *fitted* (temperature, reject threshold) is fitted on the
  VALIDATION logits and only then applied to the test logits, so the test set
  is never used to tune anything.
* If a run has no ``val_predictions.csv`` (older runs), it is regenerated from
  ``best.pt``; this needs the dataset and torch, so run on Rangpur.
* This script reports evidence for the failure-case analysis. Deciding *why*
  the model failed is your job: look at the images and the statistics, then
  argue it in the README.

Example:
    python evaluate.py --runs runs/mlp_seed0 runs/cnn_seed0 runs/convnext_seed0 \
        --names MLP CNN ConvNeXt --focus ConvNeXt

    python evaluate.py --runs runs/mlp_seed0 runs/cnn_seed0 runs/convnext_seed0 \
        --names MLP CNN ConvNeXt --focus ConvNeXt \
        --ensemble "CNN x3" runs/cnn_seed0 runs/cnn_seed1 runs/cnn_seed2 \
        --ensemble "ConvNeXt x3" runs/convnext_seed0 runs/convnext_seed1 runs/convnext_seed2 \
        --intensity-baseline --bootstrap 1000
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

CLASS_NAMES = {0: "NC", 1: "AD"}

# Plot style: first three slots of the reference categorical palette (validated
# for all-pairs separation), light surface, recessive grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "path": np.array([r["path"] for r in rows]),
        "patient": np.array([r["patient"] for r in rows]),
        "slice_idx": np.array([int(r["slice_idx"]) for r in rows]),
        "label": np.array([int(r["label"]) for r in rows]),
        "logits": np.array([[float(r["logit_nc"]), float(r["logit_ad"])] for r in rows]),
    }


def ensure_val_predictions(run_dir: Path, root_override: Optional[str]) -> Path:
    """Return the validation-logit CSV, regenerating it from best.pt if absent."""
    out = run_dir / "val_predictions.csv"
    if out.exists():
        return out
    print(f"[{run_dir.name}] val_predictions.csv missing, regenerating from best.pt ...")
    import torch
    from torch.utils.data import DataLoader

    from dataset import get_datasets
    from modules import build_model
    from train import evaluate, save_test_predictions

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    a = ckpt["args"]
    # Rebuild the exact split this model was trained on (runs from before the
    # subject-level fix have no group_by entry and used the scan-level split).
    _, val_ds, _ = get_datasets(
        root_override or a["root"], a["val_fraction"], a["split_seed"], a["img_size"],
        group_by=a.get("group_by", "scan"), meta_path=a.get("meta"),
        drop_test_overlap=a.get("drop_test_overlap", False),
        allow_test_overlap=a.get("allow_test_overlap", False),
    )
    model = build_model(ckpt["model"], **ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=a.get("num_workers", 4))
    save_test_predictions(evaluate(model, loader, device), val_ds.samples, out)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single scalar T minimising validation NLL (post-hoc temperature scaling)."""
    res = minimize_scalar(lambda lt: nll(logits, labels, float(np.exp(lt))), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> Dict:
    probs = softmax(logits)
    preds = probs.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(labels, preds, labels=[0, 1]).ravel())
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        "macro_f1": float(f.mean()),
        "per_class": {CLASS_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c])} for c in (0, 1)},
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "brier": float(np.mean((probs[:, 1] - labels) ** 2)),
    }


def patient_level_accuracy(data: Dict, logits: np.ndarray) -> Dict:
    """Supplementary: average each patient's slice probabilities, then classify."""
    _, inv = np.unique(data["patient"], return_inverse=True)
    n = inv.max() + 1
    p_ad = softmax(logits)[:, 1]
    mean_p = np.bincount(inv, weights=p_ad, minlength=n) / np.bincount(inv, minlength=n)
    label = np.zeros(n, dtype=int)
    label[inv] = data["label"]
    return {"patients": int(n), "accuracy": float(((mean_p > 0.5).astype(int) == label).mean())}


def calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Reliability bins over confidence in [0.5, 1] (two classes => confidence >= 0.5)."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            c, a = float(conf[m].mean()), float(correct[m].mean())
            bins.append((c, a, int(m.sum())))
            ece += m.mean() * abs(a - c)
    return float(ece), bins


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    """Selective risk (error rate among accepted slices) as coverage grows."""
    order = np.argsort(-conf, kind="stable")
    errors = (~correct[order]).astype(float)
    k = np.arange(1, len(conf) + 1)
    return k / len(conf), np.cumsum(errors) / k, conf[order]


def pick_threshold(conf_val: np.ndarray, correct_val: np.ndarray, target_risk: float) -> Optional[float]:
    """Lowest confidence threshold whose VALIDATION selective risk is <= target."""
    _, risk, sorted_conf = risk_coverage(conf_val, correct_val)
    ok = np.where(risk <= target_risk)[0]
    return float(sorted_conf[ok.max()]) if len(ok) else None


def apply_threshold(conf: np.ndarray, correct: np.ndarray, thr: Optional[float]) -> Dict:
    if thr is None:
        return {"coverage": 0.0, "risk": None}
    mask = conf >= thr
    return {"coverage": float(mask.mean()), "risk": float((~correct[mask]).mean()) if mask.any() else None}


# --------------------------------------------------------------------------
# Failure-case evidence
# --------------------------------------------------------------------------
def image_stats(path: str) -> Dict[str, float]:
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return {"mean_intensity": float(img.mean()), "contrast_std": float(img.std()), "foreground_frac": float((img > 0.08).mean())}


def select_failures(data: Dict, probs: np.ndarray, n_cases: int) -> List[Dict]:
    """Pick representative errors: confident FN, confident FP, one borderline, then fill.

    Order of preference: top false negative, top false positive, the error
    closest to the decision boundary, second false negative, second false positive.
    """
    labels, preds = data["label"], probs.argmax(axis=1)
    conf = probs.max(axis=1)
    err = np.where(preds != labels)[0]
    if len(err) == 0:
        return []
    fn = err[labels[err] == 1][np.argsort(-conf[err[labels[err] == 1]])]
    fp = err[labels[err] == 0][np.argsort(-conf[err[labels[err] == 0]])]
    border = err[np.argsort(np.abs(probs[err, 1] - 0.5))]
    wanted = [
        (fn[:1], "confident false negative (AD predicted NC)"),
        (fp[:1], "confident false positive (NC predicted AD)"),
        (border[:1], "borderline error (probability near 0.5)"),
        (fn[1:2], "confident false negative (AD predicted NC)"),
        (fp[1:2], "confident false positive (NC predicted AD)"),
    ]
    chosen, seen = [], set()
    for idxs, cat in wanted:
        for i in idxs:
            if int(i) not in seen and len(chosen) < n_cases:
                chosen.append({"index": int(i), "category": cat})
                seen.add(int(i))
    for i in err[np.argsort(-conf[err])]:  # fill with the next most confident errors
        if len(chosen) >= n_cases:
            break
        if int(i) not in seen:
            chosen.append({"index": int(i), "category": "confident error"})
            seen.add(int(i))
    return chosen


def patient_error_rate(data: Dict, preds: np.ndarray) -> Dict[str, float]:
    wrong = preds != data["label"]
    return {p: float(wrong[data["patient"] == p].mean()) for p in np.unique(data["patient"])}


# --------------------------------------------------------------------------
# Optional extras: ensembles, intensity baseline, patient-level bootstrap
# --------------------------------------------------------------------------
_RESOURCE_KEYS = ("trainable_params", "peak_train_vram_mb", "latency_ms_batch1",
                  "throughput_img_per_s_batch64", "train_seconds")


def ensemble_resources(parts: List[Dict]) -> Dict:
    """Cost of running every member: parameters, latency and training time add up."""
    if not all(all(k in r and r[k] is not None for k in _RESOURCE_KEYS) for r in parts):
        return {}
    return {
        "trainable_params": int(sum(r["trainable_params"] for r in parts)),
        "peak_train_vram_mb": max(r["peak_train_vram_mb"] for r in parts),  # members are trained one at a time
        "latency_ms_batch1": sum(r["latency_ms_batch1"] for r in parts),
        "throughput_img_per_s_batch64": 1.0 / sum(1.0 / r["throughput_img_per_s_batch64"] for r in parts),
        "train_seconds": sum(r["train_seconds"] for r in parts),
        "best_epoch": "mixed",
    }


def make_ensemble(members: List[Dict]):
    """Average the logits of runs that were scored on identical test and validation rows.

    ``members`` holds dicts with keys ``test``, ``val`` and ``res``. The runs must
    share the data split (same ``--split-seed`` and data flags); otherwise the rows
    differ and this raises instead of silently mixing up slices.
    """
    t0, v0 = members[0]["test"], members[0]["val"]
    for m in members[1:]:
        if not (np.array_equal(m["test"]["path"], t0["path"]) and np.array_equal(m["val"]["path"], v0["path"])):
            raise SystemExit("ensemble members were not scored on the same test/validation rows "
                             "(different split or data flags); refusing to average them")
    test = dict(t0, logits=np.mean([m["test"]["logits"] for m in members], axis=0))
    val = dict(v0, logits=np.mean([m["val"]["logits"] for m in members], axis=0))
    return test, val, ensemble_resources([m["res"] for m in members])


def fit_intensity_baseline(train_paths, train_labels, val_paths, test_paths):
    """Logistic regression on global image statistics (mean intensity, contrast, foreground).

    A sanity baseline: if three numbers that ignore anatomy already separate AD from NC,
    a deep model can reach part of its accuracy through that shortcut. Returns two-class
    logits (val, test) shaped like a network's output, so every table treats it as a model.
    """
    from sklearn.preprocessing import StandardScaler

    def feats(paths):
        s = [image_stats(str(p)) for p in paths]
        return np.array([[d["mean_intensity"], d["contrast_std"], d["foreground_frac"]] for d in s])

    scaler = StandardScaler().fit(feats(train_paths))
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(feats(train_paths)), np.asarray(train_labels))

    def logits(paths):
        z = clf.decision_function(scaler.transform(feats(paths)))
        return np.stack([-z / 2, z / 2], axis=1)  # softmax of this is sigmoid(z)

    return logits(val_paths), logits(test_paths)


def train_paths_from_run(run_dir: Path, root_override: Optional[str]):
    """Training-set image paths and labels of the split that ``run_dir`` was trained on."""
    import torch

    from dataset import get_datasets

    a = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["args"]
    train_ds, _, _ = get_datasets(
        root_override or a["root"], a["val_fraction"], a["split_seed"], a["img_size"],
        group_by=a.get("group_by", "scan"), meta_path=a.get("meta"),
        drop_test_overlap=a.get("drop_test_overlap", False),
        allow_test_overlap=a.get("allow_test_overlap", False),
    )
    return [s.path for s in train_ds.samples], [s.label for s in train_ds.samples]


def patient_bootstrap(tests: Dict[str, Dict], names: List[str], reference: str, n_boot: int, seed: int = 0) -> Dict:
    """Patient-level (cluster) bootstrap of accuracy and AUROC, with paired differences.

    Slices from one patient are strongly correlated (many patients are wrong on almost
    every slice), so resampling single slices would give confidence intervals that are
    far too narrow. Here whole patients are resampled with replacement, and every model
    is scored on the same resamples so differences between models are paired.
    """
    base = tests[names[0]]
    labels = base["label"]
    _, inv = np.unique(base["patient"], return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(inv.max() + 1)]
    p_ad = {n: softmax(tests[n]["logits"])[:, 1] for n in names}
    correct = {n: tests[n]["logits"].argmax(axis=1) == labels for n in names}
    rng = np.random.default_rng(seed)
    acc = {n: np.empty(n_boot) for n in names}
    auc = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        y = labels[idx]
        for n in names:
            acc[n][b] = correct[n][idx].mean()
            auc[n][b] = roc_auc_score(y, p_ad[n][idx])

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    def pval(d):  # two-sided bootstrap p-value for "difference is zero"
        return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))

    out = {"n_boot": n_boot, "n_patients": len(members), "reference": reference, "models": {}, "paired": {}}
    for n in names:
        out["models"][n] = {"accuracy_ci": ci(acc[n]), "auroc_ci": ci(auc[n])}
        if n != reference:
            da, du = acc[n] - acc[reference], auc[n] - auc[reference]
            out["paired"][n] = {"d_accuracy": float(da.mean()), "d_accuracy_ci": ci(da), "d_accuracy_p": pval(da),
                                "d_auroc": float(du.mean()), "d_auroc_ci": ci(du), "d_auroc_p": pval(du)}
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_reliability(results: Dict, path: Path) -> None:
    names = list(results)
    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.4 * nrows), squeeze=False, facecolor=SURFACE)
    for ax in axes.ravel()[len(names):]:  # hide unused panels
        ax.axis("off")
    for ax, (i, name) in zip(axes.ravel(), enumerate(names)):
        r = results[name]
        style_axes(ax)
        ax.plot([0.5, 1], [0.5, 1], color=INK2, linestyle=":", linewidth=1, label="perfect calibration")
        for key, ls, mfc, lab in (("raw", "-", SERIES[i % 3], "raw"), ("ts", "--", SURFACE, "temperature-scaled")):
            bins = r["bins_" + key]
            if bins:
                ax.plot([b[0] for b in bins], [b[1] for b in bins], color=SERIES[i % 3], linestyle=ls, linewidth=2,
                        marker="o", markersize=6, markerfacecolor=mfc, label=f"{lab} (ECE {r['ece_' + key]:.3f})")
        ax.set(title=f"{name} (T = {r['temperature']:.2f})", xlabel="confidence", ylabel="accuracy",
               xlim=(0.5, 1.0), ylim=(0.0, 1.02))
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_risk_coverage(results: Dict, target_risk: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    style_axes(ax)
    for i, (name, r) in enumerate(results.items()):
        # colour repeats every 3 models, so the line style also changes to keep lines distinguishable
        ax.plot(r["rc_coverage"], r["rc_risk"], color=SERIES[i % 3], linewidth=2,
                linestyle=("-", "--", ":")[(i // 3) % 3], label=name)
        op = r["reject_test"]
        if op["risk"] is not None:
            ax.plot(op["coverage"], op["risk"], marker="o", markersize=9, color=SERIES[i % 3],
                    markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    ax.axhline(target_risk, color=INK2, linestyle=":", linewidth=1)
    ax.text(0.02, target_risk, f" target risk {target_risk:.0%}", color=INK2, va="bottom", fontsize=8)
    ax.set(title="Reject option: error rate among accepted slices", xlabel="coverage (fraction of test slices accepted)",
           ylabel="selective risk (error rate)", xlim=(0, 1.0))
    ax.legend(frameon=False, title="dots: validation-chosen threshold, applied to test", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_slice_error(data: Dict, preds: np.ndarray, name: str, path: Path) -> List[Dict]:
    rows = []
    for s in np.unique(data["slice_idx"]):
        m = data["slice_idx"] == s
        ad, nc = m & (data["label"] == 1), m & (data["label"] == 0)
        rows.append({
            "slice_idx": int(s), "n": int(m.sum()), "error_rate": float((preds[m] != data["label"][m]).mean()),
            "fn_rate": float((preds[ad] == 0).mean()) if ad.any() else float("nan"),
            "fp_rate": float((preds[nc] == 1).mean()) if nc.any() else float("nan"),
        })
    fig, ax = plt.subplots(figsize=(7.0, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = [r["slice_idx"] for r in rows]
    ax.plot(x, [r["fn_rate"] for r in rows], color=SERIES[0], linewidth=2, marker="o", markersize=5,
            label="AD slices missed (false-negative rate)")
    ax.plot(x, [r["fp_rate"] for r in rows], color=SERIES[1], linewidth=2, marker="o", markersize=5,
            label="NC slices flagged (false-positive rate)")
    ax.set(title=f"{name}: error rate by slice index", xlabel="slice index", ylabel="error rate", ylim=(0, 1))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return rows


def plot_failures(cases: List[Dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(cases), figsize=(3.3 * len(cases), 4.6), squeeze=False, facecolor=SURFACE)
    for ax, c in zip(axes[0], cases):
        ax.imshow(Image.open(c["path"]).convert("L"), cmap="gray")
        ax.axis("off")
        ax.set_title(f"{c['category']}\ntrue {c['true']} -> pred {c['pred']} (conf {c['confidence']:.2f})\n"
                     f"patient {c['patient']}, slice {c['slice_idx']}\n"
                     f"{c['patient_wrong_slices']}/{c['patient_slices']} slices of this patient wrong",
                     fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def fmt(x, nd=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate and compare trained ADNI models")
    ap.add_argument("--runs", nargs="+", required=True, help="run directories written by train.py")
    ap.add_argument("--names", nargs="+", default=None, help="display names (default: directory names)")
    ap.add_argument("--focus", default=None, help="model for failure cases (default: last run)")
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--target-risk", type=float, default=0.10, help="reject option: max error rate among accepted slices")
    ap.add_argument("--n-cases", type=int, default=5, help="number of failure cases (3-5)")
    ap.add_argument("--root", default=None, help="dataset root override when regenerating val logits")
    ap.add_argument("--ensemble", nargs="+", action="append", default=[], metavar=("NAME", "RUN"),
                    help="NAME RUN [RUN ...]: score the logit average of these runs as one more model (repeatable)")
    ap.add_argument("--intensity-baseline", action="store_true",
                    help="add a logistic-regression baseline on global image statistics (needs the dataset)")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="B",
                    help="patient-level bootstrap resamples for confidence intervals (0 = off; 1000 is typical)")
    ap.add_argument("--reference", default=None, help="model the paired bootstrap differences are taken against (default: focus)")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    run_names = args.names or [d.name for d in run_dirs]
    if len(run_names) != len(run_dirs):
        raise SystemExit("--names must match --runs in length")
    names = list(run_names)
    focus = args.focus or run_names[-1]  # chosen before ensembles / baseline are appended
    if focus not in names:
        raise SystemExit(f"--focus must be one of {names}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, Dict] = {}  # run directory -> {"test", "val", "res"}; shared by --runs and --ensemble

    def load_run(d: Path) -> Dict:
        key = str(d)
        if key not in cache:
            cache[key] = {
                "test": load_predictions(d / "test_predictions.csv"),
                "val": load_predictions(ensure_val_predictions(d, args.root)),
                "res": json.load(open(d / "results.json")) if (d / "results.json").exists() else {},
            }
        return cache[key]

    tests, vals, resources = {}, {}, {}
    for name, d in zip(names, run_dirs):
        r = load_run(d)
        tests[name], vals[name], resources[name] = r["test"], r["val"], r["res"]

    seed_stats: Dict[str, Dict] = {}  # per ensemble: metrics of the individual members
    for spec in args.ensemble:
        if len(spec) < 3:
            raise SystemExit("--ensemble needs a NAME and at least two runs")
        ens_name, member_dirs = spec[0], [Path(p) for p in spec[1:]]
        if ens_name in names:
            raise SystemExit(f"ensemble name '{ens_name}' clashes with another model name")
        members = [load_run(d) for d in member_dirs]
        tests[ens_name], vals[ens_name], resources[ens_name] = make_ensemble(members)
        per = [classification_metrics(m["test"]["label"], m["test"]["logits"]) for m in members]
        seed_stats[ens_name] = {
            "runs": [str(d) for d in member_dirs],
            "accuracy": [p["accuracy"] for p in per], "auroc": [p["auroc"] for p in per],
        }
        names.append(ens_name)

    if args.intensity_baseline:
        print("fitting the intensity-only baseline (reads every training image once) ...")
        first_t, first_v = tests[run_names[0]], vals[run_names[0]]
        tr_paths, tr_labels = train_paths_from_run(run_dirs[0], args.root)
        v_logits, t_logits = fit_intensity_baseline(tr_paths, tr_labels, first_v["path"], first_t["path"])
        tests["Intensity-only"] = dict(first_t, logits=t_logits)
        vals["Intensity-only"] = dict(first_v, logits=v_logits)
        resources["Intensity-only"] = {}
        names.append("Intensity-only")

    first = tests[names[0]]
    for n in names[1:]:  # all models must be scored on the same rows in the same order
        assert np.array_equal(tests[n]["path"], first["path"]), f"{n}: test rows differ from {names[0]}"

    results: Dict[str, Dict] = {}
    for name in names:
        t, v = tests[name], vals[name]
        temp = fit_temperature(v["logits"], v["label"])  # fitted on validation only
        probs_raw, probs_ts = softmax(t["logits"]), softmax(t["logits"], temp)
        ece_raw, bins_raw = calibration_bins(probs_raw, t["label"])
        ece_ts, bins_ts = calibration_bins(probs_ts, t["label"])
        cov, risk, _ = risk_coverage(probs_ts.max(1), probs_ts.argmax(1) == t["label"])
        thr = pick_threshold(softmax(v["logits"], temp).max(1), softmax(v["logits"], temp).argmax(1) == v["label"], args.target_risk)
        results[name] = {
            "test": classification_metrics(t["label"], t["logits"]),
            "patient_level": patient_level_accuracy(t, t["logits"]),
            "temperature": temp, "ece_raw": ece_raw, "ece_ts": ece_ts, "bins_raw": bins_raw, "bins_ts": bins_ts,
            "nll_raw": nll(t["logits"], t["label"]), "nll_ts": nll(t["logits"], t["label"], temp),
            "rc_coverage": cov, "rc_risk": risk, "aurc": float(risk.mean()),
            "risk_at_80_coverage": float(risk[int(0.8 * len(risk)) - 1]),
            "reject_threshold": thr,
            "reject_test": apply_threshold(probs_ts.max(1), probs_ts.argmax(1) == t["label"], thr),
        }

    plot_reliability(results, out / "reliability.png")
    plot_risk_coverage(results, args.target_risk, out / "risk_coverage.png")

    boot = None
    if args.bootstrap > 0:
        reference = args.reference or focus
        if reference not in names:
            raise SystemExit(f"--reference must be one of {names}")
        print(f"patient-level bootstrap: {args.bootstrap} resamples x {len(names)} models ...")
        boot = patient_bootstrap(tests, names, reference, args.bootstrap)

    # ---- failure cases for the focus model (calibrated confidences) ----
    ft = tests[focus]
    probs_f = softmax(ft["logits"], results[focus]["temperature"])
    preds_f = probs_f.argmax(1)
    slice_rows = plot_slice_error(ft, preds_f, focus, out / "slice_error.png")
    per_patient = patient_error_rate(ft, preds_f)
    wrong_total = int((preds_f != ft["label"]).sum())
    rates = np.array(list(per_patient.values()))
    n_slices_per_patient = {p: int((ft["patient"] == p).sum()) for p in per_patient}
    heavy = {p for p, r in per_patient.items() if r >= 0.5}
    heavy_errors = sum(int(round(per_patient[p] * n_slices_per_patient[p])) for p in heavy)
    systematic = {
        "errors": wrong_total, "patients": len(per_patient), "patients_with_any_error": int((rates > 0).sum()),
        "patients_mostly_wrong": len(heavy), "share_of_errors_in_mostly_wrong_patients": heavy_errors / max(wrong_total, 1),
        "overlap_with_other_models": {
            n: float((preds_f[preds_f != ft["label"]] == softmax(tests[n]["logits"]).argmax(1)[preds_f != ft["label"]]).mean())
            for n in names if n != focus and wrong_total
        },
    }

    cases = select_failures(ft, probs_f, max(3, min(5, args.n_cases)))
    rng = np.random.default_rng(0)
    ref = {}
    for cls in (0, 1):  # reference image statistics from correctly classified slices of each class
        ok = np.where((preds_f == ft["label"]) & (ft["label"] == cls))[0]
        pick = rng.choice(ok, size=min(300, len(ok)), replace=False) if len(ok) else []
        stats = [image_stats(ft["path"][i]) for i in pick]
        ref[cls] = {k: (float(np.mean([s[k] for s in stats])), float(np.std([s[k] for s in stats]))) for k in
                    ("mean_intensity", "contrast_std", "foreground_frac")} if stats else {}
    for c in cases:
        i = c["index"]
        c.update({
            "path": str(ft["path"][i]), "patient": str(ft["patient"][i]), "slice_idx": int(ft["slice_idx"][i]),
            "true": CLASS_NAMES[int(ft["label"][i])], "pred": CLASS_NAMES[int(preds_f[i])],
            "confidence": float(probs_f[i].max()), "patient_slices": n_slices_per_patient[str(ft["patient"][i])],
            "patient_wrong_slices": int(round(per_patient[str(ft["patient"][i])] * n_slices_per_patient[str(ft["patient"][i])])),
            **image_stats(str(ft["path"][i])),
        })
    if cases:
        plot_failures(cases, out / "failure_cases.png")
        keys = ["category", "path", "patient", "slice_idx", "true", "pred", "confidence", "patient_slices",
                "patient_wrong_slices", "mean_intensity", "contrast_std", "foreground_frac"]
        with open(out / "failure_cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(cases)

    # ---- summary.md ----
    L = ["# Evaluation summary", "", f"Per-slice results on the held-out test set. Focus model for failure cases: **{focus}**.", "",
         "## Classification (test, per slice)", "",
         "| Model | Accuracy | Balanced acc. | Macro-F1 | AUROC | AD recall (sens.) | NC recall (spec.) | AD precision | Patient-level acc. (suppl.) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        m = r["test"]
        L.append(f"| {n} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['macro_f1'])} | {fmt(m['auroc'])} | "
                 f"{fmt(m['per_class']['AD']['recall'])} | {fmt(m['per_class']['NC']['recall'])} | "
                 f"{fmt(m['per_class']['AD']['precision'])} | {fmt(r['patient_level']['accuracy'])} |")
    L += ["", "## Confusion matrices (test; rows = true, columns = predicted)", ""]
    for n, r in results.items():
        c = r["test"]["confusion"]
        L.append(f"- **{n}**: NC->NC {c['tn']}, NC->AD {c['fp']}, AD->NC {c['fn']}, AD->AD {c['tp']}")
    L += ["", "## Calibration (temperature fitted on validation)", "",
          "| Model | T | ECE raw | ECE scaled | NLL raw | NLL scaled | Brier |", "|---|---|---|---|---|---|---|"]
    for n, r in results.items():
        L.append(f"| {n} | {r['temperature']:.2f} | {fmt(r['ece_raw'])} | {fmt(r['ece_ts'])} | {fmt(r['nll_raw'])} | "
                 f"{fmt(r['nll_ts'])} | {fmt(r['test']['brier'])} |")
    L += ["", f"## Reject option (threshold chosen on validation for <= {args.target_risk:.0%} error among accepted slices)", "",
          "| Model | Threshold | Test coverage | Test error among accepted | Error at 80% coverage | AURC |", "|---|---|---|---|---|---|"]
    for n, r in results.items():
        thr = "none reachable" if r["reject_threshold"] is None else f"{r['reject_threshold']:.3f}"
        L.append(f"| {n} | {thr} | {fmt(r['reject_test']['coverage'], pct=True)} | {fmt(r['reject_test']['risk'], pct=True)} | "
                 f"{fmt(r['risk_at_80_coverage'], pct=True)} | {fmt(r['aurc'])} |")
    if seed_stats:
        L += ["", "## Seed variability and ensembles", "",
              "Mean +/- std over the individual runs (test, per slice; std is the sample std over runs), next to the "
              "logit-average ensemble of the same runs. Differences smaller than the std are not evidence of a real effect.", "",
              "| Group | Runs | Accuracy (runs) | AUROC (runs) | Accuracy (ensemble) | AUROC (ensemble) |", "|---|---|---|---|---|---|"]
        for n, s in seed_stats.items():
            acc, auc = np.array(s["accuracy"]), np.array(s["auroc"])
            sd = (lambda x: x.std(ddof=1) if len(x) > 1 else float("nan"))
            m = results[n]["test"]
            L.append(f"| {n} | {len(acc)} | {acc.mean():.3f} +/- {sd(acc):.3f} | {auc.mean():.3f} +/- {sd(auc):.3f} | "
                     f"{fmt(m['accuracy'])} | {fmt(m['auroc'])} |")
        L += [""] + [f"- {n}: " + ", ".join(s["runs"]) for n, s in seed_stats.items()]
    if boot is not None:
        L += ["", f"## Uncertainty: patient-level bootstrap ({boot['n_boot']} resamples of {boot['n_patients']} test patients)", "",
              "Whole patients are resampled, not single slices, because errors cluster by patient; slice-level intervals would be far too narrow. "
              "Intervals are 95% percentile intervals.", "",
              "| Model | Accuracy [95% CI] | AUROC [95% CI] |", "|---|---|---|"]
        for n in names:
            m, b = results[n]["test"], boot["models"][n]
            L.append(f"| {n} | {m['accuracy']:.3f} [{b['accuracy_ci'][0]:.3f}, {b['accuracy_ci'][1]:.3f}] | "
                     f"{m['auroc']:.3f} [{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}] |")
        L += ["", f"Paired difference against **{boot['reference']}** (model minus reference, same resamples; "
                  "p = two-sided bootstrap p-value for a difference of zero, not corrected for the number of comparisons):", "",
              "| Model | Δ accuracy [95% CI] | p | Δ AUROC [95% CI] | p |", "|---|---|---|---|---|"]
        for n, d in boot["paired"].items():
            L.append(f"| {n} | {d['d_accuracy']:+.3f} [{d['d_accuracy_ci'][0]:+.3f}, {d['d_accuracy_ci'][1]:+.3f}] | {d['d_accuracy_p']:.3f} | "
                     f"{d['d_auroc']:+.3f} [{d['d_auroc_ci'][0]:+.3f}, {d['d_auroc_ci'][1]:+.3f}] | {d['d_auroc_p']:.3f} |")
    L += ["", "## Resources", "", "| Model | Params | Peak train VRAM (MB) | Latency (ms/img, batch 1) | Throughput (img/s, batch 64) | Train time (min) | Best epoch |",
          "|---|---|---|---|---|---|---|"]
    for n in names:
        s = resources[n]
        L.append(f"| {n} | {s.get('trainable_params', 'n/a'):,} | {fmt(s.get('peak_train_vram_mb'), 0)} | {fmt(s.get('latency_ms_batch1'), 2)} | "
                 f"{fmt(s.get('throughput_img_per_s_batch64'), 0)} | {fmt((s.get('train_seconds') or float('nan')) / 60, 1)} | {s.get('best_epoch', 'n/a')} |"
                 if isinstance(s.get("trainable_params"), int) else f"| {n} | n/a | n/a | n/a | n/a | n/a | n/a |")
    L += ["", f"## Is the {focus} failure systematic? (evidence)", "",
          f"- {systematic['errors']} of {len(ft['label'])} test slices are misclassified, spread over "
          f"{systematic['patients_with_any_error']} of {systematic['patients']} patients.",
          f"- {systematic['patients_mostly_wrong']} patients have >= 50% of their slices wrong; they account for "
          f"{fmt(systematic['share_of_errors_in_mostly_wrong_patients'], pct=True)} of all errors "
          "(high share = patient-level/systematic failure, low share = scattered slice-level errors).",
          "- Share of this model's errors that other models also make (high = hard cases for any model, e.g. data or label issues; low = model-specific):"]
    L += [f"  - {n}: {fmt(v, pct=True)}" for n, v in systematic["overlap_with_other_models"].items()]
    L += ["- Error rate by slice index: see `slice_error.png`; a strong dependence on slice index points to anatomical position as a trigger.", ""]
    if cases:
        L += ["## Selected failure cases", "",
              "| # | Category | True -> pred | Conf. | Patient | Slice | Patient's wrong slices | Mean intensity | Contrast (std) | Foreground |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k, c in enumerate(cases, 1):
            L.append(f"| {k} | {c['category']} | {c['true']} -> {c['pred']} | {c['confidence']:.2f} | {c['patient']} | {c['slice_idx']} | "
                     f"{c['patient_wrong_slices']}/{c['patient_slices']} | {c['mean_intensity']:.3f} | {c['contrast_std']:.3f} | {c['foreground_frac']:.3f} |")
        L += ["", "Reference (mean +/- std over correctly classified test slices of each true class):", ""]
        for cls in (0, 1):
            if ref[cls]:
                L.append(f"- {CLASS_NAMES[cls]}: " + ", ".join(f"{k} {v[0]:.3f} +/- {v[1]:.3f}" for k, v in ref[cls].items()))
    text = "\n".join(L) + "\n"
    (out / "summary.md").write_text(text)

    def clean(o):  # make everything JSON-serialisable and small
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in ("rc_coverage", "rc_risk", "bins_raw", "bins_ts")}
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(out / "metrics.json", "w") as f:
        json.dump(clean({"models": results, "focus": focus, "systematic": systematic, "slice_error": slice_rows,
                         "failure_cases": cases, "target_risk": args.target_risk,
                         "seed_stats": seed_stats, "bootstrap": boot}), f, indent=2)
    print(text)
    print(f"figures and tables written to {out}/")


if __name__ == "__main__":
    main()
