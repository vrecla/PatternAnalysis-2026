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
# Plots
# --------------------------------------------------------------------------
def plot_reliability(results: Dict, path: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False, facecolor=SURFACE)
    for ax, (i, name) in zip(axes[0], enumerate(names)):
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
        ax.plot(r["rc_coverage"], r["rc_risk"], color=SERIES[i % 3], linewidth=2, label=name)
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
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    names = args.names or [d.name for d in run_dirs]
    if len(names) != len(run_dirs):
        raise SystemExit("--names must match --runs in length")
    focus = args.focus or names[-1]
    if focus not in names:
        raise SystemExit(f"--focus must be one of {names}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tests, vals, resources = {}, {}, {}
    for name, d in zip(names, run_dirs):
        tests[name] = load_predictions(d / "test_predictions.csv")
        vals[name] = load_predictions(ensure_val_predictions(d, args.root))
        resources[name] = json.load(open(d / "results.json")) if (d / "results.json").exists() else {}
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
                         "failure_cases": cases, "target_risk": args.target_risk}), f, indent=2)
    print(text)
    print(f"figures and tables written to {out}/")


if __name__ == "__main__":
    main()
