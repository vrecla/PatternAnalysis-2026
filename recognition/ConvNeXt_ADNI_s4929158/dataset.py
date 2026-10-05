"""Data loading for ADNI AD-vs-NC classification with subject-level splitting.

Expected on-disk layout (Rangpur: /home/groups/comp3710/ADNI)::

    meta_data_with_label.json
    AD_NC/
      train/{AD,NC}/<scanID>_<sliceIdx>.jpeg
      test/{AD,NC}/<scanID>_<sliceIdx>.jpeg

Each image is one 2D slice of a brain MRI scan. The number before the
underscore is the ADNI *image (scan) ID*, NOT a person: ``meta_data_with_label.json``
maps each scan ID to a file name that contains the ADNI *subject* ID (for
example ``068_S_0473``), and one subject can have several scans from different
visits. Splitting by scan therefore lets the same person appear in both train
and validation. We split by subject instead, and ``assert_no_leakage`` verifies it.

``group_by="scan"`` reproduces the original scan-level behaviour (used by the
first training runs); the default is ``group_by="subject"``.

Run ``python dataset.py --audit`` to print split statistics, measure how much a
scan-level split leaks, and run the leakage checks.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Class name -> integer label. NC (cognitively normal) = 0, AD = 1, so that
# "positive" means Alzheimer's disease in all precision/recall/AUROC maths.
CLASS_TO_IDX: Dict[str, int] = {"NC": 0, "AD": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

META_NAME = "meta_data_with_label.json"
_FILENAME_RE = re.compile(r"^(?P<scan>\d+)_(?P<slice>\d+)\.(jpe?g|png)$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"ADNI_(\d+_S_\d+)_")


@dataclass(frozen=True)
class Sample:
    """One slice image.

    ``patient`` is the unit used for splitting and grouping: the ADNI subject ID
    when ``group_by="subject"``, otherwise the scan ID. ``scan`` is always the
    scan (image) ID taken from the file name.
    """

    path: str
    label: int
    patient: str
    slice_idx: int
    scan: str = ""


# --------------------------------------------------------------------------
# Scanning, metadata and splitting
# --------------------------------------------------------------------------
def parse_filename(path: Path) -> Tuple[str, int]:
    """Return (scan_id, slice_idx) from a '<scan>_<slice>.jpeg' name."""
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return match.group("scan"), int(match.group("slice"))


def scan_split(root: str, split: str) -> List[Sample]:
    """List every image under ``root/split/{AD,NC}`` as a Sample (patient = scan ID)."""
    samples: List[Sample] = []
    for cls_name, label in CLASS_TO_IDX.items():
        cls_dir = Path(root) / split / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {cls_dir}")
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                continue
            scan_id, slice_idx = parse_filename(img_path)
            samples.append(Sample(str(img_path), label, scan_id, slice_idx, scan_id))
    if not samples:
        raise RuntimeError(f"No images found under {Path(root) / split}")
    return samples


def load_metadata(meta_path: Path) -> Dict[str, dict]:
    with open(meta_path) as f:
        return json.load(f)


def load_subject_map(meta_path: Path) -> Dict[str, str]:
    """Map scan ID -> ADNI subject ID, parsed from the file names in the metadata."""
    subjects: Dict[str, str] = {}
    for scan_id, entry in load_metadata(meta_path).items():
        if not isinstance(entry, dict):
            continue
        for key in ("raw", "masked", "c1"):
            match = _SUBJECT_RE.search(Path(entry.get(key, "")).name)
            if match:
                subjects[str(scan_id)] = match.group(1)
                break
    return subjects


def attach_subjects(samples: Sequence[Sample], subject_map: Dict[str, str]) -> Tuple[List[Sample], int]:
    """Replace ``patient`` with the subject ID. Returns (samples, number of unmapped scans).

    A scan missing from the metadata is treated as its own subject (a warning is
    printed by the caller) rather than silently dropped.
    """
    missing, out = set(), []
    for s in samples:
        subject = subject_map.get(s.scan)
        if subject is None:
            missing.add(s.scan)
            subject = f"unmapped_{s.scan}"
        out.append(replace(s, patient=subject))
    return out, len(missing)


def patient_level_split(
    samples: Sequence[Sample], val_fraction: float = 0.15, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples into (train, val) so no patient (subject or scan) appears in both.

    Patients are shuffled and divided *within each class* so that the AD/NC
    balance is approximately preserved. A subject whose scans carry both labels
    (a diagnosis that changed between visits) is stratified by its majority label.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    labels_by_patient: Dict[str, List[int]] = defaultdict(list)
    for s in samples:
        labels_by_patient[s.patient].append(s.label)
    by_class: Dict[int, List[str]] = defaultdict(list)
    for patient, labels in labels_by_patient.items():
        by_class[Counter(labels).most_common(1)[0][0]].append(patient)

    rng = random.Random(seed)
    val_patients = set()
    for label, patients in by_class.items():
        patients = sorted(patients)  # sort first so the shuffle is reproducible
        rng.shuffle(patients)
        n_val = max(1, round(len(patients) * val_fraction))
        val_patients.update(patients[:n_val])

    train = [s for s in samples if s.patient not in val_patients]
    val = [s for s in samples if s.patient in val_patients]
    return train, val


def patients_of(samples: Sequence[Sample]) -> set:
    """Set of patient IDs present in a list of samples."""
    return {s.patient for s in samples}


def assert_no_leakage(**splits: Sequence[Sample]) -> None:
    """Raise AssertionError if any patient appears in more than one split."""
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = patients_of(splits[names[i]]) & patients_of(splits[names[j]])
            assert not overlap, (
                f"Patient leakage between '{names[i]}' and '{names[j]}': "
                f"{len(overlap)} shared patients, e.g. {sorted(overlap)[:5]}"
            )


# --------------------------------------------------------------------------
# Transforms and Dataset
# --------------------------------------------------------------------------
def get_transforms(train: bool, img_size: int = 224, aug: str = "mild") -> Callable:
    """Image transforms. Augmentation is applied to the training split only.

    ``aug="mild"`` (default): small affine jitter and brightness/contrast change.
    Brain MRIs are roughly aligned, so large rotations or flips could change
    anatomy and hurt more than help. Left-right flips are omitted because
    hemispheric asymmetry can matter.

    ``aug="strong"``: for the overfitting seen on the subject-level split. Adds a
    random crop-and-resize, a larger affine jitter, stronger intensity jitter,
    slight blur and random erasing (to stop the model relying on one local region).
    Still no flips.
    """
    if aug not in ("mild", "strong"):
        raise ValueError("aug must be 'mild' or 'strong'")
    ops: List[Callable] = [T.Grayscale(num_output_channels=1)]
    if train and aug == "strong":
        ops += [T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1))]
    else:
        ops += [T.Resize((img_size, img_size))]
    if train and aug == "mild":
        ops += [
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    elif train and aug == "strong":
        ops += [
            T.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
        ]
    ops += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    if train and aug == "strong":
        ops += [T.RandomErasing(p=0.25, scale=(0.02, 0.12), value=0.0)]
    return T.Compose(ops)


class ADNIDataset(Dataset):
    """Map-style dataset returning (image_tensor, label, sample_index).

    The sample index is returned alongside each item so the evaluation code
    can aggregate slice predictions per patient and trace failure cases back
    to the image that caused them.
    """

    def __init__(self, samples: Sequence[Sample], transform: Optional[Callable] = None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label, idx

    def class_counts(self) -> Counter:
        return Counter(s.label for s in self.samples)


def _default_meta(root: str, meta_path: Optional[str]) -> Path:
    return Path(meta_path) if meta_path else Path(root).parent / META_NAME


def get_datasets(
    root: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    img_size: int = 224,
    group_by: str = "subject",
    meta_path: Optional[str] = None,
    drop_test_overlap: bool = False,
    aug: str = "mild",
    allow_test_overlap: bool = False,
) -> Tuple[ADNIDataset, ADNIDataset, ADNIDataset]:
    """Build leakage-checked (train, val, test) datasets.

    The official ``train`` folder is split by ``group_by`` (subject or scan) into
    train/val. The official ``test`` folder is kept as the held-out test set, but
    we verify it shares no subjects with train/val. If it does, this raises unless
    ``drop_test_overlap`` is set, which removes the overlapping subjects from the
    TRAINING data only (the test set is never altered).

    ``allow_test_overlap`` is for the ABLATION ONLY: it keeps the subjects that the
    official folders share, so the train/test leakage can be measured. Train/val
    are still separated by subject. Results from this mode must be labelled leaky.
    """
    if group_by not in ("subject", "scan"):
        raise ValueError("group_by must be 'subject' or 'scan'")
    if drop_test_overlap and allow_test_overlap:
        raise ValueError("drop_test_overlap and allow_test_overlap are mutually exclusive")
    all_train = scan_split(root, "train")
    test = scan_split(root, "test")

    if group_by == "subject":
        subject_map = load_subject_map(_default_meta(root, meta_path))
        all_train, miss_train = attach_subjects(all_train, subject_map)
        test, miss_test = attach_subjects(test, subject_map)
        if miss_train or miss_test:
            print(f"WARNING: {miss_train} train and {miss_test} test scans are missing from the metadata; "
                  "each is treated as its own subject.")

    overlap = patients_of(all_train) & patients_of(test)
    if overlap and allow_test_overlap:
        print(f"WARNING (ablation): keeping {len(overlap)} {group_by}s shared by the official train and test "
              "folders. Test results from this run are LEAKY and must be labelled as such.")
    elif overlap:
        if not drop_test_overlap:
            raise ValueError(
                f"{len(overlap)} {group_by}s appear in both the official train and test folders. "
                "Pass drop_test_overlap=True (train.py: --drop-test-overlap) to remove them from training."
            )
        before = len(all_train)
        all_train = [s for s in all_train if s.patient not in overlap]
        print(f"Removed {before - len(all_train)} training slices ({len(overlap)} {group_by}s shared with the test set).")

    train, val = patient_level_split(all_train, val_fraction, seed)
    if allow_test_overlap:
        assert_no_leakage(train=train, val=val)  # train/test overlap is deliberate in this mode
    else:
        assert_no_leakage(train=train, val=val, test=test)
    return (
        ADNIDataset(train, get_transforms(True, img_size, aug)),
        ADNIDataset(val, get_transforms(False, img_size)),
        ADNIDataset(test, get_transforms(False, img_size)),
    )


def get_dataloaders(
    root: str,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    seed: int = 42,
    img_size: int = 224,
    num_workers: int = 4,
    **split_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = get_datasets(root, val_fraction, seed, img_size, **split_kwargs)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


# --------------------------------------------------------------------------
# Audit CLI
# --------------------------------------------------------------------------
def _describe(name: str, samples: Sequence[Sample], unit: str) -> None:
    counts = Counter(s.label for s in samples)
    per_patient = Counter(s.patient for s in samples)
    slices = list(per_patient.values())
    print(
        f"{name:>6}: {len(samples):6d} slices | {len(per_patient):5d} {unit}s | "
        f"NC={counts[0]} AD={counts[1]} | slices/{unit} "
        f"min={min(slices)} max={max(slices)} mean={sum(slices) / len(slices):.1f}"
    )


def audit(root: str, val_fraction: float, seed: int, meta_path: Optional[str] = None) -> None:
    """Print split statistics, quantify scan-level leakage, and run leakage checks."""
    official_train = scan_split(root, "train")
    test = scan_split(root, "test")
    print("== Part 1: official folders, unit = scan (the ID in the file name) ==")
    _describe("train", official_train, "scan")
    _describe("test", test, "scan")
    print(f"Scans shared by official train and test: {len(patients_of(official_train) & patients_of(test))}")

    meta = _default_meta(root, meta_path)
    if not meta.exists():
        print(f"\nNo metadata file at {meta}; skipping the subject-level audit.")
        return

    print(f"\n== Part 2: subject level, using {meta} ==")
    metadata = load_metadata(meta)
    subject_map = load_subject_map(meta)
    tr_s, miss_tr = attach_subjects(official_train, subject_map)
    te_s, miss_te = attach_subjects(test, subject_map)
    print(f"Scans missing from metadata: train {miss_tr}, test {miss_te}")

    scans_of: Dict[str, set] = defaultdict(set)
    for s in tr_s + te_s:
        scans_of[s.patient].add(s.scan)
    per_subject = [len(v) for v in scans_of.values()]
    print(f"{len(scans_of)} subjects over {sum(per_subject)} scans | scans/subject min={min(per_subject)} "
          f"median={median(per_subject)} max={max(per_subject)} | subjects with >1 scan: "
          f"{sum(1 for n in per_subject if n > 1)}")

    pairs = Counter()
    for s in official_train + test:
        entry = metadata.get(s.scan)
        pairs[(IDX_TO_CLASS[s.label], entry.get("label") if isinstance(entry, dict) else "n/a")] += 1
    print("Folder class vs metadata label (slice counts): " + ", ".join(f"{k}: {v}" for k, v in sorted(pairs.items(), key=str)))

    labels_of: Dict[str, set] = defaultdict(set)
    for s in tr_s + te_s:
        labels_of[s.patient].add(s.label)
    print(f"Subjects whose scans carry both AD and NC labels: {sum(1 for v in labels_of.values() if len(v) > 1)}")

    shared = patients_of(tr_s) & patients_of(te_s)
    test_slices_shared = sum(1 for s in te_s if s.patient in shared)
    print(f"Subjects shared by official train and test: {len(shared)} "
          f"({test_slices_shared} of {len(te_s)} test slices = {100 * test_slices_shared / len(te_s):.1f}%)")

    # How much did the original scan-level split leak into validation?
    old_train, old_val = patient_level_split(official_train, val_fraction, seed)
    old_train_s, _ = attach_subjects(old_train, subject_map)
    old_val_s, _ = attach_subjects(old_val, subject_map)
    leaked = patients_of(old_train_s) & patients_of(old_val_s)
    leaked_slices = sum(1 for s in old_val_s if s.patient in leaked)
    print(f"\nScan-level split (used by the first runs): {len(leaked)} of {len(patients_of(old_val_s))} validation "
          f"subjects also appear in training ({leaked_slices} of {len(old_val_s)} validation slices = "
          f"{100 * leaked_slices / len(old_val_s):.1f}% leaked)")

    train, val = patient_level_split(tr_s, val_fraction, seed)
    print("\n== After subject-level train/val split ==")
    _describe("train", train, "subject")
    _describe("val", val, "subject")
    _describe("test", te_s, "subject")
    assert_no_leakage(train=train, val=val)
    print("train/val subject leakage check: PASSED")
    if shared:
        print(f"WARNING: {len(shared)} subjects are in both official train and test. Training would then raise "
              "an error; add --drop-test-overlap to remove them from training only.")
    else:
        assert_no_leakage(train=train, val=val, test=te_s)
        print("train/val/test subject leakage check: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit the ADNI dataset splits")
    parser.add_argument("--root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--meta", default=None, help=f"metadata JSON (default: <root>/../{META_NAME})")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit", action="store_true", help="print stats and run leakage checks")
    args = parser.parse_args()
    if args.audit:
        audit(args.root, args.val_fraction, args.seed, args.meta)
    else:
        parser.print_help()
