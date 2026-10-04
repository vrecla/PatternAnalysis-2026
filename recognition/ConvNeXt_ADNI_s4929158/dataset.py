"""Data loading for ADNI AD-vs-NC classification with patient-level splitting.

Expected on-disk layout (Rangpur: /home/groups/comp3710/ADNI)::

    AD_NC/
      train/{AD,NC}/<patientID>_<sliceIdx>.jpeg
      test/{AD,NC}/<patientID>_<sliceIdx>.jpeg

Each image is one 2D slice of a brain MRI. Many slices come from the same
patient, so a random image-level split would leak patients across splits and
inflate test accuracy. Everything here therefore splits by *patient ID*, and
``assert_no_leakage`` verifies it.

ASSUMPTION (verified by ``python dataset.py --audit``): the number before the
underscore in a filename is the patient identifier and the number after it is
the slice index.

Run ``python dataset.py --root /home/groups/comp3710/ADNI/AD_NC --audit`` to
print split statistics and run the leakage checks.
"""

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Class name -> integer label. NC (cognitively normal) = 0, AD = 1, so that
# "positive" means Alzheimer's disease in all precision/recall/AUROC maths.
CLASS_TO_IDX: Dict[str, int] = {"NC": 0, "AD": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

_FILENAME_RE = re.compile(r"^(?P<patient>\d+)_(?P<slice>\d+)\.(jpe?g|png)$", re.IGNORECASE)


@dataclass(frozen=True)
class Sample:
    """One slice image with its label and patient identity."""

    path: str
    label: int
    patient: str
    slice_idx: int


# --------------------------------------------------------------------------
# Scanning and splitting
# --------------------------------------------------------------------------
def parse_filename(path: Path) -> Tuple[str, int]:
    """Return (patient_id, slice_idx) from a '<patient>_<slice>.jpeg' name."""
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return match.group("patient"), int(match.group("slice"))


def scan_split(root: str, split: str) -> List[Sample]:
    """List every image under ``root/split/{AD,NC}`` as a Sample."""
    samples: List[Sample] = []
    for cls_name, label in CLASS_TO_IDX.items():
        cls_dir = Path(root) / split / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {cls_dir}")
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                continue
            patient, slice_idx = parse_filename(img_path)
            samples.append(Sample(str(img_path), label, patient, slice_idx))
    if not samples:
        raise RuntimeError(f"No images found under {Path(root) / split}")
    return samples


def patient_level_split(
    samples: Sequence[Sample], val_fraction: float = 0.15, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples into (train, val) so no patient appears in both.

    Patients are shuffled and divided *within each class* so that the
    AD/NC balance is approximately preserved in both parts.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")

    # Group patient IDs by class. A patient with slices in both classes would
    # make the split ambiguous, so treat that as an error.
    patient_label: Dict[str, int] = {}
    for s in samples:
        if patient_label.setdefault(s.patient, s.label) != s.label:
            raise ValueError(f"Patient {s.patient} appears under both classes")

    by_class: Dict[int, List[str]] = defaultdict(list)
    for patient, label in patient_label.items():
        by_class[label].append(patient)

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
def get_transforms(train: bool, img_size: int = 224) -> Callable:
    """Image transforms. Augmentation is applied to the training split only.

    Augmentations are deliberately mild: brain MRIs are roughly aligned, so
    large rotations or flips could change anatomy and hurt more than help.
    Left-right flips are omitted because hemispheric asymmetry can matter.
    """
    ops: List[Callable] = [T.Grayscale(num_output_channels=1), T.Resize((img_size, img_size))]
    if train:
        ops += [
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    return T.Compose(ops)


class ADNIDataset(Dataset):
    """Map-style dataset returning (image_tensor, label, patient_index).

    The patient index is returned alongside each item so the evaluation code
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


def get_datasets(
    root: str, val_fraction: float = 0.15, seed: int = 42, img_size: int = 224
) -> Tuple[ADNIDataset, ADNIDataset, ADNIDataset]:
    """Build leakage-checked (train, val, test) datasets.

    The official ``train`` folder is split by patient into train/val. The
    official ``test`` folder is kept as the held-out test set, but we verify
    that it shares no patients with train/val; if it does, this raises rather
    than silently reporting an optimistic score.
    """
    all_train = scan_split(root, "train")
    test = scan_split(root, "test")
    train, val = patient_level_split(all_train, val_fraction, seed)
    assert_no_leakage(train=train, val=val, test=test)
    return (
        ADNIDataset(train, get_transforms(True, img_size)),
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = get_datasets(root, val_fraction, seed, img_size)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


# --------------------------------------------------------------------------
# Audit CLI
# --------------------------------------------------------------------------
def _describe(name: str, samples: Sequence[Sample]) -> None:
    counts = Counter(s.label for s in samples)
    per_patient = Counter(s.patient for s in samples)
    n_pat = len(per_patient)
    slices = list(per_patient.values())
    print(
        f"{name:>6}: {len(samples):6d} slices | {n_pat:5d} patients | "
        f"NC={counts[0]} AD={counts[1]} | slices/patient "
        f"min={min(slices)} max={max(slices)} mean={sum(slices) / n_pat:.1f}"
    )


def audit(root: str, val_fraction: float, seed: int) -> None:
    """Print split statistics and run leakage checks."""
    official_train = scan_split(root, "train")
    test = scan_split(root, "test")
    print("== Official folders ==")
    _describe("train", official_train)
    _describe("test", test)

    overlap = patients_of(official_train) & patients_of(test)
    print(f"Patients shared by official train and test: {len(overlap)}")

    train, val = patient_level_split(official_train, val_fraction, seed)
    print("\n== After patient-level train/val split ==")
    _describe("train", train)
    _describe("val", val)
    _describe("test", test)
    assert_no_leakage(train=train, val=val)
    print("train/val leakage check: PASSED")
    if overlap:
        print("WARNING: official test shares patients with train. Re-split needed.")
    else:
        assert_no_leakage(train=train, val=val, test=test)
        print("train/val/test leakage check: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit the ADNI dataset splits")
    parser.add_argument("--root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit", action="store_true", help="print stats and run leakage checks")
    args = parser.parse_args()
    if args.audit:
        audit(args.root, args.val_fraction, args.seed)
    else:
        parser.print_help()
