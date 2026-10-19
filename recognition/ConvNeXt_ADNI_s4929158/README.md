# ConvNeXt for Alzheimer's Disease Classification on ADNI (AD vs NC)

COMP3710/COMP7710 Pattern Recognition, individual project (Hard difficulty). Student: s4929158.

## 1. Overview

## 2. Feasibility review
**1. User need, scope and acceptance criteria.**
*User:* a clinical reviewer who wants an assistant that auto-classifies clear cases and routes uncertain ones to an expert. *Prototype:* 2D slice-level AD vs NC classifier with calibrated confidence and a reject option; not a diagnostic tool. *Acceptance criteria (testable):*
1. Test accuracy of ConvNeXt >= 0.80 (per slice) on the official test folder, with no subject in more than one split.
2. ConvNeXt compared with the MLP and CNN baselines on identical splits, reporting per-class precision/recall, macro-F1 and AUROC.
3. Calibration reported (ECE, Brier, NLL), with temperature fitted on validation only. Target: ECE <= 0.10 after scaling.
4. A reject threshold chosen on validation that gives accepted-case accuracy of at least 90% at coverage of at least 30% on test.
5. Resource constraint: peak GPU memory and inference latency profiled. Target: trains and runs on one A100 at batch 64, with batch-1 latency <= 10 ms per image.

**2. Model choice and course concepts.** Dataset: ADNI 2D slices (train 21,520 slices / 1,076 scans; test 9,000 slices / 450 scans; 20 slices per scan). Model: ConvNeXt (Hard band), chosen because it modernises a plain CNN with transformer-era design choices while staying convolutional, so it trains from scratch on a small medical dataset. Course concepts used: CNNs and normalisation, regularisation (weight decay, stochastic depth, augmentation), optimisation (AdamW, warmup + cosine), evaluation beyond accuracy (calibration, ROC), and failure analysis. Reference: Liu et al., "A ConvNet for the 2020s", CVPR 2022.

**3. Preliminary feasibility evidence.**
- *Data audit* (`python dataset.py --audit`): file names carry a scan ID, not a patient ID. Using `meta_data_with_label.json`, 1,526 scans map to 680 subjects (1-8 scans each). 216 subjects appear in both the official train and test folders (5,560 of 9,000 test slices, 61.8%). My first random scan-level split leaked 71.4% of validation slices.
- *Consequence:* validation accuracy was inflated (CNN 0.96 validation vs 0.74 test). I switched to a subject-level split and remove the 216 shared subjects from training only (official test folder untouched). Final clean split: train 10,820 slices, validation 1,880 slices (85 subjects), test 9,000 slices.
- *First results:* on the clean split ConvNeXt reaches 0.687 and the CNN 0.704 test accuracy (single runs), so the 0.80 target is not met. On a leaky split that keeps the shared subjects (train 18,100 slices) the same models reach 0.737 and 0.757, which shows how much leakage and extra data inflate the numbers (Section 5.4).

**4. Risks, budget and fallback.**
- *Risks:* (a) ConvNeXt may stay below 0.80 on the clean split because training data shrinks (this happened); (b) slices of one scan are highly correlated, so 2D slice accuracy can overstate real performance; (c) queue delays on Rangpur.
- *Budget:* one A100, 1 job per user, 12 h QOS limit. Each final model trains in 4 to 7 minutes (Section 6.3); the tuning study of about 20 runs fitted comfortably in the budget.
- *Next experiment:* tune ConvNeXt (drop path, weight decay, augmentation, size, EMA) using validation only, then report seeds, confidence intervals and the failure analysis.
- *Fallback:* if still below 0.80, report the honest result with the failure analysis, and with tutor/Ed approval consider pooling all subjects into my own subject-level split. Optional extra: GFNet comparison only after the main analysis is solid.

## 3. Dependencies and reproducibility
Python 3.11.13 with matplotlib 3.11.2, numpy 2.4.6, pillow 12.3.0, scikit-learn 1.9.1, scipy 1.17.1, torch 2.5.1+cu121 and torchvision 0.20.1+cu121.

## 4. Data and pre-processing

## 5. Results

## 6. Open research dilemma

## 7. Artificial Intelligence Usage Disclosure

## 8. References
