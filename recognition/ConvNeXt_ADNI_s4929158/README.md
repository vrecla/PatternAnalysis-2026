# ConvNeXt for Alzheimer's Disease Classification on ADNI (AD vs NC)

COMP3710/COMP7710 Pattern Recognition, individual project (Hard difficulty). Student: s4929158.

## 1. Overview
**Problem.** Classify 2D slices of T1-weighted brain MRI as Alzheimer's Disease (AD, label 1) or cognitively normal (NC, label 0), using the course copy of ADNI. The engineering question is not only accuracy: does the model's confidence mean anything on ambiguous cases, and can a reject option safely route uncertain cases to a human reviewer?

**Algorithm.** A hand-built, reduced ConvNeXt (Liu et al., 2022) trained from scratch. It keeps ConvNeXt's core design: a patchify stem, 7x7 depthwise convolutions, inverted-bottleneck MLP blocks, LayerNorm, layer scale, stochastic depth and separate downsampling layers. Dimensions (48, 96, 192, 384), depths (2, 2, 6, 2), stochastic-depth rate 0.3, 4,832,018 parameters. No pretrained weights are used. Two baselines are trained on identical splits: an MLP (590,722 parameters) and a small CNN with BatchNorm (1,173,154 parameters).

**How it works.** A 224x224 greyscale slice enters a "patchify" stem (a 4x4 convolution with stride 4, then LayerNorm) that turns the image into a 56x56 grid of 48-dimensional tokens. Four stages of ConvNeXt blocks follow (2, 2, 6 and 2 blocks, with 48, 96, 192 and 384 channels). Each block applies a 7x7 depthwise convolution to mix spatial information, LayerNorm, a 1x1 convolution that expands the channels 4x, a GELU, and a 1x1 convolution that projects back. The result is multiplied by a learnable per-channel layer-scale factor (initialised near zero so each block starts close to the identity) and added to the block's input through a residual connection, with stochastic depth randomly skipping blocks during training to regularise. Between stages, LayerNorm and a 2x2 stride-2 convolution halve the resolution and double the channels. At the end, global average pooling, LayerNorm, dropout (0.2) and a linear layer give two logits, trained with cross-entropy. The softmax of the logits is the model's confidence, which Section 6 evaluates and calibrates with temperature scaling.

**Headline result (clean, patient-level split; per slice, official test folder).** ConvNeXt reaches **0.693 +/- 0.005 test accuracy** and AUROC 0.762 +/- 0.010 over three training seeds. This is below the 0.80 Hard-tier target. The CNN baseline is as accurate (0.703 +/- 0.025) and has a higher AUROC (0.796 +/- 0.009). ConvNeXt is better calibrated and more sensitive to AD. The aggregate accuracy also hides a data problem: in the training folder every AD scan was acquired in 2011 to 2013 and every NC scan in 2017 to 2021, so a rule that looks only at the acquisition year scores 0.733 on the test set. Inside the 2019 to 2021 test scans, where the year cannot give the answer away, ConvNeXt reaches an AUROC of 0.682 and AD recall of 0.51. Details and caveats are in Sections 5 and 6.

**Figures.** Training curves: `runs/convnext_v2_dp/curves.png` (also `runs/cnn_auroc/curves.png`, `runs/mlp_auroc/curves.png`). Example predictions: `predictions/convnext_v2_dp/predictions_grid.png`.

![ConvNeXt training curves](runs/convnext_v2_dp/curves.png)
![Example test predictions](predictions/convnext_v2_dp/predictions_grid.png)


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
- Images resized to 224x224, single channel, normalised. Training augmentation is mild (small affine transform and brightness/contrast jitter, no flips, to avoid changing laterality). A horizontal flip, mixup and a stronger augmentation recipe were tested as ablations (Section 5.5).
- **Splits:** subject-level, stratified by each subject's majority label. 15% of training subjects (85 subjects, 1,880 slices) are held out for validation. `assert_no_leakage` checks that no subject is in two splits. The 216 subjects shared by the official train and test folders are removed from training only (8,820 slices), leaving 10,820 training slices. The official test folder (9,000 slices, 331 patients) is untouched and used once, at the end, with the checkpoint chosen on validation.
- Justification: slices and scans from one subject are near-duplicates, so a random split lets the model memorise people instead of learning disease features. The effect is quantified in Section 5.4.
- Using `meta_data_with_label.json`, I found that each prefix maps to a raw file name that contains an ADNI subject ID and ends in the same number as the prefix (for example prefix `1221051` maps to `ADNI_002_S_0413_..._I1221051.nii`, so the prefix is the ADNI image ID of one scan), and that one subject has 1 to 8 scans with different prefixes. Under that mapping 216 of the 680 subjects appear in both folders: subject `002_S_0413` has scan 863056 in train and scan 1221051 in test, and subject `002_S_1261` has scans 831065 and 989320 in train and scan 1270020 in test. Both settings are reported: the clean split (shared subjects removed from training; the main result) and the as-provided folders with shared subjects kept (Section 5.4, labelled "leaky" there because the shared subjects mean the test patients were also seen in training).
- Label and acquisition-year confound: the raw file names in the metadata carry a timestamp that I take to be the acquisition date (ADNI naming). Counting scans by that year and label gives the table below. In the training folder the year separates the classes perfectly, so a model can learn scan era (scanner or protocol appearance) instead of disease. The test folder breaks the pattern: all 227 NC scans are from 2019 to 2021, but so are about 88 of the AD scans, and every AD scan from 2017 onwards is in the test folder only. No split of the training folder can remove this, because training contains no AD scans from the later era. Section 5.3 measures its effect.

| Acquisition year | 2011 | 2012 | 2013 | 2014 | 2015 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|---|---|---|---|---|
| Train scans (AD share) | 57 (1.00) | 253 (1.00) | 210 (1.00) | 0 | 0 | 153 (0.00) | 292 (0.00) | 89 (0.00) | 17 (0.00) | 5 (0.00) |
| Test scans (AD share) | 0 | 0 | 54 (1.00) | 45 (1.00) | 4 (1.00) | 5 (1.00) | 28 (1.00) | 147 (0.33) | 133 (0.20) | 34 (0.35) |

- Reporting: all accuracies are per slice (each image counted separately). Patient-level accuracy is a supplementary figure only: it needs an extra aggregation rule that the task does not specify, and I use the mean of a patient's slice probabilities, thresholded at 0.5.

## 5. Results
Tuning and negative results (clean split, ConvNeXt, single seed)

Every configuration tried is listed (no cherry-picking). Selection used validation AUROC at the selected epoch only; test numbers are shown for transparency. Differences below about 0.01 validation AUROC are within the noise of an 85-subject validation set (per-epoch swings of up to 0.05).

| Run | Change from the base recipe | Val AUROC | Test acc | Test AUROC |
|---|---|---|---|---|
| v1 | drop path 0.1 | 0.839 | 0.692 | 0.768 |
| **v2 (final)** | **drop path 0.3** | **0.856** | **0.687** | **0.753** |
| v3 | + weight decay 0.1, lr 2e-4 | 0.829 | 0.680 | 0.756 |
| v4, v6, v7 | stronger augmentation (with variants) | training collapsed | about 0.50 | n/a |
| v5 | small model (1.90M parameters) | 0.830 | 0.666 | 0.769 |
| v10 | + horizontal flip | 0.846 | 0.681 | 0.785 |
| v11 | + mixup 0.2 | 0.844 | 0.631 | 0.691 |
| v12 | large model (27.8M parameters) | 0.855 | 0.617 | 0.769 |
| v13 | + EMA 0.999 | 0.845 | 0.671 | 0.756 |
| v14 | smoothed epoch selection (3 epochs) | 0.840 | 0.646 | 0.763 |
| v15 | EMA 0.998 + smoothed selection | 0.845 | 0.671 | 0.756 |

No variant beat v2 on validation by more than the noise level, so v2 was kept. Model size, augmentation, EMA and smoothing all land in the same band, which suggests that data (about 600 training subjects after removing the overlap) limits performance more than the architecture. The strong-augmentation runs failed to train (training accuracy stayed near 0.55); the cause was not investigated. The EMA runs make the validation curve smooth and settle near 0.845, suggesting that the true validation AUROC of the base recipe is about 0.84 to 0.85 and that v2's 0.856 is partly a lucky epoch. Because several configurations were compared on one small validation set, the selection carries a multiple-comparisons caveat.


## 6. Open research dilemma

## 7. Artificial Intelligence Usage Disclosure

## 8. References
