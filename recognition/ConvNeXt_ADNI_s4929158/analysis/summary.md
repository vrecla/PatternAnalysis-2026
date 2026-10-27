# Evaluation summary

Per-slice results on the held-out test set. Focus model for failure cases: **ConvNeXt**.

## Classification (test, per slice)

| Model | Accuracy | Balanced acc. | Macro-F1 | AUROC | AD recall (sens.) | NC recall (spec.) | AD precision | Patient-level acc. (suppl.) |
|---|---|---|---|---|---|---|---|---|
| MLP | 0.628 | 0.627 | 0.621 | 0.686 | 0.493 | 0.760 | 0.669 | 0.665 |
| CNN | 0.704 | 0.702 | 0.696 | 0.797 | 0.548 | 0.856 | 0.789 | 0.761 |
| ConvNeXt | 0.687 | 0.687 | 0.686 | 0.753 | 0.641 | 0.732 | 0.702 | 0.701 |
| CNN x3 | 0.729 | 0.729 | 0.728 | 0.806 | 0.670 | 0.787 | 0.756 | 0.740 |
| ConvNeXt x3 | 0.702 | 0.702 | 0.702 | 0.773 | 0.676 | 0.727 | 0.709 | 0.728 |
| Intensity-only | 0.606 | 0.605 | 0.594 | 0.665 | 0.437 | 0.772 | 0.653 | 0.656 |

## Confusion matrices (test; rows = true, columns = predicted)

- **MLP**: NC->NC 3452, NC->AD 1088, AD->NC 2259, AD->AD 2201
- **CNN**: NC->NC 3887, NC->AD 653, AD->NC 2015, AD->AD 2445
- **ConvNeXt**: NC->NC 3324, NC->AD 1216, AD->NC 1601, AD->AD 2859
- **CNN x3**: NC->NC 3574, NC->AD 966, AD->NC 1472, AD->AD 2988
- **ConvNeXt x3**: NC->NC 3301, NC->AD 1239, AD->NC 1443, AD->AD 3017
- **Intensity-only**: NC->NC 3504, NC->AD 1036, AD->NC 2509, AD->AD 1951

## Calibration (temperature fitted on validation)

| Model | T | ECE raw | ECE scaled | NLL raw | NLL scaled | Brier |
|---|---|---|---|---|---|---|
| MLP | 1.17 | 0.059 | 0.037 | 0.656 | 0.648 | 0.230 |
| CNN | 3.99 | 0.252 | 0.136 | 2.021 | 0.686 | 0.265 |
| ConvNeXt | 1.64 | 0.156 | 0.077 | 0.728 | 0.614 | 0.231 |
| CNN x3 | 2.73 | 0.214 | 0.121 | 1.316 | 0.636 | 0.234 |
| ConvNeXt x3 | 1.37 | 0.109 | 0.059 | 0.627 | 0.583 | 0.208 |
| Intensity-only | 1.10 | 0.044 | 0.032 | 0.668 | 0.664 | 0.236 |

## Reject option (threshold chosen on validation for <= 10% error among accepted slices)

| Model | Threshold | Test coverage | Test error among accepted | Error at 80% coverage | AURC |
|---|---|---|---|---|---|
| MLP | 0.927 | 0.2% | 9.1% | 33.9% | 0.296 |
| CNN | 0.716 | 77.2% | 24.3% | 25.0% | 0.191 |
| ConvNeXt | 0.830 | 39.4% | 19.8% | 27.7% | 0.209 |
| CNN x3 | 0.629 | 87.9% | 24.4% | 22.3% | 0.161 |
| ConvNeXt x3 | 0.834 | 36.6% | 16.1% | 25.7% | 0.190 |
| Intensity-only | 0.878 | 2.9% | 25.1% | 37.4% | 0.322 |

## Seed variability and ensembles

Mean +/- std over the individual runs (test, per slice; std is the sample std over runs), next to the logit-average ensemble of the same runs. Differences smaller than the std are not evidence of a real effect.

| Group | Runs | Accuracy (runs) | AUROC (runs) | Accuracy (ensemble) | AUROC (ensemble) |
|---|---|---|---|---|---|
| CNN x3 | 3 | 0.703 +/- 0.025 | 0.796 +/- 0.009 | 0.729 | 0.806 |
| ConvNeXt x3 | 3 | 0.693 +/- 0.005 | 0.762 +/- 0.010 | 0.702 | 0.773 |

- CNN x3: runs/cnn_auroc, runs/cnn_auroc_s1, runs/cnn_auroc_s2
- ConvNeXt x3: runs/convnext_v2_dp, runs/convnext_v2_dp_s1, runs/convnext_v2_dp_s2

## Uncertainty: patient-level bootstrap (1000 resamples of 331 test patients)

Whole patients are resampled, not single slices, because errors cluster by patient; slice-level intervals would be far too narrow. Intervals are 95% percentile intervals.

| Model | Accuracy [95% CI] | AUROC [95% CI] |
|---|---|---|
| MLP | 0.628 [0.587, 0.673] | 0.686 [0.633, 0.742] |
| CNN | 0.704 [0.661, 0.749] | 0.797 [0.749, 0.841] |
| ConvNeXt | 0.687 [0.652, 0.723] | 0.753 [0.709, 0.795] |
| CNN x3 | 0.729 [0.686, 0.772] | 0.806 [0.758, 0.850] |
| ConvNeXt x3 | 0.702 [0.668, 0.737] | 0.773 [0.730, 0.815] |
| Intensity-only | 0.606 [0.570, 0.642] | 0.665 [0.620, 0.709] |

Paired difference against **ConvNeXt** (model minus reference, same resamples; p = two-sided bootstrap p-value for a difference of zero, not corrected for the number of comparisons):

| Model | Δ accuracy [95% CI] | p | Δ AUROC [95% CI] | p |
|---|---|---|---|---|
| MLP | -0.059 [-0.089, -0.029] | 0.000 | -0.067 [-0.099, -0.035] | 0.000 |
| CNN | +0.017 [-0.019, +0.053] | 0.312 | +0.044 [+0.014, +0.075] | 0.004 |
| CNN x3 | +0.043 [+0.010, +0.073] | 0.010 | +0.054 [+0.024, +0.084] | 0.002 |
| ConvNeXt x3 | +0.015 [+0.003, +0.027] | 0.010 | +0.020 [+0.008, +0.031] | 0.000 |
| Intensity-only | -0.081 [-0.119, -0.045] | 0.000 | -0.088 [-0.129, -0.047] | 0.000 |

## Resources

| Model | Params | Peak train VRAM (MB) | Latency (ms/img, batch 1) | Throughput (img/s, batch 64) | Train time (min) | Best epoch |
|---|---|---|---|---|---|---|
| MLP | 590,722 | 40 | 0.16 | 388134 | 4.9 | 28 |
| CNN | 1,173,154 | 2017 | 1.02 | 5149 | 6.8 | 32 |
| ConvNeXt | 4,832,018 | 1885 | 5.55 | 5291 | 4.2 | 15 |
| CNN x3 | 3,519,462 | 2017 | 3.17 | 1718 | 16.2 | mixed |
| ConvNeXt x3 | 14,496,054 | 1885 | 17.54 | 1756 | 10.8 | mixed |
| Intensity-only | n/a | n/a | n/a | n/a | n/a | n/a |

## Is the ConvNeXt failure systematic? (evidence)

- 2817 of 9000 test slices are misclassified, spread over 237 of 331 patients.
- 103 patients have >= 50% of their slices wrong; they account for 70.9% of all errors (high share = patient-level/systematic failure, low share = scattered slice-level errors).
- Share of this model's errors that other models also make (high = hard cases for any model, e.g. data or label issues; low = model-specific):
  - MLP: 69.6%
  - CNN: 57.9%
  - CNN x3: 56.8%
  - ConvNeXt x3: 84.2%
  - Intensity-only: 60.0%
- Error rate by slice index: see `slice_error.png`; a strong dependence on slice index points to anatomical position as a trigger.

## Selected failure cases

| # | Category | True -> pred | Conf. | Patient | Slice | Patient's wrong slices | Mean intensity | Contrast (std) | Foreground |
|---|---|---|---|---|---|---|---|---|---|
| 1 | confident false negative (AD predicted NC) | AD -> NC | 0.95 | 130_S_6072 | 104 | 57/80 | 0.125 | 0.240 | 0.248 |
| 2 | confident false positive (NC predicted AD) | NC -> AD | 0.96 | 031_S_4021 | 97 | 20/20 | 0.144 | 0.254 | 0.285 |
| 3 | borderline error (probability near 0.5) | NC -> AD | 0.50 | 033_S_4177 | 87 | 12/20 | 0.115 | 0.210 | 0.283 |
| 4 | confident false negative (AD predicted NC) | AD -> NC | 0.95 | 130_S_6072 | 105 | 57/80 | 0.121 | 0.233 | 0.247 |
| 5 | confident false positive (NC predicted AD) | NC -> AD | 0.95 | 031_S_4021 | 98 | 20/20 | 0.150 | 0.264 | 0.286 |

Reference (mean +/- std over correctly classified test slices of each true class):

- NC: mean_intensity 0.123 +/- 0.020, contrast_std 0.233 +/- 0.029, foreground_frac 0.255 +/- 0.023
- AD: mean_intensity 0.110 +/- 0.019, contrast_std 0.214 +/- 0.031, foreground_frac 0.250 +/- 0.025
