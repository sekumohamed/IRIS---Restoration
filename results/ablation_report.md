# IRIS -- Ablation Study Results

All PSNR/SSIM figures below are **mean per-image** metrics on the held-out validation split (n=318), with 2 known corrupted samples excluded (file IDs 002637, 002973 -- confirmed to have ground-truth targets that are pure uniform random noise, GT std=0.289, matching the theoretical std of a uniform distribution over [0,1]; no restoration model can achieve meaningful PSNR against an unpredictable target, so these are excluded as non-informative rather than as genuine failures).

| Experiment | Parameters | Val PSNR (dB) | Val SSIM |
|---|---|---|---|
| Experiment 1: Baseline (pixel loss only) | 813,633 | 28.40 | 0.7703 |
| Experiment 2: + Structural/Edge loss | 813,633 | 28.26 | 0.7753 |
| Experiment 3: + Capacity (final model) | 4,522,449 | 29.05 | 0.7946 |

## Ablation narrative

**Experiment 1 -> Experiment 2 (loss change only, same 813K-parameter backbone):** adding structural (SSIM) and edge (Sobel gradient) loss terms on top of the baseline Charbonnier pixel loss improved SSIM (0.7703 -> 0.7753) but slightly reduced PSNR (28.40 -> 28.26 dB). This is the well-documented perception-distortion tradeoff: pixel-only loss rewards 'safe' blurred predictions that minimize average pixel error, while structural/edge loss rewards sharper, more perceptually faithful reconstructions even where that costs a small amount of raw pixel accuracy. Visual inspection confirmed this was a real improvement, not a metric artifact: fine texture (bark, fibrous surfaces) that was posterized/flattened by the pixel-only baseline was substantially better preserved after adding structural/edge loss.

**Experiment 2 -> Experiment 3 (capacity increase, same loss):** visual inspection of Experiment 2 showed that dense, high-frequency content (a crowd of faces) was still under-resolved despite the improved loss, suggesting the 813K-parameter backbone lacked sufficient representational capacity, not just the wrong training signal. Increasing the backbone to 4.5M parameters (more channels, more residual blocks, one additional block operating at full output resolution) while keeping the same structural/edge loss recovered PSNR *and* extended the SSIM gain (29.05 dB / 0.7946), beating both prior experiments on both metrics -- confirming capacity, not just loss design, was a genuine bottleneck.

## Recommendation

**Experiment 3 (IRISStronger + combined structural/edge/pixel loss) is the current best model** and is used for `evaluate.py` by default. Remaining known limitation: dense facial content in crowded scenes is still under-resolved relative to ground truth, likely reflecting a genuine information ceiling in the 128x128 input resolution rather than a fixable model deficiency -- flagged as a limitation rather than pursued further within this scope.
