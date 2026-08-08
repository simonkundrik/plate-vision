# Model card: plate-vision

Covers the artifact published as [`model-v0.1.0`](https://github.com/simonkundrik/plate-vision/releases/tag/model-v0.1.0)
and the nutrition head trained after it. Written to be read before use rather than after a
disappointment.

## What it is

A single EfficientNet-B0 exported to ONNX with two heads:

| Output | Shape | What it means |
|---|---|---|
| `logits` | (1, 101) | Food-101 dish classes |
| `nutrition_quantiles` | (1, 5, 3) | energy, protein, fat, carbohydrate, mass at the 5th, 50th, 95th percentile |

Preprocessing is compiled into the graph. It takes raw uint8 RGB at any resolution, so no
client reimplements the resize or the ImageNet constants. 16.48 MB fp32, roughly 100–400 ms
per photo in a browser tab.

Predictions are **intervals, not point estimates**, because portion size cannot be recovered
from a photograph and a single number would be precision the model does not have.

## Intended use

Estimating what a dish is, and roughly how many calories are on the plate, on-device and
without uploading the photo.

**Not suitable for** medical or clinical use, anything where being wrong by 30% matters,
diet prescription, or allergen detection. It has no notion of ingredients it cannot see, and
it will confidently name a dish it has never encountered as the nearest of 101 classes.

## Measured performance

Every figure here was measured. Nothing is projected, and the negative results are kept.

### Dish classification

| | |
|---|---|
| Food-101 test top-1 | **86.12%** |
| Food-101 test top-5 | **96.90%** |
| Creative Commons photos, top-1 | **59.21%** |
| Creative Commons photos, top-5 | **76.49%** |

The 26.91-point gap is the number that matters. Food-101's test split is drawn from the
same curated pool as its training data; the second row is 2,008 photos taken by strangers
with ordinary cameras. **Expect roughly 59%, not 86%, on a photo you take yourself.**

Per-class, the failure is coherent rather than random. Visually distinctive composed dishes
survive: spaghetti carbonara 100%, croque madame 95%, paella and bibimbap 90%. Generic or
ambiguous ones collapse: escargots 11%, omelette 21%, scallops 21%, cheesecake 32%. **27 of
101 classes score below 50%.**

That figure is an *upper bound* on true degradation: the OOD labels come from search terms
rather than from anyone inspecting the images, so some of the gap is wrong labels.

### Calorie estimation

Measured on the Nutrition5k test split, 506 dishes. Two runs are reported because the
difference between them is the point.

| | Run A, no recipe | Run B, full recipe |
|---|---|---|
| Calorie MAE | **51.6 kcal** | 56.7 kcal |
| Calorie median APE | **18.1%** | 19.4% |
| Mass MAE | 34.1 g | **33.2 g** |
| Interval crossing rate | 0.00% | 0.08% |
| 90% interval coverage | 64.6% | **82.2%** |
| Mean interval width | 120 kcal | 202 kcal |
| Train/validation loss ratio | 2.56 | **1.73** |

**Run A's better MAE was bought with overconfidence.** Its 90% interval contained the truth
64.6% of the time. The calibration curve says which bound is wrong: at the 0.05, 0.50 and
0.95 quantiles the empirical fractions were 0.180, 0.484 and 0.826. The median is well
calibrated; both bounds are pulled inward.

Run B adds mixup and cutmix blended in kilocalories rather than log space, zoom-out
augmentation, EMA, and checkpoint selection on validation pinball loss instead of MAE.
Selection is the one that mattered most: on Run A, MAE preferred epoch 33 at 64.6% coverage
over epoch 1 at 91.7%.

Calibration after the recipe: **0.075, 0.472, 0.897** against a target of 0.05, 0.50, 0.95.

Conformal prediction closes the remainder. Measured across 40 random calibration splits of
the test set, coverage moves to **90.5% ± 2.3**.

MAPE is unusable for fat and carbohydrate here, returning 1639% and 9782%, because
near-zero truths blow up the ratio. Median APE is the statistic to read.

### Where the calorie error lives

An oracle decomposition on the 506 test dishes, replacing one factor at a time with truth:

```
current energy head               51.6 kcal MAE   18.1% APE
oracle mass, predicted density    46.8            14.8%
oracle density, predicted mass    46.2            14.2%
```

**Neither oracle helps much.** Mass is 14.2% off, density 12.8%, and √(14.2² + 12.8²) ≈ 19%,
which is what is observed. The two error sources are roughly independent and combine in
quadrature, so halving either one alone buys about three points. There is no single decisive
lever.

## The thing most likely to mislead you

**Nutrition5k is a fixed overhead camera at constant height.** Apparent size in pixels
therefore *is* real size, and the model learns to read scale straight off the image. A phone
held at an unknown distance destroys that.

Simulating it by zooming the test images, for both runs:

| Camera distance | APE, run A | APE, run B | Coverage, run A | Coverage, run B |
|---|---|---|---|---|
| Known (the rig) | 18.1% | 19.4% | 64.6% | 82.2% |
| ±1.3x | 20.6% | 21.8% | 60.1% | 77.3% |
| ±2.0x | **30.6%** | **30.2%** | 42.9% | **62.1%** |

**Unknown camera distance costs 10.8 points of calorie error even after the recipe**, down
from 12.5 but not meaningfully.

The zoom-out augmentation in run B is worth reading carefully, because it did **not** do what
it was added to do. Point accuracy under scale uncertainty is unchanged: 30.6% to 30.2% is
inside the noise. What moved is coverage at the same distortion, 42.9% to 62.1%.

Augmentation cannot supply information the image does not contain. Scale genuinely is not
recoverable from a single RGB photo, so the model is no better at guessing size without the
cue. What it learned instead was to stop pretending it knows, and to widen its intervals when
the cue is missing. That is the more useful behaviour, and it is not what was intended.

Coverage at ±2.0x is still 62.1%, far from 90%, so **the intervals still overclaim on
phone-like input.**

Reproduce with `scripts/measure_scale_dependence.py`.

## Negative results, kept deliberately

**int8 quantization breaks this model.** A 5.24 MB build exists and agrees with fp32 on only
12.5–47.5% of predictions depending on calibration set size, erratic rather than improving.
This is the documented post-training quantization failure for EfficientNet: depthwise
separable convolutions with SiLU activations have per-channel ranges int8 cannot represent.
The export refuses to ship it below 90% agreement.

An earlier version of the README claimed int8 as a met target. That figure came from
measuring agreement on *randomly initialised weights*, whose activation ranges are small and
well conditioned. It reported 100% and proved nothing.

**Depth-derived volume did not help.** Integrating an overhead depth map above the table
plane gives 104.9 g mass MAE against RGB's 71.6 g even after RGB's scale advantage is
removed. The geometry is sound — implied density lands at a median 0.87 g/cm³ — but plates
and bowls occlude food while adding their own volume, a sixth of pixels are sensor dropouts,
and flat dishes sit near the noise floor. Correlation with mass ranged from 0.28 to 0.62
across subsets. The planned native depth capture was cancelled on this evidence, before any
of it was written.

**Conformal offsets fitted on held-out training data did nothing.** They came out at ±1.0
kcal for energy and negative for mass, and moved test coverage from 82.2% to 83.4%. Refitting
on a slice of the test split gave 90.5%.

The calibration set had been held out of training, which is true and is not the property
conformal prediction requires: it needs the calibration data to be **exchangeable** with the
data being predicted on. The model generalises measurably better to held-out-train than to
the official test split, so the observed misses were smaller than the real ones. *Not trained
on* is not the same as *exchangeable*, and the failure was quiet, because offsets that small
look like a well-calibrated model rather than a mis-specified experiment.

## Training data

- **Food-101**, 101k images, 101 classes, for the classifier.
- **Nutrition5k**, overhead RGB only, ~3,260 dishes with per-ingredient mass and macros.

Depth is deliberately unused as an input: a phone has no depth sensor, and training on it
would open a train/inference gap that no tuning closes.

## Limitations

- **Single dish per photo.** Realistic multi-item plates need segmentation and are out of scope.
- **101 classes.** Anything outside them is named as the nearest match, confidently.
- **Cafeteria trays, not home cooking.** Nutrition5k is Google cafeteria food on a rig.
- **No validation on phone photos of real meals.** Ground-truth mass needs a kitchen scale and
  cannot be scraped. Every calorie figure describes the rig.

## Licence

**The code is MIT. The published weights are not.**

The classifier is trained on Food-101, whose images come from Foodspotting and are not ETH
Zurich's to relicense; the stated terms are that use beyond scientific fair use must be
negotiated with the individual image owners.

**Treat the weights as research use only.** The training pipeline is MIT, so weights you can
ship are a training run away.

Nutrition5k is CC BY 4.0 and requires attribution.
