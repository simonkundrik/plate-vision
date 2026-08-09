# Model card: plate-vision

Covers the artifact published as [`model-v0.2.0`](https://github.com/simonkundrik/plate-vision/releases/tag/model-v0.2.0),
the first release whose nutrition head is trained. Written to be read before use rather than
after a disappointment.

It carries **two backbones**, one per head, each holding the weights it was fitted against.
Seven single-backbone runs traded classifier accuracy against calorie error monotonically and
none escaped it, so the cost is roughly double the file and a second forward pass rather than
a compromised head. Classification is bit-identical to `model-v0.1.0`: 100% top-1 agreement
over 512 images, maximum logit difference 0.00e+00.

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

That is a property of the method, not of any file. Offsets have to be refitted for the
checkpoint being released, with `scripts/fit_conformal.py`, and the numbers a training run
wrote before the calibration split was corrected do not deliver it. See the negative results.

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
| ±1.3x | 19.1% | 21.1% | 59.5% | 78.9% |
| ±1.6x | 25.5% | 23.4% | 46.6% | 73.3% |
| ±2.0x | **33.1%** | **28.9%** | 38.7% | **66.4%** |

**Unknown camera distance costs 9.5 points of calorie error even after the recipe**, down
from 15.0.

### These numbers replace an earlier table, which was measuring the wrong thing

The first version of this experiment cropped the 640x480 frames to a **square**. The eval
transform squashes whatever it is given to 224x224, so a square crop changed the aspect
distortion the model was trained under at the same moment it changed the scale, and part of
the reported degradation was a shape change wearing a scale label. Factors below 1 also
cropped past the frame and let PIL fill the overhang with black, a cue no camera produces.

The crop now preserves the source aspect ratio and pads by replicating the edge, which is the
convention `RandomZoomOut` already used for the same reason.

**The correction reverses a conclusion.** The old table showed run A at 30.6% and run B at
30.2% at ±2.0x, and this document concluded that the zoom-out augmentation had failed at what
it was added to do. Measured without the confound the gap is 33.1% against 28.9%: the
augmentation improves point accuracy under scale uncertainty by **4.2 points**, and costs 1.3
points on the rig, which is a coherent robustness trade rather than a null result.

The coverage story survives and is still the larger effect: 38.7% to 66.4% at the same
distortion. The model learned both to guess size better without the cue and to stop
pretending it knows. Only the first half was previously visible, and it was visible as
absent.

Coverage at ±2.0x is still 66.4%, far from 90%, so **the intervals still overclaim on
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
removed. The geometry is sound, implied density lands at a median 0.87 g/cm³, but plates
and bowls occlude food while adding their own volume, a sixth of pixels are sensor dropouts,
and flat dishes sit near the noise floor. Correlation with mass ranged from 0.28 to 0.62
across subsets. The planned native depth capture was cancelled on this evidence, before any
of it was written.

**Depth as a fourth input channel did not reproduce the paper's gain. It lost.** This is the
change with the strongest published evidence behind it: Nutrition5k reports 70.6 to 47.6 kcal
MAE, a 33% reduction, from handing the network the depth map alongside RGB. Phase H3 is that
experiment, run against phase D with an identical configuration, the same selection rule, and
both checkpoints selected at epoch 21.

| | calorie MAE | median APE |
|---|---|---|
| phase D, RGB | **54.7 kcal** | **19.6%** |
| phase H3, RGB + depth | 57.6 kcal | 20.5% |

**The model was not reading the channel.** Replacing the depth plane with a constant, and
separately with another dish's depth, on the same 506 test dishes:

| depth channel | calorie MAE | median APE |
|---|---|---|
| real | 57.64 | 20.49% |
| a constant | 58.86 | 20.96% |
| another dish's | 58.25 | 20.79% |

Real depth is worth **1.2 kcal** against a channel carrying no information at all, and 0.6
against one carrying the wrong dish's. So this is not depth costing more than it pays. It is a
model that routed around an input, and the 2.9 kcal it lost to phase D is the price of a wider
stem rather than a measurement of depth.

Two readings are consistent with that, and this project cannot currently tell them apart. The
RGB baseline here is already 22% ahead of the paper's, so there may be less for depth to add
than there was for theirs. Or 60 epochs at hyperparameters tuned for RGB, on a stem adapted
from RGB pretraining, is not enough for the channel to earn its place. **The honest claim is
that depth did not help here, not that depth cannot help.**

Reproduce with `scripts/measure_depth_contribution.py`.

The run also exposed a hole in the evaluator, which fed three channels to a four-channel model
and died in the first convolution after sixty epochs had trained. It reported as a failed run
when only its last step had failed, which is the second time in this project that a working
result has been nearly discarded because the thing measuring it broke.

**A scale reference would not recover what unknown camera distance costs.** The route was
going to detect a dinner plate, whose diameter is known to within a couple of centimetres,
and use it to convert apparent size into real size. Measuring the ceiling first, as with
depth, cancelled it.

Fitting `log(predicted / actual) = k · log(zoom)` recovers the exponent that explains the
model's behaviour under scale change. Volume would be 3, area 2:

| | fitted k | cost of unknown distance | a perfect detector recovers |
|---|---|---|---|
| Run A | 0.81 | 15.0 points | 6.6 points |
| Run B | **0.59** | 9.5 points | **0.8 points** |

**The model barely reads size at all.** Applying the physically correct correction makes
things far worse rather than better: `k=2` takes run B's calorie APE from 28.9% to 51.5%, and
`k=3` to 66.9%. A dinner plate is 26 to 28 cm, which is 7% uncertainty before any detection
error, and past 20% error the correction loses to doing nothing.

The gap between the runs is the useful part. Run B's zoom-out augmentation was added to
improve point accuracy under scale uncertainty and did not. It turns out to have already
bought most of what a scale reference could have offered, as a side effect of teaching the
model not to rely on the cue. There was less left to recover because the augmentation had
taken it.

**Conformal offsets fitted on held-out training data did nothing.** They came out at ±1.0
kcal for energy and negative for mass, and moved test coverage from 82.2% to 83.4%. Refitting
on a slice of the test split gives 92.9% on 253 dishes the offsets never saw.

The second half of this one is worse than the first. The corrected method landed in code, but
the checkpoint's `conformal.json` on disk still held the old values, and the export that
carries offsets into the bundle was verified against that file without anyone measuring the
coverage it produced. A bundle would have shipped labelled 90% and delivered 83.4%.
`scripts/fit_conformal.py` now refits from a checkpoint, reports coverage on a holdout it did
not calibrate on, and says so loudly when the result falls short.

The calibration set had been held out of training, which is true and is not the property
conformal prediction requires: it needs the calibration data to be **exchangeable** with the
data being predicted on. The model generalises measurably better to held-out-train than to
the official test split, so the observed misses were smaller than the real ones. *Not trained
on* is not the same as *exchangeable*, and the failure was quiet, because offsets that small
look like a well-calibrated model rather than a mis-specified experiment.

## Training data

- **Food-101**, 101k images, 101 classes, for the classifier.
- **Nutrition5k**, overhead RGB only, ~3,260 dishes with per-ingredient mass and macros.

Depth is unused by the shipped model. The original reasoning was that a phone has no depth
sensor and training on one would open a train/inference gap no tuning closes. Two things have
since qualified that: modern iPhone Pro devices do carry LiDAR and the market leader uses it,
and the published gain is large enough to be worth measuring rather than assumed away. The
RGB-only path stays the default, because most Android devices still cannot supply a depth map
and a model that requires one is a model most users cannot run.

**Rig depth and phone depth are not obviously the same quantity.** Nutrition5k's maps come
from a fixed camera at a median 3,562 mm, which `depth.normalise_depth` puts at 0.594 against
a training mean of 0.612 and a standard deviation of 0.052. A phone held 300 mm above a plate
normalises to 0.05, about **eleven standard deviations** outside the training range. Any
four-channel model trained on this dataset and pointed at a phone would be reading an input it
has never seen, and `depth.height_above`, which removes the camera distance and keeps the
shape, is the transform that could survive the move.

What remains genuinely unmeasured is whether a phone resolves real food as cleanly as the rig:
16% of rig pixels are dropouts, 39% on its worst frame, and a phone doing substantially worse
would make depth a harder problem rather than a rescalable one. The app carries a flagged,
estimate-free **depth lab** that measures exactly this and files it, because that number needs
hardware this project does not have. Until captures come back, nothing here claims phone depth
works.

That question is now downstream of a larger one. **Rig depth, which is the best depth this
dataset offers, did not produce a gain in this pipeline at all**, and the ablation above says
the model ignored it. Asking whether a phone's noisier version transfers is only worth doing
once something has been shown to benefit from the clean version. The depth lab is kept because
it costs nothing to leave running and the measurement is not obtainable any other way, not
because a capture is expected to unlock the paper's 33%.

## Where this sits against published work and against the market

Nutrition5k's own 2D direct-prediction results, on the dataset this model trains on:

| | MAE | MAPE |
|---|---|---|
| paper, RGB only | 70.6 kcal | 26.1% |
| paper, RGB + depth as a 4th channel | 47.6 kcal | 18.8% |
| **this model, RGB only** | **54.7 kcal** | 34.3% |

**On MAE this beats the published RGB-only baseline by 22%** and sits between it and the
RGB-D result, without a depth sensor.

The MAPE gap is an artifact of 31 tiny dishes rather than a deficit. Nutrition5k calories run
from 2 to 3,943 and the worst single error in the test split is 516% on a **10 kcal** dish.
Excluding dishes under 20 kcal, 6% of the split, gives **27.2% against the paper's 26.1%**.
That is parity, and it is why median APE is the headline here. The raw 34.3% should not be
quoted without saying what produces it.

For market context: SnapCalorie, built by the ex-Google researchers behind Nutrition5k,
publishes roughly **15% mean error using iPhone LiDAR**, and does not publish a figure for
devices without it. Their own comparisons put nutrition labels at a legally permitted 20%
error, dietitians near 40% and unaided users near 53%. A 10% target would beat dedicated
depth hardware and approach the resolution of the ground truth itself.

## Limitations

- **101 classes.** Anything outside them is named as the nearest match, confidently.
- **Cafeteria trays, not home cooking.** Nutrition5k is Google cafeteria food on a rig.
- **No validation on phone photos of real meals.** Ground-truth mass needs a kitchen scale and
  cannot be scraped. Every calorie figure describes the rig.

### One limitation that was stated wrongly

Earlier versions of this document said *"Single dish per photo. Realistic multi-item plates
need segmentation and are out of scope."* That is true of the **classifier**, which emits one
Food-101 label. It is false of the **nutrition head**.

Nutrition5k test dishes carry a median of **4 ingredients, mean 7.1, max 31**. 68% have two
or more and 46% have five or more; "generic food" in the paper's title means exactly this. So
54.7 kcal MAE and 19.6% median APE are already **mixed-plate** numbers, on plates more complex
than most home dinners. The claim understated what the model does for as long as it stood.

Stated as accuracy rather than error, on those mixed plates:

| condition | median APE | accuracy |
|---|---|---|
| fixed rig | 19.6% | **80.4%** |
| handheld, simulated at ±1.6x | 23.4% | 76.6% |
| handheld, simulated at ±2.0x | 28.9% | 71.1% |

The handheld rows are simulations produced by zooming rig photographs, not measurements of a
phone. That distinction is the reason the weighed-meal set still matters.

## Licence

**The code is MIT. The published weights are not.**

The classifier is trained on Food-101, whose images come from Foodspotting and are not ETH
Zurich's to relicense; the stated terms are that use beyond scientific fair use must be
negotiated with the individual image owners.

**Treat the weights as research use only.** The training pipeline is MIT, so weights you can
ship are a training run away.

Nutrition5k is CC BY 4.0 and requires attribution.
