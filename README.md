# plate-vision

On-device meal calorie estimation from a single photo. Point a phone camera at a plate of food and
get the dish identified plus a calorie estimate with a calibrated uncertainty range, computed
entirely on the device with no network round trip and no per-request cost.

Android app built with Expo, plus a browser demo running the same model.

## The problem

Identifying *what* is on a plate is largely solved by modern vision models. Estimating *how much* is
not, and calories are almost entirely a *how much* question.

A single photo carries no scale reference. A bowl of pasta could be 200 calories or 900 with nearly
identical pixels. Portion size, not recognition, is where essentially all of the error in photo
calorie estimation lives. Published work on monocular estimation (single image, no depth sensor)
reports roughly 25 to 40% mean absolute error.

Most consumer calorie scanners hide this by assuming a standard serving and rendering a single
confident number. This project does the opposite:

- It predicts an **interval**, not a point estimate.
- It **measures whether that interval is calibrated**, so a stated 90% range is checked against how
  often it actually contains the truth.
- It lets the user **correct the portion**, and treats those corrections as future training signal.

## Approach

Three training stages, all PyTorch:

1. **Classification.** Fine-tune EfficientNet-B0 on Food-101 (101k photos, 101 dish classes) to
   learn food-domain features from diverse, realistic images.
2. **Nutrition regression.** Reuse that backbone on Nutrition5k (~5,000 real plated meals with
   per-ingredient mass and macronutrient labels) to predict calories, protein, fat, carbs, and mass.
   Trained with pinball loss at the 0.05 / 0.50 / 0.95 quantiles, which produces the displayed range
   directly and makes interval coverage a reportable metric.
3. **Compression.** Distill a larger teacher into the B0 student, then quantize to int8, targeting a
   model small and fast enough to run on a mid-range phone and inside a browser tab.

Deployment is via ONNX, so a single exported artifact serves both the Android app
(`onnxruntime-react-native`) and the web demo (`onnxruntime-web`). Image preprocessing is compiled
into the ONNX graph itself, meaning the model takes raw uint8 pixels and there is no opportunity for
Python, Android, and browser preprocessing to silently diverge.

## Results

Filled in as the work lands. Anything still reading TBD has not been measured, and nothing here is
a projection.

| Metric | Target | Measured |
|---|---|---|
| Food-101 top-1 | 85%+ | **86.12%** |
| Food-101 top-5 | | **96.90%** |
| Calorie MAE, Nutrition5k test split | competitive with published RGB-only baselines | TBD |
| Calorie MAE, personal weighed-meal set | reported honestly, expected to be worse | TBD |
| 90% interval coverage | 90% (+/- 3pp) | TBD |
| Model size, int8 | under 10 MB | **5.24 MB** (from 16.48 MB fp32) |
| Inference p95, mid-range Android | under 100 ms | TBD |

### Baseline run

EfficientNet-B0, 30 epochs, plain cross-entropy with AdamW and a cosine schedule. No mixup, no
cutmix, no label smoothing, no EMA. 4.19 hours on a single Kaggle T4.

```
epoch 28  train 98.04%   val 86.12%   <- best
epoch 29  train 98.07%   val 86.03%
```

The 12-point gap between train and validation accuracy is the point of running this configuration
first. It is the headroom the training recipe exists to close, and it is now a measured target
rather than an assumption. Throughput was roughly 195 images per second, well below what a T4
manages on this model, so the run was bound by JPEG decode and augmentation on four vCPUs rather
than by the GPU.

### On the int8 numbers

The size reduction is measured and architecture-determined, so it holds for the trained weights.
The **latency** result does not transfer: on the development laptop int8 is *slower* than fp32,
because that CPU has AVX2 but no AVX512-VNNI and the inserted quantization nodes are overhead that
never gets repaid. Phones have ARM dot-product instructions that ONNX Runtime does use, so the
on-device number is the one that matters and it has not been taken yet.

Two test sets are reported separately and deliberately. Nutrition5k is captured overhead on a fixed
rig in Google cafeterias, which is not what a handheld phone photo looks like. A personal set of
meals photographed by phone and weighed on a kitchen scale measures that domain gap instead of
hiding it.

## Known limitations

Recorded here from the start rather than discovered by a reader:

- **Depth is ignored.** Nutrition5k ships RGB-D and the original paper shows depth improves accuracy
  substantially. A phone camera will not have it, so training uses RGB only to keep the train and
  inference distributions aligned. Results will be worse than the paper's headline numbers as a
  direct consequence of this choice.
- **Domain gap.** Cafeteria trays shot from a fixed overhead rig are not handheld photos of home
  cooking. This is the project's largest technical risk.
- **Single-dish assumption.** The first version assumes one dish per photo. Realistic multi-item
  plates need segmentation or detection and are out of scope for now.

## Repo layout

```
model/     PyTorch training, evaluation, and ONNX export
app/       Expo React Native app (Android)
web/       Vite browser demo
shared/    model_meta.json, the single source of truth for the model contract
```

`shared/model_meta.json` defines input shape, layout, dtype, output names, and preprocessing
constants. All three surfaces read it. Preprocessing drift between training and the deployed
runtimes is the most common silent failure in projects of this shape, so it has exactly one
definition.

## Data

- [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), 101 classes, 101k images.
- [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k), Google Research, CC BY 4.0.
  Only the overhead RGB imagery is used; the full archive is mostly side-angle video that this
  project does not need.

Datasets are not vendored. `model/data/` holds acquisition scripts and is gitignored for contents.

## Status

Work lands as one pull request per task so the progression is readable end to end. See the
[pull request history](../../pulls?q=is%3Apr+is%3Aclosed) for the full sequence.

## Licence

MIT. See [LICENSE](LICENSE).
