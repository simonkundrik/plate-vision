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

Full detail, including the negative results, is in the [model card](MODEL_CARD.md).

| Metric | Target | Measured |
|---|---|---|
| Food-101 top-1 | 85%+ | **86.12%** |
| Food-101 top-5 | | **96.90%** |
| Top-1 on out-of-distribution photos | | **59.21%** (see caveat below) |
| Calorie MAE, Nutrition5k test split | competitive with published RGB-only baselines | **56.7 kcal**, 19.4% median APE |
| Calorie median APE, unknown camera distance | | **28.9%** (see below) |
| Calorie MAE, personal weighed-meal set | reported honestly, expected to be worse | not collected |
| 90% interval coverage | 90% (+/- 3pp) | **82.2% as trained, 90.5% +/- 2.3 conformalised** |
| Deployable model size | under 10 MB | **16.48 MB fp32.** int8 is 5.24 MB but unusable, see below |
| Inference p95, mid-range Android | under 100 ms | not measured; ~100-400 ms in a browser tab |

### The rig is doing more work than the model

Nutrition5k is a fixed overhead camera at constant height, so apparent size in pixels *is*
real size and the model learns to read scale off the image. Simulating a phone at unknown
distance by zooming the test images:

| Camera distance | Calorie median APE | Interval coverage |
|---|---|---|
| Known (the rig) | 19.4% | 82.2% |
| ±2.0x | **28.9%** | **66.4%** |

**Unknown camera distance costs 9.5 points of calorie error.** This is the closest thing to
a deployment number obtainable without weighed photographs.

Zoom-out augmentation was added to close that gap and did not: point accuracy under scale
uncertainty is unchanged. What it did instead was make the model less confident when the cue
is missing, lifting coverage at that distortion from 38.7% to 66.4%. Augmentation cannot
supply information the image does not contain.

A scale reference, detecting a dinner plate of known diameter, would not close it either. The
model turns out barely to read apparent size: the exponent relating its error to zoom is 0.59
where volume would be 3, and a *perfect* detector recovers 0.8 of the 9.5 points. That route
was cancelled on the measurement rather than built. See the [model card](MODEL_CARD.md).

Reproduce with `model/scripts/measure_scale_dependence.py` and `measure_scale_recovery.py`.

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

### The 27-point drop on real-world photos

Food-101's test split is drawn from the same curated pool as its training data. Measured
against 2,008 Creative Commons photos taken by strangers with ordinary cameras, the same
model scores **59.21% top-1** (76.49% top-5) rather than 86.12%.

```
      86.12%   Food-101 test split
      59.21%   Creative Commons photos
      -26.91   points
```

The per-class breakdown is consistent with an honest reading rather than random collapse.
Visually distinctive composed dishes survive: spaghetti carbonara 100%, croque madame 95%,
paella and bibimbap 90%. Generic or ambiguous ones fall apart: omelette 21%, cheesecake 32%,
hot dog 35%, macaroni and cheese 35%.

**This figure is an upper bound on the true degradation.** The OOD labels come from search
terms, not from anyone inspecting the images, so some of the 27 points is wrong labels
rather than model error. `data/ood/manifest_review_sample.json` holds 100 randomly selected
images for hand-checking; reviewing them turns this into a number with an error bar. Until
that happens the honest claim is "somewhere between a meaningful drop and 27 points", not
27 points.

The set is defined by a tracked manifest of URLs and attribution rather than committed
images, so the repository redistributes nothing. All entries are CC BY, CC BY-SA, CC0, or
public domain, filtered for commercial use and modification.

### int8 quantization does not work on this model

The size reduction is real: 16.48 MB to 5.24 MB. The resulting model is not usable.

Measured against the trained baseline, int8 changes **most predictions**. Agreement with the
fp32 model, on held-out images across three calibration set sizes:

```
  32 calibration images   25.0% agreement
 128 calibration images   15.0%
 384 calibration images   47.5%
```

Erratic rather than improving, so it is not a shortage of calibration data. This is the known
post-training quantization failure for EfficientNet: depthwise separable convolutions with SiLU
activations have per-channel dynamic ranges that int8 cannot represent. Per-channel weight
quantization is already enabled and is not enough.

The export refuses to ship an int8 artifact below 90% agreement and keeps fp32 instead. A
smaller model that predicts something else is not a smaller model.

**An earlier version of this file claimed int8 as a met target.** That figure came from
measuring agreement on a model with randomly initialised weights, whose activation ranges are
small and well conditioned, which reported 100% agreement and proved nothing. Validating
quantization against an untrained model is a measurement that cannot fail.

Routes worth trying, none of them yet attempted: quantization-aware training, cross-layer
equalisation before quantizing, keeping the worst layers in fp32, or a backbone chosen for
quantization friendliness rather than for parameter count.

Latency separately does not transfer from this machine: the development laptop has AVX2 but no
AVX512-VNNI, so int8 measures *slower* than fp32 here. Phones have ARM dot-product instructions
ONNX Runtime does use, and that number has not been taken.

Two test sets are reported separately and deliberately. Nutrition5k is captured overhead on a fixed
rig in Google cafeterias, which is not what a handheld phone photo looks like. A personal set of
meals photographed by phone and weighed on a kitchen scale measures that domain gap instead of
hiding it.

## Known limitations

Recorded here from the start rather than discovered by a reader:

- **The shipped model ignores depth.** Nutrition5k ships RGB-D and the paper's own result is that
  feeding depth to the network as a fourth channel takes calorie MAE from 70.6 to 47.6, a 33%
  reduction. Every published figure here is RGB only, so the gap between them and the paper's
  headline is a direct consequence of that choice. See the depth experiment below.
- **Domain gap.** Cafeteria trays shot from a fixed overhead rig are not handheld photos of home
  cooking. This is the project's largest technical risk.
- **Single-dish assumption, for the classifier only.** It emits one Food-101 label, so a plate
  with several distinct dishes gets the nearest single name. The **nutrition head has no such
  limit**: Nutrition5k dishes carry a median of 4 ingredients and a mean of 7.1, so every
  calorie figure here is already a mixed-plate number.

## The depth experiment, and why it needs someone else's phone

Depth is the largest untried accuracy lever here, and it is what the market leader ships via
iPhone LiDAR. What blocks it is not a training run. It is that Nutrition5k's depth comes from a
**fixed overhead rig at about 3.5 m** and a phone is held over a plate at arm's length, and nobody
working on this project owns a device with a depth sensor to find out whether those are close
enough to be the same problem.

Part of the answer is already derivable. `platevision/depth.py` normalises against a 6,000 mm
ceiling, so the rig's 3,562 mm median lands at 0.594 against a training mean of 0.612 with a
standard deviation of 0.052. A phone at 300 mm normalises to 0.05, roughly **eleven standard
deviations** outside anything the model has seen. Raw absolute depth from a phone will not
transfer, and `depth.heightAbove`, which subtracts the camera distance and keeps the shape, is the
channel that might.

What cannot be derived is whether a phone sees a plate of real food as cleanly as the rig does:
the rig drops about 16% of its pixels and reaches 39% on its worst frame, and a phone struggling
far more than that would make depth a harder problem rather than a rescalable one.

So the app carries a **depth lab**: one capture, measured against the training distribution, every
number shown, and a one-tap prefilled issue. It is off unless a build sets
`EXPO_PUBLIC_ENABLE_DEPTH=1`, it changes no calorie estimate, and no photo or depth map leaves the
device. The Swift that talks to the sensor has never been compiled, which is stated on the screen,
in the module's [README](app/modules/depth-capture/README.md), and in the issue template.

**If you have an iPhone 12 Pro or later, or an iPad Pro from 2020 on, a capture from it is worth
more to this project than anything else on this page.** A failed capture is as useful as a working
one.

## Use it in your own app

The inference path ships as a library, so this does not have to be used as a whole
application. Install the client and the ONNX runtime for your platform:

```bash
npm install @plate-vision/client onnxruntime-web
```

```ts
import { load, decodeImage } from "@plate-vision/client";

const pv = await load({ model: modelUrl, bundle });
const result = await pv.analyse(await decodeImage(file));

result.dishes[0];  // { key, label, confidence }
result.nutrition;  // { energy: { low, median, high }, ... } or null
```

The same import resolves to `onnxruntime-react-native` under React Native. Preprocessing
is inside the ONNX graph, so there is nothing to reimplement per platform.

`result.nutrition` is `null` when the loaded artifact's nutrition head is untrained, which
is true of the currently published weights. The library refuses to return numbers derived
from random weights rather than letting them reach a UI looking like measurements.

**Read [INTEGRATING.md](INTEGRATING.md) before shipping this in anything.** It covers the
three behaviours that surprise people in review rather than in development, and the licence
position in detail.

**The code is MIT. The published weights are not** — they are trained on Food-101, whose
images are not ETH Zurich's to relicense. Treat them as research use only, and train your
own for anything you intend to ship.

## Running the Android app

The model is not bundled into the APK or committed here: it is tens of megabytes and would
sit in every clone. The app fetches it on first launch from a published release and caches
it under a name that includes the artifact hash, so a re-exported model is actually picked
up rather than shadowed by the previous file.

`app.json` points at a pinned release tag, so a checkout runs against the model it was
tested with. Override with `EXPO_PUBLIC_MODEL_URL` / `EXPO_PUBLIC_BUNDLE_URL`.

```bash
npx expo start --dev-client
```

The published artifact identifies the dish and **reports no calorie estimate**. Its
nutrition head is untrained, `bundle.json` records that, and the client withholds the
output rather than returning values from random weights. See [RELEASING.md](RELEASING.md).

`onnxruntime-react-native` is a native module, so Expo Go will not run this. It needs a
development build:

```bash
npx eas build --profile development --platform android
```

**The app has not been run on a device.** Everything in `app/` typechecks and its pure
logic is unit tested, but no build has been made, so the ONNX session, the camera path, and
the model download have never executed. That is a statement about what has been verified,
not a guess about whether it works.

## Repo layout

```
model/             PyTorch training, evaluation, and ONNX export
app/               Expo React Native app (Android)
web/               Vite browser demo
packages/client/   @plate-vision/client, the npm library the app and demo both use
shared/            model_meta.json, the single source of truth for the model contract
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
