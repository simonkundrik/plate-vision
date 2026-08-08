# @plate-vision/client

On-device food recognition and calibrated calorie intervals from a single photo. The same
ONNX model runs in React Native and the browser; nothing is uploaded.

```bash
npm install @plate-vision/client onnxruntime-web          # browser
npm install @plate-vision/client onnxruntime-react-native # React Native
```

The runtime is an optional peer dependency, so you install only the one you need.

## Usage

```ts
import { load, decodeImage } from "@plate-vision/client";

const pv = await load({
  model: "https://example.com/plate-vision-int8.onnx",
  bundle: await fetch("https://example.com/bundle.json").then((r) => r.json()),
});

const image = await decodeImage(fileOrUrlOrBitmap);
const result = await pv.analyse(image);

result.dishes[0];      // { key: "spaghetti_carbonara", label: "Spaghetti carbonara", confidence: 0.71 }
result.nutrition;      // { energy: { low, median, high }, protein: {...}, ... } or null
result.inferenceMs;    // time inside the model
```

React Native resolves the same import to the native runtime automatically. There is no
canvas there, so decoding is yours to do; the model accepts any resolution, so the only
reason to resize first is decode cost.

## Predictions are intervals, not numbers

`energy` comes back as `{ low, median, high }`, the model's 5th, 50th, and 95th percentile
predictions.

That is not decoration. A photograph carries no scale reference, so portion size genuinely
cannot be recovered from it, and portion size is where nearly all calorie error lives.
Published work on single-image calorie estimation reports roughly 25 to 40 percent mean
absolute error. A single number would be precision the model does not have.

Show the range. If your UI needs one figure, use `median` and say it is an estimate.

When a user corrects the portion, `scaleNutrition(result.nutrition, 0.5)` rescales every
target. It scales all three bounds, not just the median: a corrected portion is not a more
certain one, and narrowing the interval on correction would claim confidence the model
never had.

## `nutrition` can be `null`, and you should handle it

```ts
if (result.nutrition === null) {
  console.warn(result.nutritionUnavailableReason);
}
```

A model artifact can carry a trained classifier and an **untrained** nutrition head. This
library refuses to return numbers from random weights, because a plausible-looking figure
from an untrained head is the easiest way to publish something false without noticing.

Check `pv.nutritionAvailable` after loading if you want to know before analysing.

## Preprocessing happens inside the model

You pass raw RGB bytes at whatever resolution you have. Resizing, normalisation, and the
layout change are compiled into the ONNX graph.

This is deliberate. The alternative is every platform reimplementing the same ImageNet
constants and the same interpolation, and when one drifts nothing crashes: the app keeps
working and quietly returns wrong calories.

## Model artifacts and licensing

**The code in this package is MIT.** The model weights are not covered by it.

The published classifier is trained on Food-101, whose images come from Foodspotting and
are not ETH Zurich's to relicense; the stated terms are that use beyond scientific fair use
must be negotiated with the individual image owners. Treat the published weights as
**research use only** and do not ship them in a commercial product.

The nutrition head is trained on Nutrition5k, which is CC BY 4.0 and requires attribution.

If you need weights you can ship, train your own on data you have rights to. The training
pipeline is in the same repository and is MIT.

## API

| Export | Purpose |
|---|---|
| `load(options)` | Create a session using the platform's ONNX runtime |
| `PlateVision` | The session class, if you want to inject a runtime yourself |
| `decodeImage(source, maxEdge?)` | Browser only. Canvas decode to raw RGB |
| `stripAlpha(rgba)` | Drop the alpha channel from RGBA bytes |
| `scaleInterval`, `scaleNutrition` | Apply a user's portion correction |
| `softmax`, `topK`, `enforceMonotonic` | Postprocessing primitives |
| `contract`, `labels`, `classCount` | The model contract, shipped with the package |

## Versioning

The package version tracks the API. The **model artifact** is versioned separately and
declares its own contract; `load` throws if a model's class count disagrees with the label
list this package ships, rather than returning predictions mapped to the wrong names.
