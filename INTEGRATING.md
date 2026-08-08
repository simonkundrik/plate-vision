# Integrating plate-vision

For putting this in an application you did not write. Read the three surprises first; they
are the parts that cause trouble in review rather than in development.

## Three things that will surprise you

**1. `nutrition` can be `null`, and often is.** A model artifact can carry a trained
classifier and an untrained nutrition head. The library refuses to convert random weights
into kilocalories, so it returns `null` with a stated reason rather than a plausible number.
The currently published artifact is exactly this case.

```ts
if (result.nutrition === null) {
  console.warn(result.nutritionUnavailableReason);
}
```

Check `pv.nutritionAvailable` after loading if you want to know before analysing.

**2. Predictions are intervals, not numbers.** `energy` is `{ low, median, high }`, the 5th,
50th and 95th percentile. A photograph carries no scale reference, so portion size genuinely
cannot be recovered from it, and portion size is where nearly all calorie error lives. If
your interface needs one figure use `median`, and say it is an estimate.

**3. The headline accuracy is not the accuracy you will get.** 86.12% top-1 is measured on
Food-101's own test split. On Creative Commons photos taken by strangers it is **59.21%**,
and 27 of 101 classes score below 50%. Calorie figures are measured on a fixed overhead
camera rig; a phone held at an unknown distance costs about 12 points of calorie error.

Budget for the second number in each pair. See the [model card](MODEL_CARD.md).

## Install

```bash
npm install @plate-vision/client onnxruntime-web          # browser
npm install @plate-vision/client onnxruntime-react-native # React Native
```

The runtime is an optional peer dependency, so you install only the one you need. The same
import resolves to the right one through the package's exports map.

## Quickstart

```ts
import { load, decodeImage, parseBundle } from "@plate-vision/client";

const manifest = await fetch(`${base}/bundle.json`).then((r) => r.json());

const pv = await load({
  model: `${base}/${manifest.artifact.name}`,
  bundle: parseBundle(manifest),
});

const result = await pv.analyse(await decodeImage(file));

result.dishes[0];   // { key, label, confidence }
result.nutrition;   // { energy: { low, median, high }, ... } or null
result.inferenceMs; // time inside the model
```

`parseBundle` throws on a manifest it cannot read rather than filling in defaults. A model
whose provenance is unknown is what this library exists to refuse.

`load` also throws when the artifact's class count disagrees with the bundled label list,
rather than returning predictions mapped to the wrong names.

## Getting a model

Artifacts are published on [GitHub Releases](https://github.com/simonkundrik/plate-vision/releases),
not bundled into the package: they are tens of megabytes and version separately from the API.

**Pin a release tag rather than tracking the latest.** An artifact that changes underneath a
build you tested is a build whose behaviour you cannot reproduce.

**Browsers cannot fetch release assets directly.** GitHub sends no `Access-Control-Allow-Origin`
header on them, so serve the file from your own origin. React Native is not subject to CORS
and can use the release URL as-is. `web/scripts/fetch-model.mjs` in this repository does the
copy and verifies size and SHA-256 against the manifest, which catches a truncated download
before it becomes a confusing runtime error.

## Preprocessing is inside the model

You pass raw RGB bytes at whatever resolution you have. Resizing, normalisation and the
layout change are compiled into the ONNX graph.

This is deliberate. The alternative is every platform reimplementing the same ImageNet
constants and the same interpolation, and when one drifts nothing crashes: the app keeps
working and quietly returns wrong calories. Browser and Python agree to the digit on the same
photo because neither reimplements anything.

React Native has no canvas, so decoding there is yours to do. The model accepts any
resolution, so the only reason to resize first is decode cost.

## When it is wrong, let the user say so

`scaleNutrition(result.nutrition, 0.5)` rescales every target for a corrected portion. It
scales all three bounds rather than the median alone, because a corrected portion is not a
more certain one, and narrowing the interval on correction would claim confidence the model
never had.

Portion correction is the only correction signal available at inference time. An interface
that shows a number and offers no way to disagree with it throws that away.

## Barcode lookup

For packaged food, reading the wrapper beats estimating from the photo. The vision model is
worst at exactly the case a barcode is best at.

```ts
import { lookupBarcode, nutritionFromProduct } from "@plate-vision/client/barcode";

const result = await lookupBarcode(scanned);
if (result.found) {
  const nutrition = nutritionFromProduct(result.product, massEstimate);
}
```

**This is the one part of the library that makes a network request**, which is why it has its
own entry point rather than being reachable from the main import. Everything else runs
on-device and uploads nothing.

**It gives exact composition, not exact calories.** Open Food Facts states kilocalories per
100g; it does not say how much is on the plate. Against this project's own error
decomposition, where calorie error splits about evenly between mass and density, a barcode
collapses the density half and leaves the mass half untouched. All of the remaining interval
width comes from the mass, which is the honest attribution: pass a zero-width mass, because
the user weighed it or ate a stated serving, and you correctly get a zero-width answer.

Roughly a third of scans miss. `found: false` carries a reason and is an ordinary outcome for
a community-maintained database rather than an error.

## Compatibility

The package version tracks the **API**. The model version tracks the **weights**. They move
independently.

`bundle.json` carries `schema_version`, and `parseBundle` refuses a version it does not
understand rather than reading fields that may have moved. An old client and a new artifact
fail loudly instead of silently misinterpreting each other.

## Licence, in detail

**The code is MIT.** Both `@plate-vision/client` and the `platevision` Python package.

**The published weights are not, and this is the part that matters for shipping.**

The classifier is trained on Food-101, whose images come from Foodspotting. ETH Zurich does
not own them, and the stated terms are that use beyond scientific fair use must be negotiated
with the individual image owners. **Treat the published weights as research use only. Do not
ship them in a commercial product.**

The nutrition head is trained on Nutrition5k, which is CC BY 4.0. That licence permits
commercial use and requires **attribution**: credit Google Research's Nutrition5k dataset in
your application's acknowledgements or licences screen.

If you need weights you can ship without either constraint, train your own. The full
pipeline is in this repository and is MIT, and `RELEASING.md` documents publishing your own
artifacts.

## What this is not suitable for

Medical or clinical use, diet prescription, allergen detection, or anything where being wrong
by 30% matters. It has no notion of ingredients it cannot see, it assumes one dish per photo,
and it will confidently name an unfamiliar dish as the nearest of 101 classes.
