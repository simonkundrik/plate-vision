# depth-capture

Experimental. One depth frame, measured against the training distribution, reported by hand.
**It changes no calorie estimate**, and nothing outside the depth lab screen reads it.

## What this is for

Nutrition5k's own published result is that feeding depth to the network as a fourth input
channel takes calorie MAE from 70.6 to 47.6 kcal, a 33% reduction, and depth is what the
market leader ships via iPhone LiDAR. That makes depth the single largest untried lever in
this project.

There is a problem with acting on it. Nutrition5k's depth comes from a **fixed overhead rig
at roughly 3.5 m**, and a phone is held over a plate at arm's length. Whether a model trained
on the first works on the second is an empirical question, and the arithmetic already says
part of the answer will be no:

`platevision/depth.py` normalises against a 6,000 mm ceiling, so the rig's 3,562 mm median
lands at 0.594, and training standardises around a mean of 0.612 with a standard deviation of
0.052. A phone at 300 mm normalises to 0.05, which is **about eleven standard deviations**
below anything the model has seen. Raw absolute depth from a phone is not going to transfer.

What is genuinely unknown, and what a capture from real hardware settles:

- **Dropout rate.** The rig drops about 16% of pixels, worst case 39%. If a phone drops far
  more on real food surfaces, depth is a harder problem than the paper suggests rather than a
  rescalable one.
- **Food relief.** How far the food stands above the surface behind it. This is the signal
  depth actually adds, and unlike absolute distance it does not change with how the phone is
  held. `depth.heightAbove` is the transform built for exactly this.
- **Accuracy flag.** Whether the platform reports the values as metric at all.

## What has and has not been verified

| | |
|---|---|
| `packages/client/src/depth.ts` | Unit tested, and checked against `platevision/depth.py` through a generated fixture |
| `src/depth/decode.ts` | Unit tested against a hand-built buffer |
| `src/depth/report.ts` | Unit tested |
| `ios/DepthCaptureModule.swift` | **Never compiled.** Written on a machine with no Apple hardware |
| The screen | Typechecked. Never rendered |

That table is why this ships behind a flag with an issue template attached rather than as a
feature. The people who can find out are the ones holding the hardware.

## Building it

The module is Apple-only, so an Android build is unaffected by everything in this directory.
Expo Go cannot load a custom native module at all, so it needs a development build, which
needs an Apple developer account. **Nobody working on this project has one**, which is why
[CONTRIBUTING.md](../../../CONTRIBUTING.md) asks for exactly that.

```bash
EXPO_PUBLIC_ENABLE_DEPTH=1 npx eas build --profile development --platform ios
```

Without the flag the entry point is not rendered, which is deliberate: an experiment that
reports "not available" to every user is worse than one they never meet.

LiDAR is on iPhone 12 Pro and later Pro models, and on iPad Pro from 2020. On anything else
`support()` returns false with that as the reason.

## Android

Not implemented, and the config declares Apple only, so the native module is simply absent
and the JavaScript reports it.

The route exists if it is wanted: ARCore's Depth API, through
`Frame.acquireDepthImage16Bits()`, already returns unsigned millimetres, which is the same
convention as Nutrition5k and as this module. It needs an ARCore session rather than a single
photo capture, which is a larger piece of work than the one-shot path above, and it should
wait until an Apple capture has said whether phone depth is worth the trouble.

## Design notes

**Unfiltered on purpose.** `isDepthDataFiltered = false` keeps the holes where the sensor
returned nothing. Filtering interpolates over them, and the dropout rate is one of the two
things worth measuring, so filtering would hide it.

**Millimetres at the boundary.** ARKit reports metres as float32, ARCore reports unsigned
millimetres, Nutrition5k stores unsigned millimetres. The conversion happens in Swift so that
exactly one unit crosses into JavaScript, rather than two conventions meeting somewhere in
the middle.

**Zero means "no reading" everywhere.** It is what Nutrition5k encodes, what `depth.py`
fills with the median of the valid pixels, and what this module writes for a NaN.
