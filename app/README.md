# app

Expo React Native app, Android target.

```
App.tsx                     stage machine: capture -> working -> result
src/theme.ts                palette, 8pt spacing, type scale
src/types.ts                Interval, Analysis, portion scaling
src/labels.ts               class names, read from ../shared
src/inference/              model boundary (stubbed until PR #14)
src/components/             IntervalBar, PortionControl
src/screens/                CaptureScreen, ResultScreen
```

## Running it

```bash
npm install
npm run android      # needs a development build, see below
npm run web          # quick layout check, camera does not work here
npm run typecheck
npm run lint
```

## Why a development build rather than Expo Go

`onnxruntime-react-native` is a native module, so Expo Go cannot load it. The flow is one
EAS cloud build of the development client, installed on the device once, after which
JavaScript changes reload without rebuilding. The free tier allows 15 Android builds a
month, which is not a constraint under that workflow.

## Design decisions

**The interval is the headline, not a number.** A photograph carries no scale reference, so
portion size is genuinely unknowable and a single figure would be false precision. The
result screen shows `410–780 kcal` at display size with the median beneath it, and draws
the range as a band.

**The portion control is the most important thing on the screen.** It lets the person
holding the plate correct the one thing the model cannot see. Discrete steps rather than a
slider: nobody can judge "1.37 portions" by eye, and a slider would mean another native
module for no gain in accuracy.

**The placeholder announces itself.** Until PR #14 wires the real model, the result screen
carries a visible notice saying the numbers describe nothing. A demo that quietly shows
invented figures is how a project ends up claiming something it cannot do, and the failure
is invisible precisely because the numbers look reasonable.

**The photo is the only saturated thing on screen.** Everything else is near-neutral with a
single warm amber accent, chosen partly because it suits food and partly because it is not
the blue-violet that generated app UI arrives wearing by default.

**The contract is imported, not copied.** `metro.config.js` watches `../shared` so the app
reads the same `food101_labels.json` the exporter and training code read. A second copy is
a second ordering, and a drifted ordering does not crash: it renames every prediction.

## What is verified, and what is not

Typecheck and lint pass, and the app boots and renders under `expo start --web`.

The result screen has not been seen rendered: the web preview cannot reach it without a
camera, and the target is Android regardless. That happens on the first device build.

The web preview logs a `textShadow*` deprecation warning. It comes from react-native-web
internals and still appears with the relevant style deleted, so it is not this code and
does not affect the Android build, which does not use react-native-web at all.
