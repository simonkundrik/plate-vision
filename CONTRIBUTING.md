# Contributing

## The one thing that would help most

**A depth capture from an iPhone Pro or iPad Pro.**

Depth is the largest untried accuracy lever in this project. Nutrition5k's own published result is
that feeding depth to the network as a fourth input channel takes calorie MAE from 70.6 to 47.6,
a 33% reduction, and depth is what the market leader ships via iPhone LiDAR.

What blocks acting on it is not a training run and not an idea. It is that Nutrition5k's depth
comes from a **fixed overhead rig at about 3.5 m**, a phone is held over a plate at arm's length,
and I own neither a device with a depth sensor nor an Apple developer account to build for one.

Part of the answer is already derivable and is written up in the
[model card](MODEL_CARD.md): a phone at 300 mm normalises about eleven standard deviations outside
the training range, so raw absolute depth cannot transfer as trained. What is **not** derivable is
whether a phone resolves real food as cleanly as the rig does. The rig drops about 16% of its
pixels and 39% on its worst frame. A phone doing substantially worse would make depth a harder
problem rather than a rescalable one, and that number needs hardware.

### What you need

**A device with LiDAR**, which means an iPhone 12 Pro or later Pro model, or an iPad Pro from 2020
on. A simulator is no use: the whole point is the sensor.

There are two ways to get the app onto it, and **the first needs no paid Apple account**.

#### With a Mac, and no paid account

Xcode's free provisioning signs a build for a device you own with an ordinary Apple ID. It expires
after seven days, which does not matter for taking one capture.

```bash
git clone https://github.com/simonkundrik/plate-vision.git
cd plate-vision && npm ci
cd app && EXPO_PUBLIC_ENABLE_DEPTH=1 npx expo run:ios --device
```

Pick your phone when it asks. The first run opens Xcode's signing setup if the project has no team
selected yet.

#### Without a Mac, using EAS

EAS builds on Expo's own macOS workers, so this works from Windows or Linux. It needs a free Expo
account and a **paid** Apple developer account, because a cloud build has to sign against real
credentials.

```bash
npm install -g eas-cli && eas login

cd app
eas init                 # replaces the projectId in app.json with yours
eas device:create        # registers your device, required for internal distribution
EXPO_PUBLIC_ENABLE_DEPTH=1 eas build --profile development --platform ios
```

`eas init` rewrites `extra.eas.projectId` in `app.json`. That is expected and local to you: the
committed value points at this repository's own EAS project, which you will not have access to.
Please leave that edit out of any pull request.

### Then

Install the build, open **Depth lab** from the camera screen, point it at a real plate of food,
and tap through to file the result.

### What gets reported

About a dozen summary numbers: resolution, dropout rate, depth percentiles, how far the food
stands above the surface behind it, and how far all of that sits from the training distribution.

**No photograph and no depth map are included.** The app's standing promise is that the picture is
analysed on the device and is not uploaded, and this feature does not make an exception to it. You
see the exact text before anything opens, and one function produces both the preview and the filed
report so the two cannot differ.

### Please do report the failures

The Swift in `app/modules/depth-capture/ios/` has **never been compiled**. It was written on a
Windows machine with no Apple hardware attached, against what the AVFoundation documentation says
the API does. A build error is a real finding and the likeliest first one.

The same goes for a capture that returns nothing, or values that look wrong. And the camera
distance check is *expected* to come back out of range for the reason above. That is the predicted
result, not a fault in your capture, and the app colours it amber rather than red so nobody throws
away a good capture for looking wrong.

## Anything else

Issues and pull requests are welcome. Nothing here is a rule for its own sake, but two things are
worth knowing about how this repository works.

**Every claim carries its conditions.** Accuracy figures in this project are quoted with the split,
the metric, and the tiny-dish handling attached, because all three move the number by several
points. A figure without them is the kind of claim this project has spent its whole history
refusing to make. The [model card](MODEL_CARD.md) records the negative results at the same length
as the positive ones, deliberately.

**Say what has and has not been verified.** "Typechecks" and "runs on a device" are different
statements and both get made explicitly. Several sections of the README exist only to say what was
never measured.

### Local checks

CI runs these, so running them first saves a round trip:

```bash
npm ci && npm run typecheck && npm run lint && npm test
```

```bash
cd model && ruff check . && ruff format --check . && pytest -q
```

The Python job additionally checks two generated files are current, since both are mirrors that
fail silently when they drift:

```bash
cd model && python scripts/sync_contract.py --check && python scripts/emit_depth_fixture.py --check
```
