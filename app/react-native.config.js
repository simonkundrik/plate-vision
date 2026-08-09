// Point autolinking at where npm actually put things.
//
// This is an npm workspace, so dependencies hoist to the repository root rather than
// sitting in `app/node_modules`. iOS autolinking writes paths into the Podfile relative to
// `app/ios`, and it resolved `onnxruntime-react-native` to `../node_modules/...`, which is
// `app/node_modules` and does not exist. `pod install` then fails with
//
//     No podspec found for `onnxruntime-react-native` in `../node_modules/onnxruntime-react-native`
//
// which reads like a broken package rather than a layout problem.
//
// Resolving through Node finds the package wherever it was hoisted to, so this is correct
// whether it lands at the root or beside the app.
//
// Found by the ios module workflow on its first run. It is not a CI-only fix: any
// contributor running `npx expo run:ios` from `app/` on a Mac would have hit exactly this,
// which is most of the value of having that job at all.

// `root` alone was not enough: the podspec path is computed separately, so it is given
// outright. Absolute, because the whole failure was a relative path resolved from the wrong
// base.

const path = require("path");

const root = path.dirname(require.resolve("onnxruntime-react-native/package.json"));

module.exports = {
  dependencies: {
    "onnxruntime-react-native": {
      root,
      platforms: {
        ios: { podspecPath: path.join(root, "onnxruntime-react-native.podspec") },
      },
    },
  },
};
