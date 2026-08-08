// Expose legacy bridge modules to JavaScript under the New Architecture.
//
// Expo SDK 57 generates a bridgeless app: MainApplication uses ExpoReactHostFactory and
// DefaultNewArchitectureEntryPoint, and there is no ReactNativeHost at all. There is also no
// newArchEnabled escape hatch left in @expo/prebuild-config or expo-build-properties.
//
// onnxruntime-react-native 1.24.3, the newest published, registers as a plain ReactPackage
// with no codegenConfig. Under bridgeless that means NativeModules.Onnxruntime resolves to
// null, which surfaces as "Cannot read property 'install' of null" on the first inference.
//
// React Native still ships the interop layer for exactly this case; it is simply off by
// default. useTurboModuleInterop routes legacy ReactPackage modules through the TurboModule
// registry so NativeModules finds them.
//
// This is half the fix. The other half is patches/onnxruntime-react-native+1.24.3.patch,
// which stops install() reaching for CatalystInstance, an object bridgeless does not have.
// Both are needed: the flag makes the module visible, the patch makes it work once called.

const { withMainApplication } = require("expo/config-plugins");

const IMPORTS = [
  "import android.util.Log",
  "import com.facebook.react.internal.featureflags.ReactNativeFeatureFlags",
  "import com.facebook.react.internal.featureflags.ReactNativeFeatureFlagsDefaults",
];

// ReactNativeFeatureFlags.override throws IllegalStateException if any flag has already been
// read, and an uncaught throw in Application.onCreate is an app that opens and instantly
// closes. That is exactly what the first version of this plugin did: it injected the call
// after super.onCreate(), by which point something in Expo's startup had already read a flag.
//
// Two changes. It goes as early as possible, and it cannot take the app down. A failed
// override means onnxruntime is not reachable and the first photo reports that plainly,
// which is a diagnosable state. A crash on launch is not.
//
// This is deliberately the opposite of the build-time behaviour below, where a missing
// anchor throws. Refusing to produce a broken build is right; refusing to start an app
// because an optimisation could not be applied is not.
const OVERRIDE = `    try {
      // Legacy ReactPackage modules are invisible to NativeModules under bridgeless without
      // this. onnxruntime-react-native is one. See app/plugins/withTurboModuleInterop.js.
      ReactNativeFeatureFlags.override(
        object : ReactNativeFeatureFlagsDefaults() {
          override fun useTurboModuleInterop(): Boolean = true
        }
      )
    } catch (e: IllegalStateException) {
      // Something read a feature flag before this ran, so the override is refused. The app
      // still starts; onnxruntime will fail to resolve and say so on the first photo.
      Log.w("plate-vision", "could not enable useTurboModuleInterop: " + e.message)
    }
`;

// Immediately after super.onCreate(), which is as early as this can run while still being
// legal Android. Every line below it in the generated onCreate touches React Native.
const ANCHOR = "    super.onCreate()";

function addImports(contents) {
  let next = contents;
  for (const line of IMPORTS) {
    if (next.includes(line)) continue;
    next = next.replace(
      "import com.facebook.react.ReactPackage",
      `import com.facebook.react.ReactPackage\n${line}`,
    );
  }
  return next;
}

const withTurboModuleInterop = (config) =>
  withMainApplication(config, (cfg) => {
    if (cfg.modResults.language !== "kt") {
      throw new Error(
        `withTurboModuleInterop expects a Kotlin MainApplication, got ${cfg.modResults.language}`,
      );
    }

    let contents = addImports(cfg.modResults.contents);

    if (contents.includes("useTurboModuleInterop")) {
      cfg.modResults.contents = contents;
      return cfg;
    }

    if (!contents.includes(ANCHOR)) {
      // Failing loudly beats writing nothing. A plugin that silently no-ops leaves an app
      // that builds, installs, and dies on the first photo.
      throw new Error(
        "withTurboModuleInterop could not find loadReactNative(this) in MainApplication. " +
          "The Expo template changed; update the anchor.",
      );
    }

    // After the anchor, not before: super.onCreate() has to run first.
    contents = contents.replace(ANCHOR, `${ANCHOR}\n${OVERRIDE}`);
    cfg.modResults.contents = contents;
    return cfg;
  });

module.exports = withTurboModuleInterop;
