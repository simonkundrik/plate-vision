/**
 * Where the model comes from.
 *
 * The artifact is tens of megabytes, so it is published on GitHub Releases rather than
 * bundled into the APK or committed here. That makes its location configuration, and
 * configuration that is missing has to be reported as missing: an app that quietly falls
 * back to something is an app that will one day analyse a meal with the wrong model.
 */

import Constants from "expo-constants";

type ModelSource = {
  modelUrl: string;
  bundleUrl: string;
};

const extra = (Constants.expoConfig?.extra ?? {}) as Partial<ModelSource>;

/**
 * Environment variables win over app.json so a build can be pointed at a different release
 * without editing tracked files.
 */
const resolve = (envValue: string | undefined, configured: string | undefined): string =>
  (envValue ?? configured ?? "").trim();

export const modelSource = (): ModelSource | null => {
  const modelUrl = resolve(process.env.EXPO_PUBLIC_MODEL_URL, extra.modelUrl);
  const bundleUrl = resolve(process.env.EXPO_PUBLIC_BUNDLE_URL, extra.bundleUrl);

  if (!modelUrl || !bundleUrl) return null;
  return { modelUrl, bundleUrl };
};

/**
 * Whether the experimental depth capture is reachable in this build.
 *
 * Off unless `EXPO_PUBLIC_ENABLE_DEPTH` is set, and off by default on purpose: the feature
 * measures a depth sensor against the training distribution and **changes no estimate**. A
 * user who has not opted in should never meet a screen full of diagnostics, and a build
 * handed to a tester should be the one that carries it.
 *
 * The flag is only half the gate. The other half is whether the hardware exists at all,
 * which the native module answers and this cannot.
 */
export const depthExperimentEnabled = (): boolean => {
  const flag = (process.env.EXPO_PUBLIC_ENABLE_DEPTH ?? "").trim().toLowerCase();
  return flag === "1" || flag === "true";
};

export const MISSING_MODEL_SOURCE =
  "No model is configured for this build. Set EXPO_PUBLIC_MODEL_URL and " +
  "EXPO_PUBLIC_BUNDLE_URL to a published release before building.";
