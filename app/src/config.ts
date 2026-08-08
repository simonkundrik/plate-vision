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

export const MISSING_MODEL_SOURCE =
  "No model is configured for this build. Set EXPO_PUBLIC_MODEL_URL and " +
  "EXPO_PUBLIC_BUNDLE_URL to a published release before building.";
