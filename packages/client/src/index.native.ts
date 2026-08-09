/**
 * React Native entry point. Backed by `onnxruntime-react-native`.
 *
 * Resolved through the `react-native` condition in package.json exports, so integrators
 * import the same specifier on both platforms and get the right runtime.
 *
 * There is no canvas here, so image decoding is left to the caller. `expo-image-manipulator`
 * plus a JPEG decoder is the usual route; the model accepts any resolution, so the only
 * reason to resize first is decode cost.
 */

import { PlateVision } from "./plate-vision";
import type { LoadOptions } from "./types";

export { PlateVision } from "./plate-vision";
export { classCount, contract, labels, labelForIndex, keyForIndex } from "./contract";
export { artifactOf, parseBundle, SUPPORTED_SCHEMA_VERSIONS } from "./bundle";
export type { ArtifactDescriptor } from "./bundle";
export { averageNutrition, averageProbabilities, enforceMonotonic, softmax, topK } from "./postprocess";
export { scaleInterval, scaleNutrition } from "./portion";
export { buildViews, centreCrop, flipHorizontal, DEFAULT_VIEWS } from "./views";
export { stripAlpha } from "./session";
export type {
  Analysis,
  AnalyseOptions,
  DishPrediction,
  EvidenceRoute,
  Interval,
  LoadOptions,
  ModelBundle,
  NutritionEstimate,
  RgbImage,
  TargetTransform,
} from "./types";

/** Load a model using `onnxruntime-react-native`. */
export const load = async (options: LoadOptions): Promise<PlateVision> => {
  const ort = await import("onnxruntime-react-native");
  return PlateVision.load(ort as never, options);
};
