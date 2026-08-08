/**
 * App-level shapes.
 *
 * The result shape itself comes from `@plate-vision/client`, which is the package this app
 * exists to demonstrate. Redeclaring it here would create a second definition that drifts
 * from the library the app actually calls, so the app only adds what is genuinely its own:
 * the photo it captured, and whether the numbers came from a real model.
 */

import type { Analysis } from "@plate-vision/client";

export type {
  Analysis,
  DishPrediction,
  Interval,
  NutritionEstimate,
} from "@plate-vision/client";

export { scaleInterval } from "@plate-vision/client";

/** A library result plus the things only the app knows about it. */
export type MealAnalysis = Analysis & {
  /** Local URI of the captured photo. */
  photoUri: string;
  /** True while the real model is not yet wired in. Surfaced in the UI, not hidden. */
  placeholder: boolean;
};
