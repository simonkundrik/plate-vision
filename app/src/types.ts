/** Shapes shared between the inference layer and the screens. */

export type DishPrediction = {
  /** Food-101 class key, e.g. "spaghetti_carbonara". */
  key: string;
  /** Human-readable name for display. */
  label: string;
  /** Softmax probability in [0, 1]. */
  confidence: number;
};

/**
 * One nutrition target as an interval rather than a number.
 *
 * The three values are the model's 5th, 50th, and 95th percentile predictions. Storing
 * them as a triple rather than collapsing to `median` is deliberate: portion size cannot
 * be recovered from a photograph, so a single figure would be false precision, and the
 * interface is built to show the range as the primary reading.
 */
export type Interval = {
  low: number;
  median: number;
  high: number;
};

export type NutritionEstimate = {
  energy: Interval;
  protein: Interval;
  fat: Interval;
  carbohydrate: Interval;
  mass: Interval;
};

export type Analysis = {
  photoUri: string;
  dishes: DishPrediction[];
  nutrition: NutritionEstimate;
  /** Milliseconds spent in the model, shown so the on-device claim is checkable. */
  inferenceMs: number;
  /** True while the real model is not yet wired in. Surfaced in the UI, not hidden. */
  placeholder: boolean;
};

/** Scale every interval by a portion multiplier the user picked. */
export const scaleInterval = (interval: Interval, factor: number): Interval => ({
  low: interval.low * factor,
  median: interval.median * factor,
  high: interval.high * factor,
});
