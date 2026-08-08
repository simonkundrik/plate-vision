/**
 * Portion correction.
 *
 * The model sees a photograph, and a photograph carries no scale reference. Portion size is
 * where nearly all calorie error lives, so letting a user say "that was half of what you
 * assumed" is not a nicety, it is the only correction signal available at inference time.
 *
 * Scaling is linear and applied to the whole interval, which keeps the reported uncertainty
 * proportional rather than pretending a corrected portion is a more certain one.
 */

import type { Interval, NutritionEstimate } from "./types";

/** Multiply an interval by a portion factor, preserving its relative width. */
export const scaleInterval = (interval: Interval, factor: number): Interval => ({
  low: interval.low * factor,
  median: interval.median * factor,
  high: interval.high * factor,
});

/** Apply a portion factor to every target in an estimate. */
export const scaleNutrition = (
  nutrition: NutritionEstimate,
  factor: number,
): NutritionEstimate => ({
  energy: scaleInterval(nutrition.energy, factor),
  protein: scaleInterval(nutrition.protein, factor),
  fat: scaleInterval(nutrition.fat, factor),
  carbohydrate: scaleInterval(nutrition.carbohydrate, factor),
  mass: scaleInterval(nutrition.mass, factor),
});
