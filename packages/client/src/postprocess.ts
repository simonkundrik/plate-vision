/** Turning raw model output into the typed result an integrator receives. */

import { assertContractMatches, keyForIndex, labelForIndex, medianIndex, targetKeys } from "./contract";
import type { DishPrediction, Interval, NutritionEstimate, TargetTransform } from "./types";

/**
 * Numerically stable softmax.
 *
 * The maximum is subtracted before exponentiating. Without that, a logit around 90 or
 * above overflows to Infinity in float64 and every probability comes back NaN.
 */
export const softmax = (logits: ArrayLike<number>): Float64Array => {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i += 1) if (logits[i] > max) max = logits[i];

  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - max);
    out[i] = value;
    sum += value;
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= sum;
  return out;
};

/** The `k` highest-probability classes, most confident first. */
export const topK = (logits: ArrayLike<number>, k: number): DishPrediction[] => {
  assertContractMatches(logits.length);
  const probabilities = softmax(logits);

  const order = Array.from({ length: probabilities.length }, (_, index) => index).sort(
    (a, b) => probabilities[b] - probabilities[a],
  );

  return order.slice(0, Math.max(0, k)).map((index) => ({
    key: keyForIndex(index),
    label: labelForIndex(index),
    confidence: probabilities[index],
  }));
};

/**
 * Sort a quantile triple so the interval bounds cannot cross.
 *
 * Nothing in quantile regression couples the three outputs, so a model can predict a 5th
 * percentile above its 95th on unfamiliar input. That is a negative-width interval, and
 * anything computed from it, including a coverage claim, is meaningless.
 */
export const enforceMonotonic = (values: number[]): number[] => [...values].sort((a, b) => a - b);

/**
 * Undo the training-time transform: standardised log space back to real units.
 *
 * Mirrors `expm1(value * std + mean)` on the Python side. Quantiles survive this because
 * it is monotonically increasing, which is the property that lets the head train in log
 * space at all.
 */
export const inverseTransform = (
  value: number,
  transform: TargetTransform,
  targetIndex: number,
): number => Math.expm1(value * transform.std[targetIndex] + transform.mean[targetIndex]);

/** Build the nutrition estimate from a flat (targets x quantiles) output. */
export const toNutrition = (
  raw: ArrayLike<number>,
  transform: TargetTransform,
  quantileCount: number,
): NutritionEstimate => {
  const result: Record<string, Interval> = {};

  targetKeys.forEach((key, targetIndex) => {
    const offset = targetIndex * quantileCount;
    const slice: number[] = [];
    for (let q = 0; q < quantileCount; q += 1) {
      slice.push(inverseTransform(raw[offset + q], transform, targetIndex));
    }
    const ordered = enforceMonotonic(slice);
    result[key] = {
      low: ordered[0],
      median: ordered[medianIndex],
      high: ordered[ordered.length - 1],
    };
  });

  return result as unknown as NutritionEstimate;
};
