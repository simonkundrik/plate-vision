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

/** The `k` highest-probability classes, most confident first. Takes raw logits. */
export const topK = (logits: ArrayLike<number>, k: number): DishPrediction[] =>
  topKFromProbabilities(softmax(logits), k);

/**
 * The same ranking, for input that is already a probability distribution.
 *
 * Exists because passing probabilities to `topK` silently produces nonsense rather than an
 * error: softmax of a softmax is nearly uniform over 101 classes, so the ordering survives
 * and only the confidence is destroyed. Averaging views produced a correct dish at 2%
 * confidence instead of 94%, and every ordering assertion still passed.
 */
export const topKFromProbabilities = (
  probabilities: ArrayLike<number>,
  k: number,
): DishPrediction[] => {
  assertContractMatches(probabilities.length);

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

/**
 * Average softmax probabilities across views.
 *
 * Probabilities rather than logits, because logits from different views sit on no common
 * scale: averaging them lets whichever view happened to be most confident dominate, which
 * is the opposite of what averaging is for.
 */
export const averageProbabilities = (views: ArrayLike<number>[]): Float64Array => {
  if (views.length === 0) throw new Error("cannot average zero views");

  const first = softmax(views[0]);
  if (views.length === 1) return first;

  const summed = first;
  for (let v = 1; v < views.length; v += 1) {
    const probabilities = softmax(views[v]);
    if (probabilities.length !== summed.length) {
      throw new Error(
        `view ${v} returned ${probabilities.length} classes, expected ${summed.length}`,
      );
    }
    for (let i = 0; i < summed.length; i += 1) summed[i] += probabilities[i];
  }

  for (let i = 0; i < summed.length; i += 1) summed[i] /= views.length;
  return summed;
};

/**
 * Average nutrition estimates across views, in real units.
 *
 * The bounds are averaged along with the median. Averaging only the median and keeping one
 * view's interval would report a central estimate no single view made, wrapped in the
 * uncertainty of one that did.
 */
export const averageNutrition = (views: NutritionEstimate[]): NutritionEstimate => {
  if (views.length === 0) throw new Error("cannot average zero views");
  if (views.length === 1) return views[0];

  const keys = Object.keys(views[0]) as (keyof NutritionEstimate)[];
  const result = {} as NutritionEstimate;

  for (const key of keys) {
    const mean = (side: keyof Interval) =>
      views.reduce((total, view) => total + view[key][side], 0) / views.length;
    result[key] = { low: mean("low"), median: mean("median"), high: mean("high") };
  }

  return result;
};
