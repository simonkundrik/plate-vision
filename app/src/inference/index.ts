/**
 * Inference boundary.
 *
 * The real implementation loads the exported ONNX graph through `@plate-vision/client` and
 * lands in the next PR. Until then this returns a fixed plausible result so the screens can
 * be built and reviewed against something.
 *
 * The stub reports `placeholder: true` and the UI says so on screen. A demo that silently
 * shows invented numbers is how a portfolio project ends up claiming something it cannot
 * do, and the failure is invisible precisely because the numbers look reasonable.
 *
 * It returns the library's own `Analysis` shape rather than a parallel one, so swapping the
 * stub for a real `PlateVision.analyse` call changes what produces the value and nothing
 * that consumes it.
 */

import { labelForIndex, keyForIndex, classCount } from "@plate-vision/client";

import type { MealAnalysis } from "../types";

/** Roughly a plate of carbonara, so the screens are exercised with believable magnitudes. */
const PLACEHOLDER_ENERGY = { low: 410, median: 620, high: 980 };

/** Resolve a class key to its contract index so the stub speaks in the model's terms. */
const indexOfKey = (key: string): number => {
  for (let index = 0; index < classCount; index += 1) {
    if (keyForIndex(index) === key) return index;
  }
  throw new Error(`${key} is not in the label contract`);
};

const PLACEHOLDER_DISHES = [
  { key: "spaghetti_carbonara", confidence: 0.71 },
  { key: "spaghetti_bolognese", confidence: 0.14 },
  { key: "pad_thai", confidence: 0.05 },
].map(({ key, confidence }) => ({
  key,
  label: labelForIndex(indexOfKey(key)),
  confidence,
}));

export const analyse = async (photoUri: string): Promise<MealAnalysis> => {
  const started = Date.now();

  // Stands in for the model call. No artificial delay: pretending to be slow would make
  // the eventual real latency look like a regression.
  return {
    photoUri,
    dishes: PLACEHOLDER_DISHES,
    nutrition: {
      energy: PLACEHOLDER_ENERGY,
      protein: { low: 14, median: 24, high: 38 },
      fat: { low: 16, median: 30, high: 52 },
      carbohydrate: { low: 48, median: 71, high: 104 },
      mass: { low: 210, median: 320, high: 470 },
    },
    nutritionUnavailableReason: null,
    inferenceMs: Date.now() - started,
    placeholder: true,
  };
};
