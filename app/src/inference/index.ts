/**
 * Inference boundary.
 *
 * The real implementation loads the exported ONNX graph through
 * onnxruntime-react-native and lands in PR #14. Until then this returns a fixed plausible
 * result so the screens can be built and reviewed against something.
 *
 * The stub reports `placeholder: true` and the UI says so on screen. A demo that silently
 * shows invented numbers is how a portfolio project ends up claiming something it cannot
 * do, and the failure is invisible precisely because the numbers look reasonable.
 */

import type { Analysis } from "../types";
import { displayName } from "../labels";

/** Roughly a plate of carbonara, so the screens are exercised with believable magnitudes. */
const PLACEHOLDER_ENERGY = { low: 410, median: 620, high: 980 };

export const analyse = async (photoUri: string): Promise<Analysis> => {
  const started = Date.now();

  // Stands in for the model call. No artificial delay: pretending to be slow would make
  // the eventual real latency look like a regression.
  const dishes = [
    { key: "spaghetti_carbonara", confidence: 0.71 },
    { key: "spaghetti_bolognese", confidence: 0.14 },
    { key: "pad_thai", confidence: 0.05 },
  ].map((entry) => ({ ...entry, label: displayName(entry.key) }));

  return {
    photoUri,
    dishes,
    nutrition: {
      energy: PLACEHOLDER_ENERGY,
      protein: { low: 14, median: 24, high: 38 },
      fat: { low: 16, median: 30, high: 52 },
      carbohydrate: { low: 48, median: 71, high: 104 },
      mass: { low: 210, median: 320, high: 470 },
    },
    inferenceMs: Date.now() - started,
    placeholder: true,
  };
};
