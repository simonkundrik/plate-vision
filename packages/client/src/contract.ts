/**
 * The model contract, shipped inside the package.
 *
 * These are copies of `shared/` synced at build time. A test asserts they still match the
 * originals, which is what keeps them from drifting: a label ordering that has moved does
 * not crash anything, it renames every prediction.
 */

import labelsJson from "./generated/food101_labels.json" with { type: "json" };
import metaJson from "./generated/model_meta.json" with { type: "json" };

type LabelEntry = { index: number; key: string; display: string };

const entries = labelsJson.labels as LabelEntry[];
const byIndex = new Map(entries.map((entry) => [entry.index, entry]));

export const contract = metaJson;
export const labels = entries;
export const classCount = entries.length;

/** Input tensor name the graph expects. */
export const inputName: string = metaJson.input.name;

/** Output tensor names, in contract order. */
export const outputNames: string[] = Object.keys(metaJson.outputs);

/** Nutrition target keys, in the order they appear on output axis 1. */
export const targetKeys: string[] = metaJson.outputs.nutrition_quantiles.targets.map(
  (target: { key: string }) => target.key,
);

/** Quantile levels, in the order they appear on output axis 2. */
export const quantiles: number[] = metaJson.outputs.nutrition_quantiles.quantiles;

/** Index of the median quantile, used for the central estimate. */
export const medianIndex: number = quantiles.indexOf(0.5);

export const keyForIndex = (index: number): string => byIndex.get(index)?.key ?? "";
export const labelForIndex = (index: number): string => byIndex.get(index)?.display ?? "";

/**
 * Guard against a bundle that was produced for a different contract.
 *
 * Loading a model whose head width disagrees with the label list produces predictions
 * mapped to the wrong names, which looks like a badly trained model rather than a
 * mismatched artifact.
 */
export const assertContractMatches = (logitsLength: number): void => {
  if (logitsLength !== classCount) {
    throw new Error(
      `model returns ${logitsLength} classes but the bundled label list has ${classCount}. ` +
        "The model artifact and this package version do not match.",
    );
  }
};
