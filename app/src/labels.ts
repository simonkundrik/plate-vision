/**
 * Class label lookup, read from the shared contract.
 *
 * Metro is configured to watch ../shared, so this imports the same file the exporter and
 * the training code read. Copying the label list into the app would create a second
 * ordering that can drift, and a drifted ordering does not crash: it renames every
 * prediction.
 */

import labels from "../../shared/food101_labels.json";

type LabelEntry = { index: number; key: string; display: string };

const entries = labels.labels as LabelEntry[];

const byKey = new Map(entries.map((entry) => [entry.key, entry]));

export const classCount = entries.length;

/** Human-readable name for a class key, falling back to the key itself. */
export const displayName = (key: string): string => byKey.get(key)?.display ?? key;

/** Class key for a logits index, used once real model output arrives. */
export const keyForIndex = (index: number): string => entries[index]?.key ?? "";

/** Human-readable name for a logits index. */
export const labelForIndex = (index: number): string => entries[index]?.display ?? "";
