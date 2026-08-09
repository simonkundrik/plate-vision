/**
 * Turning a depth capture into something a tester can send back.
 *
 * The whole point of shipping this feature unverified is that the answers live on hardware
 * nobody working on it owns. That only pays off if reporting a result is one tap, and if
 * what gets reported is the numbers rather than a description of them.
 *
 * Two things deliberately do **not** go into a report: the photograph, and the depth map
 * itself. The app's standing promise is that the picture is analysed on the device and is
 * not uploaded, and a feature that quietly starts posting images to a public issue tracker
 * would break that promise in the worst possible place. What travels is a dozen summary
 * statistics, and the tester sees all of them before anything opens.
 *
 * Pure functions. Nothing here touches a native module, so it is all unit tested.
 */

import type { DepthDescription, DistributionCheck } from "@plate-vision/client/depth";

export const ISSUE_BASE = "https://github.com/simonkundrik/plate-vision/issues/new";
export const ISSUE_TEMPLATE = "depth-capture-report.yml";

export type CaptureContext = {
  sensor: string;
  filtered: boolean;
  accuracy: "absolute" | "relative";
  /** Platform string, e.g. `ios 18.2`. Not the device name, which users personalise. */
  platform: string;
  /** App version, so a report can be tied to the code that produced it. */
  appVersion: string;
};

const percent = (value: number): string => `${(value * 100).toFixed(1)}%`;
const mm = (value: number): string => (Number.isNaN(value) ? "n/a" : `${Math.round(value)} mm`);

/**
 * The measurement block, as markdown.
 *
 * Rendered on screen before it is sent, and pasted into the issue unchanged. One function
 * for both, so a tester cannot approve one thing and file another.
 */
export const formatMeasurements = (
  description: DepthDescription,
  checks: DistributionCheck[],
  context: CaptureContext,
): string => {
  const lines = [
    `platform:      ${context.platform}`,
    `app:           ${context.appVersion}`,
    `sensor:        ${context.sensor || "unknown"}`,
    `accuracy:      ${context.accuracy}`,
    `filtered:      ${context.filtered}`,
    "",
    `resolution:    ${description.width} x ${description.height}`,
    `valid pixels:  ${description.validPixels} of ${description.pixels}`,
    `dropouts:      ${percent(description.dropoutFraction)}`,
    `saturated:     ${percent(description.saturatedFraction)}`,
    "",
    `depth p5:      ${mm(description.p5Mm)}`,
    `depth median:  ${mm(description.medianMm)}`,
    `depth p95:     ${mm(description.p95Mm)}`,
    `relief:        ${mm(description.reliefMm)}`,
    "",
    `normalised:    ${description.normalisedMedian.toFixed(4)}`,
    `sigma:         ${description.sigmaFromTraining.toFixed(1)} from the training mean`,
    "",
    "checks:",
    ...checks.map((c) => `  ${c.inDistribution ? "ok  " : "out "} ${c.name}: ${c.value}`),
  ];

  return lines.join("\n");
};

/**
 * A prefilled issue URL for GitHub's issue forms.
 *
 * Form fields are prefilled by query parameters named after their `id` in
 * `.github/ISSUE_TEMPLATE/depth-capture-report.yml`. The two files are a matched pair, and
 * a rename on either side silently produces an empty form, so the ids are pulled out here
 * rather than being spelled inline twice.
 */
export const ISSUE_FIELDS = {
  measurements: "measurements",
  happened: "happened",
  device: "device",
} as const;

export const buildIssueUrl = (measurements: string, title?: string): string => {
  const query = new URLSearchParams({
    template: ISSUE_TEMPLATE,
    labels: "depth-capture",
    title: title ?? "Depth capture report",
    [ISSUE_FIELDS.measurements]: measurements,
  });
  return `${ISSUE_BASE}?${query.toString()}`;
};
