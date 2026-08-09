import { describe, expect, it } from "vitest";

import { averageNutrition, averageProbabilities } from "../postprocess";
import type { NutritionEstimate } from "../types";
import { buildViews, centreCrop, flipHorizontal } from "../views";

/** A width x height image where each pixel's red channel encodes its x position. */
const ramp = (width: number, height: number) => {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data[(y * width + x) * 3] = x;
    }
  }
  return { data, width, height };
};

const redAt = (image: { data: Uint8Array; width: number }, x: number, y: number) =>
  image.data[(y * image.width + x) * 3];

describe("flipHorizontal", () => {
  it("mirrors the image", () => {
    const flipped = flipHorizontal(ramp(4, 2));
    expect(redAt(flipped, 0, 0)).toBe(3);
    expect(redAt(flipped, 3, 0)).toBe(0);
  });

  it("preserves dimensions and length", () => {
    const flipped = flipHorizontal(ramp(5, 3));
    expect(flipped.width).toBe(5);
    expect(flipped.data.length).toBe(5 * 3 * 3);
  });

  it("is its own inverse", () => {
    const original = ramp(6, 2);
    expect([...flipHorizontal(flipHorizontal(original)).data]).toEqual([...original.data]);
  });

  it("does not mutate its input", () => {
    const original = ramp(4, 1);
    flipHorizontal(original);
    expect(redAt(original, 0, 0)).toBe(0);
  });
});

describe("centreCrop", () => {
  it("keeps the middle", () => {
    const cropped = centreCrop(ramp(10, 10), 0.5);
    expect(cropped.width).toBe(5);
    // Columns 2 through 6 of the original: the crop starts at floor((10-5)/2) = 2.
    expect(redAt(cropped, 0, 0)).toBe(2);
  });

  it("returns the same image at a fraction of one", () => {
    const original = ramp(4, 4);
    expect(centreCrop(original, 1)).toBe(original);
  });

  it("refuses to crop outwards", () => {
    // There are no pixels outside the photograph to include, so a fraction above 1 is not
    // a zoom out, it is a mistake.
    expect(() => centreCrop(ramp(4, 4), 1.5)).toThrow(/crop fraction/);
    expect(() => centreCrop(ramp(4, 4), 0)).toThrow(/crop fraction/);
  });

  it("keeps three bytes per pixel", () => {
    const cropped = centreCrop(ramp(9, 9), 0.5);
    expect(cropped.data.length).toBe(cropped.width * cropped.height * 3);
  });
});

describe("buildViews", () => {
  it("one view is the original untouched", () => {
    const original = ramp(4, 4);
    const views = buildViews(original, 1);
    expect(views).toHaveLength(1);
    expect(views[0]).toBe(original);
  });

  it("adds the flip before the zoom", () => {
    // The measurement supports this ordering: the flip is the cheapest useful view, and the
    // third view carried most of the remaining gain.
    const views = buildViews(ramp(10, 10), 2);
    expect(views[1].width).toBe(10);
    expect(redAt(views[1], 0, 0)).toBe(9);
  });

  it("caps at the number of distinct views available", () => {
    expect(buildViews(ramp(10, 10), 99)).toHaveLength(4);
  });

  it("rejects a nonsense count", () => {
    expect(() => buildViews(ramp(4, 4), 0)).toThrow(/positive integer/);
    expect(() => buildViews(ramp(4, 4), 1.5)).toThrow(/positive integer/);
  });
});

describe("averageProbabilities", () => {
  it("averages in probability space, not logit space", () => {
    // The reason this exists. One confident view and one uncertain view should meet in the
    // middle; averaging logits would let the confident one dominate.
    const confident = [10, 0];
    const uncertain = [0, 0];
    const averaged = averageProbabilities([confident, uncertain]);

    expect(averaged[0]).toBeGreaterThan(0.5);
    expect(averaged[0]).toBeLessThan(0.999);
  });

  it("sums to one", () => {
    const averaged = averageProbabilities([
      [1, 2, 3],
      [3, 2, 1],
    ]);
    expect(averaged.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("a single view is plain softmax", () => {
    const single = averageProbabilities([[1, 2, 3]]);
    expect(single.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
  });

  it("refuses views of differing widths", () => {
    expect(() => averageProbabilities([[1, 2], [1, 2, 3]])).toThrow(/expected/);
  });

  it("refuses no views at all", () => {
    expect(() => averageProbabilities([])).toThrow(/zero views/);
  });
});

describe("averageNutrition", () => {
  const estimate = (energy: number): NutritionEstimate => ({
    energy: { low: energy - 50, median: energy, high: energy + 50 },
    protein: { low: 1, median: 2, high: 3 },
    fat: { low: 1, median: 2, high: 3 },
    carbohydrate: { low: 1, median: 2, high: 3 },
    mass: { low: 100, median: 150, high: 200 },
  });

  it("averages every bound, not only the median", () => {
    // Averaging the median alone would report a central estimate no view made, wrapped in
    // the uncertainty of one that did.
    const averaged = averageNutrition([estimate(100), estimate(200)]);
    expect(averaged.energy.median).toBeCloseTo(150);
    expect(averaged.energy.low).toBeCloseTo(100);
    expect(averaged.energy.high).toBeCloseTo(200);
  });

  it("covers every target", () => {
    const averaged = averageNutrition([estimate(100), estimate(200)]);
    expect(Object.keys(averaged).sort()).toEqual(
      ["carbohydrate", "energy", "fat", "mass", "protein"].sort(),
    );
  });

  it("a single view passes through", () => {
    const only = estimate(100);
    expect(averageNutrition([only])).toBe(only);
  });

  it("refuses no views at all", () => {
    expect(() => averageNutrition([])).toThrow(/zero views/);
  });
});

describe("averaged views keep their confidence", () => {
  /**
   * The bug this exists for produced a *correct dish* at 2% confidence instead of 94%.
   *
   * topK applies softmax internally, so handing it an already-averaged probability
   * distribution softmaxes twice. Over 101 classes that is nearly uniform. The ordering
   * survives, so every ranking assertion passes and only the number is wrong, which is why
   * it was invisible until a browser showed the figure to a human.
   */
  it("does not softmax an already-normalised distribution", async () => {
    const { topK, topKFromProbabilities, softmax } = await import("../postprocess");
    const { classCount } = await import("../contract");

    const logits = new Float64Array(classCount);
    logits[7] = 10;

    const fromLogits = topK(logits, 1)[0].confidence;
    const fromProbabilities = topKFromProbabilities(softmax(logits), 1)[0].confidence;

    expect(fromProbabilities).toBeCloseTo(fromLogits, 6);
    expect(fromProbabilities).toBeGreaterThan(0.9);
  });

  it("averaging views that agree preserves a confident answer", async () => {
    const { averageProbabilities, topKFromProbabilities } = await import("../postprocess");
    const { classCount } = await import("../contract");

    const confident = new Float64Array(classCount);
    confident[7] = 10;

    const averaged = averageProbabilities([confident, confident, confident]);
    const top = topKFromProbabilities(averaged, 1)[0];

    // Three views agreeing must not dilute the answer towards uniform.
    expect(top.confidence).toBeGreaterThan(0.9);
  });
});
