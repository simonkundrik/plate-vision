import { describe, expect, it } from "vitest";

import { scaleInterval, scaleNutrition } from "../portion";
import type { Interval, NutritionEstimate } from "../types";

const interval = (low: number, median: number, high: number): Interval => ({
  low,
  median,
  high,
});

const estimate = (): NutritionEstimate => ({
  energy: interval(400, 620, 980),
  protein: interval(14, 24, 38),
  fat: interval(16, 30, 52),
  carbohydrate: interval(48, 71, 104),
  mass: interval(210, 320, 470),
  route: "absolute",
});

describe("scaleInterval", () => {
  it("scales all three bounds", () => {
    expect(scaleInterval(interval(100, 200, 400), 0.5)).toEqual({
      low: 50,
      median: 100,
      high: 200,
    });
  });

  it("preserves relative width", () => {
    // The point of scaling all three bounds rather than the median alone: a corrected
    // portion is not a more certain one, and narrowing the interval on correction would
    // claim confidence the model never had.
    const original = interval(100, 200, 400);
    const scaled = scaleInterval(original, 2.5);
    expect(scaled.high / scaled.low).toBeCloseTo(original.high / original.low);
  });

  it("is identity at a factor of one", () => {
    const original = interval(1, 2, 3);
    expect(scaleInterval(original, 1)).toEqual(original);
  });

  it("does not mutate its input", () => {
    const original = interval(100, 200, 400);
    scaleInterval(original, 3);
    expect(original).toEqual({ low: 100, median: 200, high: 400 });
  });
});

describe("scaleNutrition", () => {
  it("scales every target", () => {
    const scaled = scaleNutrition(estimate(), 0.5);
    expect(scaled.energy.median).toBe(310);
    expect(scaled.protein.median).toBe(12);
    expect(scaled.mass.high).toBe(235);
  });

  it("leaves the original untouched", () => {
    const original = estimate();
    scaleNutrition(original, 2);
    expect(original.energy.median).toBe(620);
  });
});
