import { describe, expect, it } from "vitest";

import { classCount, medianIndex, quantiles, targetKeys } from "../contract";
import { enforceMonotonic, inverseTransform, softmax, toNutrition, topK } from "../postprocess";
import type { TargetTransform } from "../types";

const logitsFor = (peaks: Record<number, number>): Float32Array => {
  const logits = new Float32Array(classCount);
  for (const [index, value] of Object.entries(peaks)) logits[Number(index)] = value;
  return logits;
};

describe("softmax", () => {
  it("sums to one", () => {
    const out = softmax([1, 2, 3, 4]);
    expect([...out].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it("survives logits large enough to overflow a naive implementation", () => {
    // Math.exp(1000) is Infinity. Subtracting the maximum first is the only reason this
    // returns probabilities rather than NaN.
    const out = softmax([1000, 999, 998]);
    expect([...out].every(Number.isFinite)).toBe(true);
    expect([...out].reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(out[0]).toBeGreaterThan(out[1]);
  });

  it("is uniform for equal logits", () => {
    const out = softmax([2, 2, 2, 2]);
    expect([...out]).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});

describe("topK", () => {
  it("returns candidates most confident first", () => {
    const result = topK(logitsFor({ 5: 10, 9: 8, 20: 6 }), 3);
    expect(result.map((entry) => entry.key)).toHaveLength(3);
    expect(result[0].confidence).toBeGreaterThan(result[1].confidence);
    expect(result[1].confidence).toBeGreaterThan(result[2].confidence);
  });

  it("resolves keys and display names from the bundled contract", () => {
    const [best] = topK(logitsFor({ 0: 10 }), 1);
    expect(best.key).toBe("apple_pie");
    expect(best.label).toBe("Apple pie");
  });

  it("rejects a model whose head width disagrees with the label list", () => {
    // A mismatched artifact otherwise produces predictions mapped to the wrong names,
    // which reads as a badly trained model rather than the wrong file.
    expect(() => topK(new Float32Array(50), 3)).toThrow(/do not match/);
  });

  it("never returns more candidates than exist", () => {
    expect(topK(logitsFor({ 1: 5 }), 500)).toHaveLength(classCount);
  });
});

describe("enforceMonotonic", () => {
  it("sorts a crossed interval", () => {
    // Nothing in quantile regression couples the outputs, so a model can predict its 5th
    // percentile above its 95th. That is a negative-width interval.
    expect(enforceMonotonic([9, 5, 1])).toEqual([1, 5, 9]);
  });

  it("leaves an ordered interval alone", () => {
    expect(enforceMonotonic([1, 5, 9])).toEqual([1, 5, 9]);
  });

  it("does not mutate its input", () => {
    const input = [9, 5, 1];
    enforceMonotonic(input);
    expect(input).toEqual([9, 5, 1]);
  });
});

describe("inverseTransform", () => {
  const transform: TargetTransform = { mean: [5, 0], std: [1, 1], keys: ["energy", "mass"] };

  it("mirrors expm1(value * std + mean)", () => {
    expect(inverseTransform(0, transform, 0)).toBeCloseTo(Math.expm1(5), 10);
  });

  it("is monotonically increasing, which is what preserves the quantiles", () => {
    const low = inverseTransform(-1.5, transform, 0);
    const mid = inverseTransform(0, transform, 0);
    const high = inverseTransform(1.5, transform, 0);
    expect(low).toBeLessThan(mid);
    expect(mid).toBeLessThan(high);
  });

  it("maps a standardised zero back through the per-target mean", () => {
    expect(inverseTransform(0, transform, 1)).toBeCloseTo(Math.expm1(0), 10);
  });
});

describe("toNutrition", () => {
  const transform: TargetTransform = {
    mean: [5.09, 2.32, 2.05, 2.64, 5.1],
    std: [1.1, 1.23, 1.19, 0.93, 0.82],
    keys: targetKeys,
  };

  const flat = (perTarget: number[]): Float32Array => {
    const out = new Float32Array(targetKeys.length * quantiles.length);
    targetKeys.forEach((_, target) => {
      perTarget.forEach((value, q) => {
        out[target * quantiles.length + q] = value;
      });
    });
    return out;
  };

  it("produces one interval per contract target", () => {
    const nutrition = toNutrition(flat([-1.5, 0, 1.5]), transform, quantiles.length);
    expect(Object.keys(nutrition).sort()).toEqual([...targetKeys, "route"].sort());
  });

  it("labels the photograph-only route", () => {
    // An estimate that reaches an integrator without stating where it came from cannot be
    // reported honestly, because the routes have error distributions nothing alike.
    const nutrition = toNutrition(flat([-1.5, 0, 1.5]), transform, quantiles.length);
    expect(nutrition.route).toBe("absolute");
  });

  it("orders low, median, high", () => {
    const { energy } = toNutrition(flat([-1.5, 0, 1.5]), transform, quantiles.length);
    expect(energy.low).toBeLessThan(energy.median);
    expect(energy.median).toBeLessThan(energy.high);
  });

  it("repairs a crossed interval rather than emitting a negative width", () => {
    const { energy } = toNutrition(flat([1.5, 0, -1.5]), transform, quantiles.length);
    expect(energy.high).toBeGreaterThan(energy.low);
  });

  it("returns plausible kilocalories for a standardised median of zero", () => {
    const { energy } = toNutrition(flat([-1, 0, 1]), transform, quantiles.length);
    // expm1(5.09) is about 162, which is the right order of magnitude for a plate.
    expect(energy.median).toBeGreaterThan(50);
    expect(energy.median).toBeLessThan(500);
  });

  it("uses the contract's median index rather than assuming the middle", () => {
    expect(medianIndex).toBe(quantiles.indexOf(0.5));
  });
});
