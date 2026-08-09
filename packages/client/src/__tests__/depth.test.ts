import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  compareToTraining,
  describeDepth,
  dropoutFraction,
  heightAbove,
  MAX_DEPTH_MM,
  normaliseDepth,
  TRAINING_DEPTH,
  type DepthMap,
} from "../depth";

const map = (values: number[], width = values.length): DepthMap => ({
  data: Uint16Array.from(values),
  width,
  height: values.length / width,
});

describe("normaliseDepth", () => {
  it("scales millimetres into [0, 1] against the same ceiling as depth.py", () => {
    const out = normaliseDepth(map([0, 3000, 6000], 3));
    // The zero is a dropout and is filled, so only the two real values are checked here.
    expect(out[1]).toBeCloseTo(0.5, 6);
    expect(out[2]).toBeCloseTo(1.0, 6);
  });

  it("fills dropouts with the median of the valid pixels, not with zero", () => {
    // Zero would read as "against the lens" and the maximum as "infinitely far". Both are
    // strong, wrong signals where the sensor had nothing to say.
    const out = normaliseDepth(map([0, 1000, 2000, 3000], 4));
    expect(out[0]).toBeCloseTo(2000 / MAX_DEPTH_MM, 6);
  });

  it("takes the median the way numpy does, averaging the middle pair", () => {
    const out = normaliseDepth(map([0, 1000, 2000, 3000, 4000], 5));
    expect(out[0]).toBeCloseTo(2500 / MAX_DEPTH_MM, 6);
  });

  it("returns the midpoint when the sensor reported nothing at all", () => {
    const out = normaliseDepth(map([0, 0, 0], 3));
    expect(Array.from(out)).toEqual([0.5, 0.5, 0.5]);
  });

  it("clips saturated pixels rather than letting one dominate the frame", () => {
    // A single 65535 would otherwise compress every real value towards zero.
    const out = normaliseDepth(map([65535, 3000, 3200], 3));
    expect(out[0]).toBeCloseTo(1.0, 6);
    expect(out[1]).toBeCloseTo(0.5, 6);
  });

  it("refuses a map whose data does not match its stated shape", () => {
    expect(() => normaliseDepth({ data: Uint16Array.from([1, 2, 3]), width: 2, height: 2 })).toThrow(
      /2x2 but carries 3 values/,
    );
  });
});

describe("dropoutFraction", () => {
  it("counts only exact zeros", () => {
    expect(dropoutFraction(map([0, 0, 1, 1], 4))).toBeCloseTo(0.5, 6);
    expect(dropoutFraction(map([1, 2, 3, 4], 4))).toBe(0);
  });
});

describe("heightAbove", () => {
  it("measures upwards from a high percentile of the valid depths", () => {
    // Surface at the 90th percentile of [1000, 2000, 3000] is 2800, so the closest pixel
    // stands 1800 above it and the furthest is clipped to zero rather than going negative.
    const out = heightAbove(map([1000, 2000, 3000], 3));
    expect(out[0]).toBeCloseTo(1800 / MAX_DEPTH_MM, 6);
    expect(out[1]).toBeCloseTo(800 / MAX_DEPTH_MM, 6);
    expect(out[2]).toBe(0);
  });

  it("leaves dropouts flat rather than inventing a height for them", () => {
    const out = heightAbove(map([0, 1000, 3000], 3));
    expect(out[0]).toBe(0);
  });

  it("returns a flat map when nothing was reported", () => {
    expect(Array.from(heightAbove(map([0, 0], 2)))).toEqual([0, 0]);
  });
});

describe("describeDepth", () => {
  const rig = map([3400, 3500, 3600, 3700, 0, 3562], 6);

  it("reports the shape and how much of it the sensor filled", () => {
    const d = describeDepth(rig);
    expect(d.pixels).toBe(6);
    expect(d.validPixels).toBe(5);
    expect(d.dropoutFraction).toBeCloseTo(1 / 6, 6);
    expect(d.saturatedFraction).toBe(0);
  });

  it("reports the value a four-channel model would actually receive", () => {
    const d = describeDepth(rig);
    expect(d.normalisedMedian).toBeCloseTo(d.medianMm / MAX_DEPTH_MM, 9);
  });

  it("puts a rig-distance capture within three standard deviations of training", () => {
    expect(Math.abs(describeDepth(rig).sigmaFromTraining)).toBeLessThan(3);
  });

  it("puts a phone-distance capture roughly eleven standard deviations away", () => {
    // The prediction this whole feature exists to check: a phone at 300 mm normalises to
    // 0.05 against a training mean of 0.612 with a standard deviation of 0.052.
    const phone = describeDepth(map([290, 300, 310], 3));
    expect(phone.sigmaFromTraining).toBeLessThan(-10);
    expect(phone.sigmaFromTraining).toBeGreaterThan(-12);
  });

  it("measures relief from the surface down to the closest pixel", () => {
    // Food standing 40 mm proud of a table 300 mm from the lens.
    const d = describeDepth(map([260, 300, 300, 300], 4));
    expect(d.reliefMm).toBeCloseTo(40, 6);
  });
});

describe("compareToTraining", () => {
  it("passes a rig capture on distance and fails a phone capture on it", () => {
    const rig = compareToTraining(describeDepth(map([3500, 3562, 3600], 3)));
    const phone = compareToTraining(describeDepth(map([290, 300, 310], 3)));
    const distanceOf = (checks: ReturnType<typeof compareToTraining>) =>
      checks.find((c) => c.name === "Camera distance");

    expect(distanceOf(rig)?.inDistribution).toBe(true);
    expect(distanceOf(phone)?.inDistribution).toBe(false);
  });

  it("says on the failing distance check that the failure is expected", () => {
    // A tester seeing red has to be able to tell a broken capture from the predicted
    // result, or good captures get thrown away for looking wrong.
    const phone = compareToTraining(describeDepth(map([290, 300, 310], 3)));
    expect(phone.find((c) => c.name === "Camera distance")?.note).toMatch(/expected to be out of range/);
  });

  it("holds dropouts to the rig's worst map rather than to its average", () => {
    const noisy = describeDepth(map(new Array(10).fill(0).map((_, i) => (i < 5 ? 0 : 300)), 10));
    expect(noisy.dropoutFraction).toBeCloseTo(0.5, 6);
    const check = compareToTraining(noisy).find((c) => c.name === "Sensor dropouts");
    expect(check?.inDistribution).toBe(false);
    expect(check?.expected).toContain(`${(TRAINING_DEPTH.worstDropout * 100).toFixed(1)}%`);
  });

  it("reports every check whether or not it passed", () => {
    const checks = compareToTraining(describeDepth(map([290, 300, 310], 3)));
    expect(checks).toHaveLength(4);
    expect(checks.every((c) => c.note.length > 0)).toBe(true);
  });
});

/**
 * The tests above check this module against itself, which cannot catch a mirror that is
 * wrong in the same way twice. `depth-fixture.json` is generated by running the real
 * `platevision.depth` on these maps, so a divergence between the two implementations fails
 * here rather than showing up as a model fed inputs it was never trained on.
 *
 * Regenerate with `model/scripts/emit_depth_fixture.py`.
 */
describe("parity with platevision.depth", () => {
  type Case = {
    name: string;
    width: number;
    height: number;
    data: number[];
    normalised: number[];
    heightAbove: number[];
    dropoutFraction: number;
  };

  const cases: Case[] = JSON.parse(
    readFileSync(fileURLToPath(new URL("./depth-fixture.json", import.meta.url)), "utf-8"),
  );

  it("covers the rig, the phone, and both sensor failures", () => {
    expect(cases.map((c) => c.name)).toEqual(["rig", "phone", "broken", "empty"]);
  });

  for (const c of cases) {
    it(`matches Python on the ${c.name} map`, () => {
      const subject: DepthMap = {
        data: Uint16Array.from(c.data),
        width: c.width,
        height: c.height,
      };

      expect(dropoutFraction(subject)).toBeCloseTo(c.dropoutFraction, 9);

      const normalised = Array.from(normaliseDepth(subject));
      const heights = Array.from(heightAbove(subject));
      expect(normalised).toHaveLength(c.normalised.length);

      for (let i = 0; i < c.normalised.length; i += 1) {
        expect(normalised[i]).toBeCloseTo(c.normalised[i], 6);
        expect(heights[i]).toBeCloseTo(c.heightAbove[i], 6);
      }
    });
  }
});
