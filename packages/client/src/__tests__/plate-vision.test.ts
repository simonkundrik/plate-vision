import { describe, expect, it, vi } from "vitest";

import { classCount, quantiles, targetKeys } from "../contract";
import { PlateVision } from "../plate-vision";
import { stripAlpha } from "../session";
import type { ModelBundle, RgbImage, TargetTransform } from "../types";

const transform: TargetTransform = {
  mean: [5.09, 2.32, 2.05, 2.64, 5.1],
  std: [1.1, 1.23, 1.19, 0.93, 0.82],
  keys: targetKeys,
};

const trainedBundle: ModelBundle = {
  schemaVersion: 3,
  headsTrained: { logits: true, nutritionQuantiles: true },
  targetTransform: transform,
  // A releasable artifact carries these. Without them the raw quantiles cover about 82% of
  // the truth while the type calls them the 5th and 95th percentile, so the library refuses
  // to hand them out and a fixture without them is not a model anyone could ship.
  conformal: { keys: [...targetKeys], offsets: targetKeys.map(() => 5), alpha: 0.1 },
};

/**
 * A stand-in for onnxruntime. Injecting the runtime rather than importing it is what makes
 * this testable at all: neither ORT package needs to be installed to exercise the logic.
 */
const fakeOrt = (overrides: { outputs?: Record<string, { data: ArrayLike<number> }> } = {}) => {
  const captured: { dims?: number[]; feeds?: Record<string, unknown> } = {};

  // Distinct values per quantile. All zeros would collapse every interval to a point and
  // make an ordering assertion pass or fail for the wrong reason.
  const nutrition = new Float32Array(targetKeys.length * quantiles.length);
  targetKeys.forEach((_, target) => {
    [-1, 0, 1].forEach((value, q) => {
      nutrition[target * quantiles.length + q] = value;
    });
  });

  const outputs = overrides.outputs ?? {
    logits: { data: new Float32Array(classCount).fill(0).map((_, i) => (i === 7 ? 10 : 0)) },
    nutrition_quantiles: { data: nutrition },
  };

  const ort = {
    Tensor: class {
      constructor(
        public type: string,
        public data: Uint8Array,
        public dims: number[],
      ) {
        captured.dims = dims;
      }
    },
    InferenceSession: {
      create: vi.fn(async () => ({
        run: vi.fn(async (feeds: Record<string, unknown>) => {
          captured.feeds = feeds;
          return outputs;
        }),
        release: vi.fn(async () => undefined),
      })),
    },
  };

  return { ort: ort as never, captured };
};

const image = (width = 4, height = 3): RgbImage => ({
  data: new Uint8Array(width * height * 3).fill(128),
  width,
  height,
});

describe("PlateVision", () => {
  it("returns dish candidates from the logits", async () => {
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "model.onnx", bundle: trainedBundle });

    const result = await pv.analyse(image());

    expect(result.dishes).toHaveLength(3);
    expect(result.dishes[0].confidence).toBeGreaterThan(result.dishes[1].confidence);
  });

  it("sends a uint8 NHWC tensor at the image's own resolution", async () => {
    // The graph resizes internally. Resizing here as well would apply it twice and put
    // the client's interpolation in a path the model was never evaluated against.
    const { ort, captured } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });

    await pv.analyse(image(640, 480));

    expect(captured.dims).toEqual([1, 480, 640, 3]);
  });

  it("produces nutrition when the head is trained", async () => {
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });

    const result = await pv.analyse(image());

    expect(result.nutrition).not.toBeNull();
    expect(result.nutrition?.energy.high).toBeGreaterThan(result.nutrition!.energy.low);
    expect(result.nutritionUnavailableReason).toBeNull();
  });

  it("withholds nutrition when the head is untrained", async () => {
    // The property this library exists to guarantee. A model can carry a trained
    // classifier and an untrained nutrition head, and returning plausible numbers from
    // random weights is the easiest way for an integrator to publish something false.
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, {
      model: "m",
      bundle: { ...trainedBundle, headsTrained: { logits: true, nutritionQuantiles: false } },
    });

    const result = await pv.analyse(image());

    expect(result.nutrition).toBeNull();
    expect(result.nutritionUnavailableReason).toMatch(/untrained/i);
    expect(result.dishes.length).toBeGreaterThan(0);
  });

  it("withholds nutrition when a trained head shipped without conformal offsets", async () => {
    // The case that made schema 2 unsafe to accept blindly. Raw quantiles from this model
    // cover about 82% while being labelled 90%, and there is no way to present that
    // honestly, so it is withheld exactly as an untrained head is.
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, {
      model: "m",
      bundle: { ...trainedBundle, conformal: null },
    });

    const result = await pv.analyse(image());

    expect(result.nutrition).toBeNull();
    expect(result.nutritionUnavailableReason).toMatch(/conformal/i);
    expect(result.dishes.length).toBeGreaterThan(0);
  });

  it("loads a schema 2 bundle, which published artifacts still carry", async () => {
    // Raising the floor to 3 alone broke the demo and every installed build against
    // model-v0.1.0. A new client meeting an old artifact is the safe direction: it knows
    // version 2 carries no offsets rather than having to guess.
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, {
      model: "m",
      bundle: {
        ...trainedBundle,
        schemaVersion: 2,
        headsTrained: { logits: true, nutritionQuantiles: false },
        conformal: null,
      },
    });

    const result = await pv.analyse(image());
    expect(result.dishes.length).toBeGreaterThan(0);
  });

  it("withholds nutrition when the bundle carries no target transform", async () => {
    // Without it the outputs are unitless numbers, not kilocalories.
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, {
      model: "m",
      bundle: { ...trainedBundle, targetTransform: null },
    });

    const result = await pv.analyse(image());

    expect(result.nutrition).toBeNull();
    expect(result.nutritionUnavailableReason).toMatch(/target transform/i);
  });

  it("reports whether nutrition is available before being asked to analyse", async () => {
    const { ort } = fakeOrt();
    const usable = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });
    const unusable = await PlateVision.load(ort, {
      model: "m",
      bundle: { ...trainedBundle, targetTransform: null },
    });

    expect(usable.nutritionAvailable).toBe(true);
    expect(unusable.nutritionAvailable).toBe(false);
  });

  it("honours topK", async () => {
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle, topK: 5 });
    expect((await pv.analyse(image())).dishes).toHaveLength(5);
  });

  it("rejects an image whose byte length disagrees with its dimensions", async () => {
    // Almost always alpha that was never stripped, which would silently shift every pixel.
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });

    await expect(
      pv.analyse({ data: new Uint8Array(10), width: 4, height: 3 }),
    ).rejects.toThrow(/Alpha must be stripped/);
  });

  it("reports a missing output rather than returning undefined", async () => {
    const { ort } = fakeOrt({ outputs: { logits: { data: new Float32Array(classCount) } } });
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });

    await expect(pv.analyse(image())).rejects.toThrow(/nutrition_quantiles/);
  });

  it("measures inference time", async () => {
    const { ort } = fakeOrt();
    const pv = await PlateVision.load(ort, { model: "m", bundle: trainedBundle });
    expect((await pv.analyse(image())).inferenceMs).toBeGreaterThanOrEqual(0);
  });
});

describe("stripAlpha", () => {
  it("drops every fourth byte", () => {
    const rgba = new Uint8Array([1, 2, 3, 255, 4, 5, 6, 128]);
    expect([...stripAlpha(rgba)]).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("produces three quarters of the input length", () => {
    expect(stripAlpha(new Uint8Array(400)).length).toBe(300);
  });
});
