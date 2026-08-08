/**
 * Runtime-agnostic session handling.
 *
 * The ONNX Runtime module is injected rather than imported. The two runtimes have the same
 * surface for what this needs, so the logic lives here once, and injecting it means the
 * core is unit-testable without either native package installed. Tests supply a fake.
 */

import { inputName } from "./contract";
import type { RgbImage } from "./types";

/** The slice of the ONNX Runtime API this package uses. */
export type OrtLike = {
  Tensor: new (type: string, data: Uint8Array, dims: number[]) => unknown;
  InferenceSession: {
    create: (model: string | ArrayBuffer | Uint8Array, options?: unknown) => Promise<OrtSession>;
  };
};

export type OrtSession = {
  run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: ArrayLike<number> }>>;
  release?: () => Promise<void>;
};

export type RawOutput = {
  logits: ArrayLike<number>;
  nutrition: ArrayLike<number>;
  inferenceMs: number;
};

export const createSession = (
  ort: OrtLike,
  model: string | ArrayBuffer | Uint8Array,
): Promise<OrtSession> => ort.InferenceSession.create(model);

/**
 * Run one image through the graph.
 *
 * The tensor is uint8 NHWC at the image's own resolution. The model resizes and normalises
 * internally, so callers pass raw camera pixels and never reimplement preprocessing;
 * three separate implementations of the same ImageNet constants is how the numbers on one
 * platform quietly stop matching the others.
 */
export const runSession = async (
  ort: OrtLike,
  session: OrtSession,
  image: RgbImage,
): Promise<RawOutput> => {
  const expected = image.width * image.height * 3;
  if (image.data.length !== expected) {
    throw new Error(
      `image data is ${image.data.length} bytes but ${image.width}x${image.height} RGB ` +
        `needs ${expected}. Alpha must be stripped before calling.`,
    );
  }

  const tensor = new ort.Tensor("uint8", image.data, [1, image.height, image.width, 3]);

  const started = Date.now();
  const outputs = await session.run({ [inputName]: tensor });
  const inferenceMs = Date.now() - started;

  const logits = outputs.logits?.data;
  const nutrition = outputs.nutrition_quantiles?.data;
  if (!logits || !nutrition) {
    throw new Error(
      `model returned ${Object.keys(outputs).join(", ") || "nothing"}; ` +
        "expected logits and nutrition_quantiles",
    );
  }

  return { logits, nutrition, inferenceMs };
};

/** Drop the alpha channel from RGBA bytes, which is what canvas and most decoders emit. */
export const stripAlpha = (rgba: Uint8Array | Uint8ClampedArray): Uint8Array => {
  const pixels = rgba.length / 4;
  const rgb = new Uint8Array(pixels * 3);
  for (let i = 0; i < pixels; i += 1) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return rgb;
};
