/**
 * Browser entry point. Backed by `onnxruntime-web`.
 *
 * React Native resolves `index.native.ts` instead, through the `react-native` condition in
 * package.json exports. Both entry points expose the same API; only the runtime differs.
 */

import { PlateVision } from "./plate-vision";
import { stripAlpha } from "./session";
import type { LoadOptions, RgbImage } from "./types";

export { PlateVision } from "./plate-vision";
export { classCount, contract, labels, labelForIndex, keyForIndex } from "./contract";
export { enforceMonotonic, softmax, topK } from "./postprocess";
export { stripAlpha } from "./session";
export type {
  Analysis,
  DishPrediction,
  Interval,
  LoadOptions,
  ModelBundle,
  NutritionEstimate,
  RgbImage,
  TargetTransform,
} from "./types";

/** Load a model using `onnxruntime-web`. */
export const load = async (options: LoadOptions): Promise<PlateVision> => {
  const ort = await import("onnxruntime-web");
  return PlateVision.load(ort as never, options);
};

/**
 * Decode an image element, blob, or URL into raw RGB via canvas.
 *
 * Convenience only. The model resizes internally, so no resizing happens here beyond an
 * optional cap, which exists to keep decode cost down on very large photographs rather
 * than for correctness.
 */
export const decodeImage = async (
  source: CanvasImageSource | Blob | string,
  maxEdge = 640,
): Promise<RgbImage> => {
  const bitmap =
    typeof source === "string"
      ? await createImageBitmap(await (await fetch(source)).blob())
      : source instanceof Blob
        ? await createImageBitmap(source)
        : source;

  const width = "width" in bitmap ? Number(bitmap.width) : 0;
  const height = "height" in bitmap ? Number(bitmap.height) : 0;
  const scale = Math.min(1, maxEdge / Math.max(width, height));
  const targetWidth = Math.max(1, Math.round(width * scale));
  const targetHeight = Math.max(1, Math.round(height * scale));

  const canvas = new OffscreenCanvas(targetWidth, targetHeight);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("could not obtain a 2d canvas context");

  context.drawImage(bitmap as CanvasImageSource, 0, 0, targetWidth, targetHeight);
  const { data } = context.getImageData(0, 0, targetWidth, targetHeight);

  return { data: stripAlpha(data), width: targetWidth, height: targetHeight };
};
