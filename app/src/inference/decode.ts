/**
 * Turning a captured photo into the raw pixels the model takes.
 *
 * React Native has no canvas, so the browser's `decodeImage` has no counterpart here and
 * decoding is the app's job. `jpeg-js` is pure JavaScript, which is slower than a native
 * decoder but has the property that matters most for this project: it runs in Node, so the
 * conversion below is covered by tests rather than only ever exercised on a phone.
 *
 * The model resizes internally, so nothing here resizes for correctness. Downscaling before
 * decode is purely about cost: a 12-megapixel photo decoded in JavaScript is seconds of
 * work for pixels the graph immediately throws away.
 */

import { toByteArray } from "base64-js";
import { decode as decodeJpeg } from "jpeg-js";
import type { RgbImage } from "@plate-vision/client";

/**
 * Longest edge the photo is reduced to before decode.
 *
 * Comfortably above the model's 224px input so the in-graph resize still has detail to
 * work from, and far below a phone camera's native resolution.
 */
export const DECODE_MAX_EDGE = 640;

/** Decode JPEG bytes into row-major RGB, dropping the alpha channel jpeg-js emits. */
export const decodeJpegToRgb = (bytes: Uint8Array): RgbImage => {
  // useTArray keeps the output a Uint8Array rather than a Node Buffer, which does not
  // exist in React Native.
  const raw = decodeJpeg(bytes, { useTArray: true });

  const { width, height, data } = raw;
  const expected = width * height * 4;
  if (data.length !== expected) {
    throw new Error(
      `decoded ${data.length} bytes for a ${width}x${height} image, expected ${expected}`,
    );
  }

  // jpeg-js always emits RGBA. Passing that straight to the model would shift every pixel
  // by one channel and produce predictions from what is effectively a different image.
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0, source = 0, target = 0; pixel < width * height; pixel += 1) {
    rgb[target] = data[source];
    rgb[target + 1] = data[source + 1];
    rgb[target + 2] = data[source + 2];
    source += 4;
    target += 3;
  }

  return { data: rgb, width, height };
};

/** Decode a base64 JPEG, which is how expo-file-system hands back binary content. */
export const decodeBase64Jpeg = (base64: string): RgbImage =>
  decodeJpegToRgb(toByteArray(base64));

/**
 * Scale factor to apply before decode, or 1 when the photo is already small enough.
 *
 * Separate from the manipulator call so the arithmetic is testable. Getting this wrong in
 * the direction of "no resize" costs seconds per photo; getting it wrong in the direction
 * of over-shrinking silently degrades every prediction.
 */
export const decodeScale = (width: number, height: number, maxEdge = DECODE_MAX_EDGE): number => {
  const longest = Math.max(width, height);
  if (longest <= 0) throw new Error("photo dimensions must be positive");
  return longest <= maxEdge ? 1 : maxEdge / longest;
};
