/**
 * Decoding a native depth payload, kept away from the native module so it can be tested.
 *
 * `modules/depth-capture/index.ts` cannot be imported outside a device build: it resolves a
 * native module at import time. This is the part of that path with actual logic in it, and
 * the app's existing boundary applies, that anything testable lives where it can run.
 */

import { toByteArray } from "base64-js";

import type { DepthMap } from "@plate-vision/client/depth";

/** The shape the native side returns. Declared here because this file defines what it means. */
export type NativeCapture = {
  width: number;
  height: number;
  /** Little-endian unsigned 16-bit millimetres, one per pixel. */
  base64: string;
  filtered: boolean;
  accuracy: "absolute" | "relative";
  sensor: string;
};

/**
 * Base64 unsigned 16-bit millimetres into the map shape the rest of the project uses.
 *
 * Little-endian is read explicitly rather than by casting the byte buffer to a
 * `Uint16Array`, which would inherit the platform's endianness. Every device this can run on
 * is little-endian today, so the wrong version would pass every test and every review, and
 * would produce plausible, wrong distances the first time that stopped being true.
 */
export const decodeDepth = (payload: NativeCapture): DepthMap => {
  const bytes = toByteArray(payload.base64);
  const expected = payload.width * payload.height * 2;
  if (bytes.length !== expected) {
    throw new Error(
      `depth payload is ${bytes.length} bytes, expected ${expected} for ${payload.width}x${payload.height}`,
    );
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const data = new Uint16Array(payload.width * payload.height);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = view.getUint16(i * 2, true);
  }

  return { data, width: payload.width, height: payload.height };
};
