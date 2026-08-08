/**
 * Inference boundary.
 *
 * Acquires the model, decodes the captured photo, and runs it through
 * `@plate-vision/client`. Everything the result means, including whether a nutrition
 * estimate is legitimate at all, is the library's decision; this file only carries the
 * answer to the screen along with the photo it came from.
 */

import { load, type PlateVision } from "@plate-vision/client";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";

import type { MealAnalysis } from "../types";
import { DECODE_MAX_EDGE, decodeBase64Jpeg } from "./decode";
import { ModelUnavailableError, resolveModel } from "./model";

export { ModelUnavailableError } from "./model";

/**
 * One session, kept alive across photos.
 *
 * Creating an ONNX session parses and prepares the whole graph, which is the expensive
 * part; doing it per photo would put a second of avoidable work in front of every result
 * and make the on-device latency claim meaningless.
 */
let session: Promise<PlateVision> | null = null;

const sessionFor = async (): Promise<{ pv: PlateVision }> => {
  if (!session) {
    session = (async () => {
      const { path, bundle } = await resolveModel();
      return load({ model: path, bundle });
    })();
  }

  try {
    return { pv: await session };
  } catch (error) {
    // A failed load must not be cached as a permanent failure. The usual cause is a
    // network problem fetching the manifest, and the next attempt should be allowed to
    // succeed rather than returning the same stale rejection forever.
    session = null;
    throw error;
  }
};

/** Reduce the photo and hand back its JPEG bytes, base64 as expo-image-manipulator emits. */
const photoBytes = async (photoUri: string): Promise<string> => {
  const context = ImageManipulator.manipulate(photoUri);
  // The model resizes internally, so this is about decode cost rather than correctness:
  // a full-resolution phone photo decoded in JavaScript is seconds of work for pixels the
  // graph immediately discards.
  context.resize({ width: DECODE_MAX_EDGE });

  const image = await context.renderAsync();
  const saved = await image.saveAsync({ format: SaveFormat.JPEG, base64: true });

  if (!saved.base64) throw new Error("the image manipulator returned no image data");
  return saved.base64;
};

export const analyse = async (photoUri: string): Promise<MealAnalysis> => {
  const { pv } = await sessionFor();
  const image = decodeBase64Jpeg(await photoBytes(photoUri));
  const result = await pv.analyse(image);

  return { ...result, photoUri, placeholder: false };
};

/** Whether a thrown error is the app's own "no usable model" case rather than a bug. */
export const isModelUnavailable = (error: unknown): error is ModelUnavailableError =>
  error instanceof ModelUnavailableError;
