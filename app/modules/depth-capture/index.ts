/**
 * JavaScript face of the experimental depth capture module.
 *
 * The native side exists for Apple hardware only. On Android, on the web, and in Expo Go,
 * `requireOptionalNativeModule` returns null and everything here reports "not supported"
 * with a reason, rather than throwing from an import and taking the whole app down on
 * devices that were never the target.
 *
 * Nothing in this file has run on a device. It is typechecked, and the pure decoding below
 * is unit tested against a hand-built buffer, but the native module it talks to has never
 * been compiled. See `README.md` in this directory.
 */

import { requireOptionalNativeModule } from "expo-modules-core";

import type { DepthMap } from "@plate-vision/client/depth";

import { decodeDepth, type NativeCapture } from "../../src/depth/decode";

/** What the native module returns from `support()`. */
type NativeSupport = {
  supported: boolean;
  reason: string;
  sensor: string;
};

type NativeModule = {
  support: () => NativeSupport;
  capture: () => Promise<NativeCapture>;
};

const native = requireOptionalNativeModule<NativeModule>("DepthCapture");

export type DepthSupport = {
  supported: boolean;
  /** Shown to the user verbatim, so it has to read as an explanation rather than a code. */
  reason: string;
  sensor: string;
};

export type DepthCapture = {
  map: DepthMap;
  /** Name the platform gives the sensor, for the issue report. */
  sensor: string;
  /**
   * Whether the platform interpolated over the holes.
   *
   * Requested as false. If it ever arrives true, the dropout figure downstream is measuring
   * the smoother rather than the sensor, and the capture should not be compared against
   * Nutrition5k's dropout rate at all.
   */
  filtered: boolean;
  /**
   * `relative` means the values carry no metric unit.
   *
   * Every millimetre figure downstream assumes `absolute`. A relative capture is not a worse
   * measurement, it is a different quantity, and it must not be quietly reported as
   * millimetres.
   */
  accuracy: "absolute" | "relative";
};

export const NOT_BUILT =
  "This build has no depth capture module. It is compiled for Apple hardware only, and " +
  "Expo Go cannot load it at all: it needs a development build.";

export const depthSupport = (): DepthSupport => {
  if (!native) return { supported: false, reason: NOT_BUILT, sensor: "" };
  return native.support();
};

export const captureDepth = async (): Promise<DepthCapture> => {
  if (!native) throw new Error(NOT_BUILT);

  const payload = await native.capture();
  return {
    map: decodeDepth(payload),
    sensor: payload.sensor,
    filtered: payload.filtered,
    accuracy: payload.accuracy,
  };
};
