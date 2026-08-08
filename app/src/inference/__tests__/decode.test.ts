import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { DECODE_MAX_EDGE, decodeBase64Jpeg, decodeJpegToRgb, decodeScale } from "../decode";

const here = dirname(fileURLToPath(import.meta.url));

/** 12x8: left half pure red, right half pure blue. Saved at quality 95. */
const fixture = () => new Uint8Array(readFileSync(join(here, "fixtures", "red-blue-12x8.jpg")));

const pixelAt = (rgb: Uint8Array, width: number, x: number, y: number) => {
  const offset = (y * width + x) * 3;
  return [rgb[offset], rgb[offset + 1], rgb[offset + 2]];
};

describe("decodeJpegToRgb", () => {
  it("recovers the image dimensions", () => {
    const image = decodeJpegToRgb(fixture());
    expect(image.width).toBe(12);
    expect(image.height).toBe(8);
  });

  it("produces exactly three bytes per pixel", () => {
    // The model rejects a buffer whose length disagrees with its dimensions, but by then
    // the cause is several layers away. jpeg-js emits RGBA and this is where it stops.
    const image = decodeJpegToRgb(fixture());
    expect(image.data.length).toBe(12 * 8 * 3);
  });

  it("keeps channels in RGB order rather than shifting them", () => {
    // The failure this guards against does not crash. Leaving alpha in place shifts every
    // pixel one channel along, and the model then predicts from a different image while
    // looking like it merely predicts badly.
    const image = decodeJpegToRgb(fixture());

    const [r, g, b] = pixelAt(image.data, image.width, 1, 4);
    expect(r).toBeGreaterThan(200);
    expect(g).toBeLessThan(60);
    expect(b).toBeLessThan(60);
  });

  it("distinguishes the two halves of the fixture", () => {
    const image = decodeJpegToRgb(fixture());

    const left = pixelAt(image.data, image.width, 1, 4);
    const right = pixelAt(image.data, image.width, 10, 4);

    expect(left[0]).toBeGreaterThan(left[2]);
    expect(right[2]).toBeGreaterThan(right[0]);
  });

  it("rejects bytes that are not a JPEG", () => {
    expect(() => decodeJpegToRgb(new Uint8Array([1, 2, 3, 4]))).toThrow();
  });
});

describe("decodeBase64Jpeg", () => {
  it("matches decoding the same bytes directly", () => {
    // expo-file-system and the image manipulator both hand back base64, so this is the
    // path the app actually takes.
    const bytes = fixture();
    const base64 = Buffer.from(bytes).toString("base64");

    expect(decodeBase64Jpeg(base64).data).toEqual(decodeJpegToRgb(bytes).data);
  });
});

describe("decodeScale", () => {
  it("leaves a small photo alone", () => {
    expect(decodeScale(320, 240)).toBe(1);
  });

  it("scales by the longest edge, not the width", () => {
    // A portrait photo is taller than it is wide, and scaling on width would leave the
    // long edge well above the cap while appearing to have worked.
    expect(decodeScale(1200, 1600)).toBeCloseTo(DECODE_MAX_EDGE / 1600);
  });

  it("is exactly one at the boundary", () => {
    expect(decodeScale(DECODE_MAX_EDGE, 100)).toBe(1);
  });

  it("rejects degenerate dimensions", () => {
    expect(() => decodeScale(0, 0)).toThrow(/positive/);
  });
});
