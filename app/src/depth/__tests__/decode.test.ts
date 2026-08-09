import { describe, expect, it } from "vitest";

import { decodeDepth, type NativeCapture } from "../decode";

/** Little-endian unsigned 16-bit, the way the native side writes it. */
const encode = (values: number[]): string => {
  const bytes = new Uint8Array(values.length * 2);
  const view = new DataView(bytes.buffer);
  values.forEach((v, i) => view.setUint16(i * 2, v, true));
  return Buffer.from(bytes).toString("base64");
};

const payload = (values: number[], width: number, height: number): NativeCapture => ({
  width,
  height,
  base64: encode(values),
  filtered: false,
  accuracy: "absolute",
  sensor: "Back LiDAR Camera",
});

describe("decodeDepth", () => {
  it("reads millimetres back in the order they were written", () => {
    const map = decodeDepth(payload([300, 1000, 3562, 65535], 4, 1));
    expect(Array.from(map.data)).toEqual([300, 1000, 3562, 65535]);
    expect(map.width).toBe(4);
    expect(map.height).toBe(1);
  });

  it("reads little-endian regardless of what the host would do", () => {
    // 0x012C is 300. Written low byte first, it must not come back as 0x2C01 = 11265.
    const bytes = Uint8Array.from([0x2c, 0x01]);
    const map = decodeDepth({
      width: 1,
      height: 1,
      base64: Buffer.from(bytes).toString("base64"),
      filtered: false,
      accuracy: "absolute",
      sensor: "",
    });
    expect(map.data[0]).toBe(300);
  });

  it("keeps dropouts as zero rather than turning them into a distance", () => {
    const map = decodeDepth(payload([0, 300, 0], 3, 1));
    expect(Array.from(map.data)).toEqual([0, 300, 0]);
  });

  it("refuses a payload whose length disagrees with its stated size", () => {
    // A truncated transfer would otherwise decode into a valid, smaller map and be measured
    // as though the sensor had simply seen less.
    expect(() => decodeDepth(payload([1, 2, 3], 2, 2))).toThrow(/6 bytes, expected 8 for 2x2/);
  });
});
