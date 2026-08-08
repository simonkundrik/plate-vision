import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { classCount, contract, inputName, labels, medianIndex, outputNames, quantiles, targetKeys } from "../contract";

const here = dirname(fileURLToPath(import.meta.url));
const sharedDir = join(here, "..", "..", "..", "..", "shared");
const readShared = (name: string) => JSON.parse(readFileSync(join(sharedDir, name), "utf8"));

describe("bundled contract", () => {
  it("matches the repository contract", () => {
    // The published package cannot reference files outside itself, so the contract is
    // copied in. This is what stops the copy from drifting: a moved label ordering does
    // not crash, it renames every prediction.
    expect(contract).toEqual(readShared("model_meta.json"));
  });

  it("matches the repository label list", () => {
    expect(labels).toEqual(readShared("food101_labels.json").labels);
  });

  it("carries all 101 classes", () => {
    expect(classCount).toBe(101);
  });

  it("preserves the canonical ordering, which is not alphabetical", () => {
    // cheesecake precedes cheese_plate because underscore sorts below letters in ASCII
    // but Food-101's own ordering does not. Re-sorting would relabel a chunk of the
    // class space.
    const keys = labels.map((entry) => entry.key);
    expect(keys.indexOf("cheesecake")).toBeLessThan(keys.indexOf("cheese_plate"));
    expect(keys[0]).toBe("apple_pie");
    expect(keys[keys.length - 1]).toBe("waffles");
  });

  it("indexes contiguously from zero", () => {
    expect(labels.map((entry) => entry.index)).toEqual(labels.map((_, i) => i));
  });
});

describe("derived accessors", () => {
  it("exposes the input tensor name the graph expects", () => {
    expect(inputName).toBe("image");
  });

  it("lists outputs in contract order", () => {
    expect(outputNames).toEqual(["logits", "nutrition_quantiles"]);
  });

  it("lists five nutrition targets", () => {
    expect(targetKeys).toEqual(["energy", "protein", "fat", "carbohydrate", "mass"]);
  });

  it("lists ascending quantile levels around a median", () => {
    expect(quantiles).toEqual([0.05, 0.5, 0.95]);
    expect(medianIndex).toBe(1);
  });

  it("declares preprocessing as in-graph, which is why clients do none", () => {
    expect(contract.preprocessing.location).toBe("in_graph");
    expect(contract.preprocessing.order).toEqual(["to_float_unit_range", "resize", "normalize"]);
  });

  it("declares a uint8 NHWC RGB input", () => {
    expect(contract.input.dtype).toBe("uint8");
    expect(contract.input.layout).toBe("NHWC");
    expect(contract.input.channel_order).toBe("RGB");
  });
});
