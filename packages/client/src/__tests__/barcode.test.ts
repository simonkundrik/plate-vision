import { describe, expect, it, vi } from "vitest";

import {
  isValidBarcode,
  lookupBarcode,
  nutritionFromProduct,
  parseProduct,
  type ProductNutrition,
} from "../barcode";

/** A real Open Food Facts response, trimmed to the fields this module reads. */
const nutella = {
  status: 1,
  product: {
    product_name: "Nutella",
    brands: "Nutella, Ferrero, Yum yum",
    nutriments: {
      "energy-kcal_100g": 539,
      proteins_100g: 6.3,
      fat_100g: 30.9,
      carbohydrates_100g: 57.5,
    },
  },
};

const product = (over: Partial<ProductNutrition> = {}): ProductNutrition => ({
  barcode: "3017620422003",
  name: "Nutella",
  brand: "Nutella",
  energyPer100g: 539,
  proteinPer100g: 6.3,
  fatPer100g: 30.9,
  carbohydratePer100g: 57.5,
  ...over,
});

describe("isValidBarcode", () => {
  it("accepts the usual lengths", () => {
    expect(isValidBarcode("3017620422003")).toBe(true); // EAN-13
    expect(isValidBarcode("12345678")).toBe(true); // EAN-8
  });

  it("rejects anything that is not digits", () => {
    // Validating before the request turns a scanner misread into an immediate answer
    // rather than a round trip and a confusing "not found".
    expect(isValidBarcode("abc123")).toBe(false);
    expect(isValidBarcode("301-762-042")).toBe(false);
  });

  it("rejects lengths no barcode uses", () => {
    expect(isValidBarcode("1234")).toBe(false);
    expect(isValidBarcode("123456789012345")).toBe(false);
  });
});

describe("parseProduct", () => {
  it("reads composition from a real response", () => {
    const result = parseProduct("3017620422003", nutella);
    expect(result.found).toBe(true);
    if (!result.found) return;

    expect(result.product.name).toBe("Nutella");
    expect(result.product.energyPer100g).toBe(539);
    expect(result.product.fatPer100g).toBe(30.9);
  });

  it("takes the most specific brand from the list", () => {
    const result = parseProduct("3017620422003", nutella);
    expect(result.found && result.product.brand).toBe("Nutella");
  });

  it("reports an unknown barcode as not found rather than throwing", () => {
    // A miss is an ordinary outcome for a community database, not an error. Roughly a
    // third of scans miss.
    const result = parseProduct("0000000000000", { status: 0 });
    expect(result.found).toBe(false);
    expect(result).toMatchObject({ reason: expect.stringContaining("not in Open Food Facts") });
  });

  it("refuses a product with no energy value", () => {
    // The common failure for a community database: someone photographed the front of the
    // packet and nobody typed in the table on the back. Without this the product becomes a
    // confident zero calories.
    const result = parseProduct("123456789", {
      status: 1,
      product: { product_name: "Mystery", nutriments: {} },
    });
    expect(result.found).toBe(false);
    expect(result).toMatchObject({ reason: expect.stringContaining("no energy value") });
  });

  it("defaults missing macros to zero but never a missing energy", () => {
    const result = parseProduct("123456789", {
      status: 1,
      product: { product_name: "Sparse", nutriments: { "energy-kcal_100g": 100 } },
    });
    expect(result.found).toBe(true);
    expect(result.found && result.product.proteinPer100g).toBe(0);
  });

  it("rejects a negative energy value", () => {
    const result = parseProduct("123456789", {
      status: 1,
      product: { nutriments: { "energy-kcal_100g": -5 } },
    });
    expect(result.found).toBe(false);
  });

  it("survives a product with no name", () => {
    const result = parseProduct("123456789", {
      status: 1,
      product: { nutriments: { "energy-kcal_100g": 100 } },
    });
    expect(result.found && result.product.name).toBe("Unnamed product");
  });
});

describe("lookupBarcode", () => {
  // Parameters are declared even though the body ignores them: without them vi.fn types
  // the call record as an empty tuple and asserting on the request becomes impossible.
  const respond = (payload: unknown, ok = true, status = 200) =>
    vi.fn(async (_url: string, _init?: { headers?: Record<string, string> }) => ({
      ok,
      status,
      json: async () => payload,
    }));

  it("fetches and parses", async () => {
    const fetch = respond(nutella);
    const result = await lookupBarcode("3017620422003", { fetch });

    expect(result.found).toBe(true);
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("identifies itself to the API", async () => {
    // Open Food Facts asks callers to. An anonymous flood is how a free community API ends
    // up rate-limiting everyone.
    const fetch = respond(nutella);
    await lookupBarcode("3017620422003", { fetch });

    const [, init] = fetch.mock.calls[0];
    expect(init?.headers?.["User-Agent"]).toContain("plate-vision");
  });

  it("does not call the network for a malformed barcode", async () => {
    const fetch = respond(nutella);
    const result = await lookupBarcode("not-a-barcode", { fetch });

    expect(result.found).toBe(false);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("reports an HTTP failure without throwing", async () => {
    const result = await lookupBarcode("3017620422003", { fetch: respond({}, false, 503) });
    expect(result).toMatchObject({ found: false, reason: expect.stringContaining("503") });
  });
});

describe("nutritionFromProduct", () => {
  it("scales composition by mass", () => {
    const nutrition = nutritionFromProduct(product(), { low: 20, median: 20, high: 20 });
    expect(nutrition.energy.median).toBeCloseTo(107.8);
  });

  it("puts all the uncertainty in the mass, where it belongs", () => {
    // The packet is not guessing about composition. Nobody knows how much is on the plate.
    const nutrition = nutritionFromProduct(product(), { low: 10, median: 20, high: 40 });

    expect(nutrition.energy.low / nutrition.energy.median).toBeCloseTo(0.5);
    expect(nutrition.energy.high / nutrition.energy.median).toBeCloseTo(2.0);
  });

  it("a known mass gives a known answer", () => {
    // Weighed on a scale, or a stated serving the user confirmed: a zero-width mass
    // interval must produce a zero-width calorie interval rather than manufactured doubt.
    const nutrition = nutritionFromProduct(product(), { low: 15, median: 15, high: 15 });
    expect(nutrition.energy.low).toBeCloseTo(nutrition.energy.high);
  });

  it("labels itself as the barcode route", async () => {
    // Not the vision model's route, and not measured by the vision model's numbers.
    const nutrition = nutritionFromProduct(product(), { low: 10, median: 20, high: 40 });
    expect(nutrition.route).toBe("barcode");
  });

  it("refuses the vision model's conformal widening", async () => {
    // The offsets were fitted on held-out Nutrition5k photographs, so they describe the
    // vision model's miss. Widening a stated composition by them manufactures doubt about a
    // figure printed on the packet.
    const { applyConformal } = await import("../postprocess");
    const nutrition = nutritionFromProduct(product(), { low: 10, median: 20, high: 40 });

    expect(() => applyConformal(nutrition, { keys: ["energy"], offsets: [50] })).toThrow(
      /calibrated on the absolute route/,
    );
  });

  it("passes the mass through unchanged", () => {
    const mass = { low: 10, median: 20, high: 40 };
    expect(nutritionFromProduct(product(), mass).mass).toEqual(mass);
  });

  it("scales every macro, not only energy", () => {
    const nutrition = nutritionFromProduct(product(), { low: 100, median: 100, high: 100 });
    expect(nutrition.fat.median).toBeCloseTo(30.9);
    expect(nutrition.carbohydrate.median).toBeCloseTo(57.5);
  });
});
