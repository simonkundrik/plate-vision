/**
 * Nutrition from a barcode, via Open Food Facts.
 *
 * The vision model is worst at exactly the case a barcode is best at: a wrapped product,
 * where the composition is printed on the packet and the pixels show a wrapper. Reading the
 * wrapper beats estimating from the photo.
 *
 * **This module makes a network request.** The rest of `@plate-vision/client` runs entirely
 * on-device and uploads nothing, which is the reason it exists, so this lives behind its own
 * entry point rather than being importable by accident:
 *
 * ```ts
 * import { lookupBarcode } from "@plate-vision/client/barcode";
 * ```
 *
 * What it does and does not buy, measured against this project's own error decomposition:
 * calorie error splits about evenly between mass and density, roughly independently. A
 * barcode collapses the density half to nothing, because the composition is stated rather
 * than inferred. It does nothing for the mass half: knowing a jar is 539 kcal per 100g does
 * not say how much of it is on the plate. Expect a meaningful improvement, not an exact
 * answer, unless the amount is also known.
 */

import type { Interval, NutritionEstimate } from "./types";

const API = "https://world.openfoodfacts.org/api/v2/product";

// Open Food Facts asks callers to identify themselves. An anonymous flood is how a free
// community API ends up rate-limiting everyone.
const USER_AGENT = "plate-vision/0.1 (https://github.com/simonkundrik/plate-vision)";

/** Composition per 100 grams, exactly as the packet states it. */
export type ProductNutrition = {
  barcode: string;
  name: string;
  brand: string | null;
  /** Kilocalories per 100g. */
  energyPer100g: number;
  proteinPer100g: number;
  fatPer100g: number;
  carbohydratePer100g: number;
};

export type BarcodeLookup =
  | { found: true; product: ProductNutrition }
  | { found: false; reason: string };

type FetchLike = (url: string, init?: { headers?: Record<string, string> }) => Promise<{
  ok: boolean;
  status: number;
  json: () => Promise<unknown>;
}>;

/**
 * A barcode is 8 to 14 digits. Validating before the request turns a scanner misread into
 * an immediate answer rather than a round trip and a confusing "not found".
 */
export const isValidBarcode = (barcode: string): boolean => /^\d{8,14}$/.test(barcode.trim());

const numberOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;

/**
 * Look up a product's composition.
 *
 * Returns `{ found: false }` with a reason rather than throwing, because "this barcode is
 * not in the database" is an ordinary outcome for a community-maintained dataset, not an
 * error. Roughly a third of scans miss.
 */
export const lookupBarcode = async (
  barcode: string,
  options: { fetch?: FetchLike; signal?: AbortSignal } = {},
): Promise<BarcodeLookup> => {
  const trimmed = barcode.trim();
  if (!isValidBarcode(trimmed)) {
    return { found: false, reason: `${barcode} is not a barcode; expected 8 to 14 digits` };
  }

  const doFetch = options.fetch ?? (globalThis.fetch as unknown as FetchLike);
  if (!doFetch) {
    return { found: false, reason: "no fetch implementation available in this environment" };
  }

  const fields = "code,product_name,brands,nutriments";
  const response = await doFetch(`${API}/${trimmed}.json?fields=${fields}`, {
    headers: { "User-Agent": USER_AGENT },
  });

  if (!response.ok) {
    return { found: false, reason: `Open Food Facts returned ${response.status}` };
  }

  return parseProduct(trimmed, await response.json());
};

/**
 * Turn an Open Food Facts response into a product, or say why it cannot.
 *
 * Separated from the request so it is testable without a network, and because the failure
 * that matters is not the network one: entries in a community database are frequently
 * missing the nutriments, and a product with a name but no energy value would otherwise
 * become a confident zero.
 */
export const parseProduct = (barcode: string, payload: unknown): BarcodeLookup => {
  const body = payload as {
    status?: number;
    product?: {
      product_name?: unknown;
      brands?: unknown;
      nutriments?: Record<string, unknown>;
    };
  };

  if (body?.status !== 1 || !body.product) {
    return { found: false, reason: `${barcode} is not in Open Food Facts` };
  }

  const nutriments = body.product.nutriments ?? {};
  const energy = numberOrNull(nutriments["energy-kcal_100g"]);

  if (energy === null) {
    // The common case for a community database: the product exists, someone photographed
    // the front of the packet, and nobody typed in the table on the back.
    return {
      found: false,
      reason: `${barcode} is in Open Food Facts but has no energy value recorded`,
    };
  }

  const name = typeof body.product.product_name === "string" ? body.product.product_name : "";
  const brands = typeof body.product.brands === "string" ? body.product.brands : "";

  return {
    found: true,
    product: {
      barcode,
      name: name.trim() || "Unnamed product",
      // Open Food Facts stores brands as a comma-separated list, most specific first.
      brand: brands.split(",")[0]?.trim() || null,
      energyPer100g: energy,
      proteinPer100g: numberOrNull(nutriments["proteins_100g"]) ?? 0,
      fatPer100g: numberOrNull(nutriments["fat_100g"]) ?? 0,
      carbohydratePer100g: numberOrNull(nutriments["carbohydrates_100g"]) ?? 0,
    },
  };
};

/**
 * Combine a stated composition with an estimated mass.
 *
 * All of the remaining uncertainty comes from the mass, which is the honest attribution: the
 * packet says how many calories are in 100g and is not guessing, while nobody knows how much
 * of it is on the plate. Passing a zero-width mass interval, because the user weighed it or
 * ate a stated serving, correctly produces a zero-width calorie interval.
 */
export const nutritionFromProduct = (
  product: ProductNutrition,
  massGrams: Interval,
): NutritionEstimate => {
  const per = (gramsPer100: number): Interval => ({
    low: (massGrams.low * gramsPer100) / 100,
    median: (massGrams.median * gramsPer100) / 100,
    high: (massGrams.high * gramsPer100) / 100,
  });

  return {
    energy: per(product.energyPer100g),
    protein: per(product.proteinPer100g),
    fat: per(product.fatPer100g),
    carbohydrate: per(product.carbohydratePer100g),
    mass: massGrams,
    // Labelled, not assumed. This estimate and a vision estimate have error distributions
    // nothing alike, and quoting one accuracy for both is how a near-exact figure measured
    // on wrapped products becomes a claim about dinner.
    route: "barcode",
  };
};
