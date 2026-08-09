/** Public types. Everything an integrator touches is declared here. */

export type DishPrediction = {
  /** Food-101 class key, e.g. `spaghetti_carbonara`. Stable across model versions. */
  key: string;
  /** Human-readable name suitable for display. */
  label: string;
  /** Softmax probability in [0, 1]. */
  confidence: number;
};

/**
 * A predicted quantity as an interval rather than a number.
 *
 * `low` and `high` are the model's 5th and 95th percentile predictions, `median` the 50th.
 * They are separate fields rather than a single value because portion size cannot be
 * recovered from a photograph: a point estimate would be precision the model does not have.
 */
export type Interval = {
  low: number;
  median: number;
  high: number;
};

/**
 * Which evidence produced an estimate.
 *
 * Ordered best evidence first, mirroring `ROUTE_PRIORITY` in `platevision/routing.py`, which
 * is where per-route accuracy is measured. Their error distributions are nothing alike: a
 * barcode states the composition where the vision model infers it, so an integrator that
 * cannot tell them apart cannot report either honestly.
 */
export type EvidenceRoute = "barcode" | "chain_menu" | "scale_reference" | "absolute";

export type NutritionEstimate = {
  /** Kilocalories. */
  energy: Interval;
  /** Grams. */
  protein: Interval;
  fat: Interval;
  carbohydrate: Interval;
  /** Total plate mass in grams. */
  mass: Interval;
  /**
   * Where these numbers came from.
   *
   * Required rather than optional, so an estimate cannot reach an integrator without stating
   * its provenance. `absolute` means the photograph alone, which is the least accurate route
   * and the one every published figure in this project was measured on.
   */
  route: EvidenceRoute;
};

export type Analysis = {
  /** Dish candidates, most confident first. */
  dishes: DishPrediction[];
  /**
   * Nutrition estimate, or `null` when the loaded model's nutrition head is not trained.
   *
   * Deliberately nullable rather than optional-and-usually-there. A model can ship with a
   * trained classifier and an untrained nutrition head, and returning plausible numbers
   * from random weights is the single easiest way for an integrator to publish something
   * false without noticing.
   */
  nutrition: NutritionEstimate | null;
  /** Why nutrition is null, when it is. */
  nutritionUnavailableReason: string | null;
  /** Milliseconds spent inside the model, excluding decode. */
  inferenceMs: number;
};

/** Raw pixels, already decoded. Height and width are free; the model resizes internally. */
export type RgbImage = {
  /** Row-major RGB bytes, length `width * height * 3`. */
  data: Uint8Array;
  width: number;
  height: number;
};

/**
 * Per-target statistics for undoing the training-time transform.
 *
 * The model predicts in standardised log space, so without these its nutrition outputs are
 * unitless numbers rather than kilocalories. They travel in the model bundle for that
 * reason.
 */
export type TargetTransform = {
  mean: number[];
  std: number[];
  keys: string[];
};

/**
 * Per-target interval widening, fitted on held-out data by the exporter.
 *
 * Applied to the outer quantiles and not the median: conformal prediction says nothing
 * about the point estimate, and the median was already the well-calibrated output.
 */
export type ConformalOffsets = {
  keys: string[];
  offsets: number[];
  alpha: number;
};

/** Metadata published alongside a model artifact. */
export type ModelBundle = {
  schemaVersion: number;
  /** Which output heads carry trained weights. */
  headsTrained: {
    logits: boolean;
    nutritionQuantiles: boolean;
  };
  targetTransform: TargetTransform | null;
  conformal: ConformalOffsets | null;
};

export type LoadOptions = {
  /**
   * The ONNX model. A URL or path in React Native, a URL or ArrayBuffer on the web.
   */
  model: string | ArrayBuffer | Uint8Array;
  /** Bundle metadata describing the artifact. */
  bundle: ModelBundle;
  /** How many dish candidates to return. */
  topK?: number;
};

export type AnalyseOptions = {
  /**
   * How many augmented views to average, 1 to 4. Costs one forward pass each.
   *
   * Measured on the Nutrition5k test split, three views took calorie MAE from 56.7 to 54.0
   * and interval coverage from 82.2% to 84.4%. It is the only change in this project that
   * improved accuracy and calibration together rather than trading one against the other.
   */
  views?: number;
};
