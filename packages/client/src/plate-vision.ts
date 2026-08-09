import { quantiles } from "./contract";
import {
  applyConformal,
  averageNutrition,
  averageProbabilities,
  toNutrition,
  topKFromProbabilities,
} from "./postprocess";
import { buildViews } from "./views";
import { createSession, runSession, type OrtLike, type OrtSession } from "./session";
import type { Analysis, AnalyseOptions, LoadOptions, ModelBundle, RgbImage } from "./types";

const DEFAULT_TOP_K = 3;

const NUTRITION_HEAD_UNTRAINED =
  "This model artifact ships an untrained nutrition head. Its outputs would be values " +
  "from random weights, so they are withheld rather than returned.";

const NO_TARGET_TRANSFORM =
  "The model bundle carries no target transform, so nutrition outputs cannot be converted " +
  "from the model's internal scale into kilocalories.";

/**
 * A loaded model.
 *
 * Construct with {@link PlateVision.load}. One instance holds one ONNX session; keep it
 * alive across analyses rather than reloading, since session creation is the expensive part.
 */
export class PlateVision {
  private constructor(
    private readonly ort: OrtLike,
    private readonly session: OrtSession,
    private readonly bundle: ModelBundle,
    private readonly topKCount: number,
  ) {}

  static async load(ort: OrtLike, options: LoadOptions): Promise<PlateVision> {
    const session = await createSession(ort, options.model);
    return new PlateVision(ort, session, options.bundle, options.topK ?? DEFAULT_TOP_K);
  }

  /** Whether this artifact can produce nutrition figures at all. */
  get nutritionAvailable(): boolean {
    return this.bundle.headsTrained.nutritionQuantiles && this.bundle.targetTransform !== null;
  }

  /**
   * Analyse one decoded image.
   *
   * Nutrition comes back as `null` whenever the loaded artifact cannot legitimately
   * produce it, with a reason. That is deliberate rather than a convenience: a model can
   * carry a trained classifier and an untrained nutrition head, and handing back plausible
   * numbers from random weights is the easiest way for an integrator to publish something
   * false without ever noticing.
   */
  async analyse(image: RgbImage, options: AnalyseOptions = {}): Promise<Analysis> {
    const views = buildViews(image, options.views ?? 1);

    // Sequential rather than concurrent. One ONNX session serialises its own calls, so
    // firing them together buys nothing and makes the reported time meaningless.
    const outputs = [];
    let inferenceMs = 0;
    for (const view of views) {
      const raw = await runSession(this.ort, this.session, view);
      outputs.push(raw);
      inferenceMs += raw.inferenceMs;
    }

    // Probabilities, not logits. Logits from different views are not on a common scale, so
    // averaging them weights whichever view happened to be most confident.
    const dishes = topKFromProbabilities(
      averageProbabilities(outputs.map((o) => o.logits)),
      this.topKCount,
    );

    if (!this.bundle.headsTrained.nutritionQuantiles) {
      return this.withoutNutrition(dishes, inferenceMs, NUTRITION_HEAD_UNTRAINED);
    }
    if (!this.bundle.targetTransform) {
      return this.withoutNutrition(dishes, inferenceMs, NO_TARGET_TRANSFORM);
    }

    // Averaged in real units, after the inverse transform. The model predicts in
    // standardised log space, and a mean there is a geometric mean of kilocalories, which
    // is not what "average these estimates" means to anyone reading the number.
    const transform = this.bundle.targetTransform;
    const perView = outputs.map((o) => toNutrition(o.nutrition, transform, quantiles.length));

    // Widened last, after averaging. The offsets were fitted against the model's output,
    // and applying them per view then averaging would apply them once and report them
    // three times.
    const averaged = averageNutrition(perView);
    const conformal = this.bundle.conformal;

    return {
      dishes,
      nutrition: conformal ? applyConformal(averaged, conformal) : averaged,
      nutritionUnavailableReason: null,
      inferenceMs,
    };
  }

  async release(): Promise<void> {
    await this.session.release?.();
  }

  private withoutNutrition(
    dishes: Analysis["dishes"],
    inferenceMs: number,
    reason: string,
  ): Analysis {
    return { dishes, nutrition: null, nutritionUnavailableReason: reason, inferenceMs };
  }
}
