import { quantiles } from "./contract";
import { toNutrition, topK } from "./postprocess";
import { createSession, runSession, type OrtLike, type OrtSession } from "./session";
import type { Analysis, LoadOptions, ModelBundle, RgbImage } from "./types";

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
  async analyse(image: RgbImage): Promise<Analysis> {
    const raw = await runSession(this.ort, this.session, image);
    const dishes = topK(raw.logits, this.topKCount);

    if (!this.bundle.headsTrained.nutritionQuantiles) {
      return this.withoutNutrition(dishes, raw.inferenceMs, NUTRITION_HEAD_UNTRAINED);
    }
    if (!this.bundle.targetTransform) {
      return this.withoutNutrition(dishes, raw.inferenceMs, NO_TARGET_TRANSFORM);
    }

    return {
      dishes,
      nutrition: toNutrition(raw.nutrition, this.bundle.targetTransform, quantiles.length),
      nutritionUnavailableReason: null,
      inferenceMs: raw.inferenceMs,
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
