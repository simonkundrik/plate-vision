/**
 * The browser demo.
 *
 * Uses the same `@plate-vision/client` code the app does. The runtime is imported here and
 * injected rather than left to the library's `load()`, because onnxruntime-web needs its
 * WebAssembly location configured before a session is created.
 */

import * as ort from "onnxruntime-web";
import {
  PlateVision,
  parseBundle,
  decodeImage,
  scaleNutrition,
  type Analysis,
  type EvidenceRoute,
  type Interval,
  type NutritionEstimate,
} from "@plate-vision/client";
// Separate entry point on purpose: this is the only import in the project that touches the
// network, and burying it in the main barrel would make that easy to miss.
import { isValidBarcode, lookupBarcode, nutritionFromProduct } from "@plate-vision/client/barcode";

const MODEL_DIR = `${import.meta.env.BASE_URL}model/`;

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
};

const status = el("status");
const controls = el("controls");
const stage = el("stage");
const result = el("result");
const preview = el<HTMLImageElement>("preview");
const video = el<HTMLVideoElement>("video");
const shoot = el<HTMLButtonElement>("shoot");

const say = (message: string, tone: "info" | "error" = "info") => {
  status.textContent = message;
  status.classList.toggle("error", tone === "error");
  status.hidden = false;
};

// Single threaded and no SIMD assumptions. Cross-origin isolation is required for threads,
// which means COOP/COEP headers GitHub Pages does not send; asking for threads anyway
// fails at session creation rather than degrading.
ort.env.wasm.wasmPaths = `${import.meta.env.BASE_URL}ort/`;
ort.env.wasm.numThreads = 1;

const loadModel = async (): Promise<PlateVision> => {
  const manifest = await (await fetch(`${MODEL_DIR}bundle.json`)).json();
  // Throws on a manifest it cannot read rather than guessing at a model's provenance.
  const bundle = parseBundle(manifest);

  return PlateVision.load(ort as never, {
    model: `${MODEL_DIR}${manifest.artifact.name}`,
    bundle,
  });
};

const percent = (value: number) => `${Math.round(value * 100)}%`;
const round = (value: number) => Math.round(value);

/**
 * What each route means, in the words someone reading a calorie figure needs.
 *
 * Shown on every estimate rather than only the interesting ones. A number whose provenance
 * appears only when it is flattering is worse than no label at all, because the absence
 * stops meaning anything.
 */
const ROUTE_NOTE: Record<EvidenceRoute, string> = {
  barcode: "Read off the packet. The composition is stated, not estimated.",
  chain_menu: "From the restaurant's published figures.",
  scale_reference: "Portion size recovered from an object of known size in the photo.",
  absolute: "Estimated from the photograph alone. Median error is about 19% on the test set.",
};

const renderNutrition = (nutrition: NutritionEstimate | null, reason: string | null) => {
  const container = el("nutrition");

  if (!nutrition) {
    // Not an error state. The model cannot answer this part, and saying why is more useful
    // than an empty row, which would invite the reading that the meal has no calories.
    container.innerHTML = `
      <div class="withheld">
        <h3>No calorie estimate</h3>
        <p>${reason ?? "This model cannot produce nutrition figures."}</p>
      </div>`;
    return;
  }

  const scaled = scaleNutrition(nutrition, 1);

  // A range is a claim about uncertainty. When there is none, "2-2 g" states it twice and
  // reads as a formatting bug rather than as a quantity somebody knows.
  const span = (interval: Interval, unit: string) =>
    round(interval.low) === round(interval.high)
      ? `${round(interval.median)} ${unit}`
      : `${round(interval.low)}&ndash;${round(interval.high)} ${unit}`;

  const rows = (["protein", "fat", "carbohydrate", "mass"] as const)
    .map(
      (key) =>
        `<tr><th>${key[0].toUpperCase()}${key.slice(1)}</th>
         <td>${span(scaled[key], "g")}</td></tr>`,
    )
    .join("");

  // A zero-width interval is not the model hedging less, it is the user having weighed the
  // food. Printing "120-120 kcal" reads as a rounding artifact rather than a known answer.
  const known = scaled.energy.low === scaled.energy.high;

  container.innerHTML = `
    <p class="range-label">${known ? "Energy" : "Estimated energy"}</p>
    <p class="range">${span(scaled.energy, "")}<span class="unit">kcal</span></p>
    <p class="muted">${known ? "from a stated mass" : `most likely ${round(scaled.energy.median)} kcal`}</p>
    <table class="macros">${rows}</table>
    <p class="route"><span class="route-tag">${nutrition.route.replace("_", " ")}</span>
      ${ROUTE_NOTE[nutrition.route]}</p>`;
};

/** The model's own mass estimate, when it has one, so the barcode route can reuse it. */
let predictedMass: Interval | null = null;

/**
 * How much of the product was eaten.
 *
 * A typed number is a **stated** mass, so it becomes a zero-width interval and the calorie
 * answer is exact: the packet is not guessing about composition and the user is not guessing
 * about the portion. Leaving it blank falls back to the model's estimate, which is where all
 * the remaining uncertainty then lives.
 */
const massForLookup = (typed: string): Interval | string => {
  const trimmed = typed.trim();
  if (trimmed) {
    const grams = Number(trimmed);
    if (!Number.isFinite(grams) || grams <= 0) return "Grams has to be a positive number.";
    return { low: grams, median: grams, high: grams };
  }
  if (predictedMass) return predictedMass;
  return "This model cannot estimate portion mass, so enter the grams you ate.";
};

const render = (analysis: Analysis) => {
  const [top, ...rest] = analysis.dishes;

  el("dish").textContent = top ? top.label : "Unrecognised";
  el("confidence").textContent = top
    ? `${percent(top.confidence)} confidence`
    : "no confident match";

  el("alternatives").innerHTML = rest
    .map((dish) => `<li>${dish.label} <span class="muted">${percent(dish.confidence)}</span></li>`)
    .join("");

  renderNutrition(analysis.nutrition, analysis.nutritionUnavailableReason);

  predictedMass = analysis.nutrition?.mass ?? null;
  if (predictedMass) {
    el<HTMLInputElement>("grams").placeholder = `${round(predictedMass.median)} (estimated)`;
  }
  el("barcode-status").textContent = "";

  el("timing").textContent = `Analysed in this tab in ${Math.round(analysis.inferenceMs)} ms. Nothing was uploaded.`;
  result.hidden = false;
};

const main = async () => {
  let pv: PlateVision;
  try {
    pv = await loadModel();
  } catch (error) {
    say(error instanceof Error ? error.message : String(error), "error");
    return;
  }

  status.hidden = true;
  controls.hidden = false;

  const analyse = async (source: Blob | HTMLVideoElement) => {
    say("Analysing…");
    try {
      const image = await decodeImage(source as never);
      // Three views: measured at 54.0 kcal MAE against 56.7 for one, and 84.4%
      // coverage against 82.2%. Costs three forward passes.
      render(await pv.analyse(image, { views: 3 }));
      status.hidden = true;
    } catch (error) {
      say(error instanceof Error ? error.message : String(error), "error");
    }
  };

  el<HTMLInputElement>("file").addEventListener("change", async (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (!file) return;

    preview.src = URL.createObjectURL(file);
    stage.hidden = false;
    video.hidden = true;
    shoot.hidden = true;
    await analyse(file);
  });

  el("camera").addEventListener("click", async () => {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      await video.play();
      stage.hidden = false;
      video.hidden = false;
      shoot.hidden = false;
      preview.removeAttribute("src");
    } catch {
      say("Could not open the camera. Choosing a photo works the same way.", "error");
    }
  });

  el("lookup").addEventListener("click", async () => {
    const note = el("barcode-status");
    const mass = massForLookup(el<HTMLInputElement>("grams").value);
    if (typeof mass === "string") {
      note.textContent = mass;
      return;
    }

    // Validated before the request is announced, not just before it is made. `lookupBarcode`
    // already declines to send a malformed code, so saying "asking Open Food Facts" first
    // would claim a request that never happens, on the one control here that leaves the
    // device. Momentary and still false.
    const code = el<HTMLInputElement>("barcode").value;
    if (!isValidBarcode(code)) {
      note.textContent = `${code.trim() || "That"} is not a barcode. Expected 8 to 14 digits.`;
      return;
    }

    note.textContent = "Asking Open Food Facts…";
    const found = await lookupBarcode(code);

    if (!found.found) {
      // A miss is an ordinary outcome for a community database, roughly a third of scans.
      // Saying which barcode missed beats a bare "not found".
      note.textContent = found.reason;
      return;
    }

    const { product } = found;
    // Open Food Facts often repeats the product name as the brand, and "Nutella (Nutella)"
    // reads as a bug in this page rather than as a quirk of the database.
    const brand = product.brand && product.brand !== product.name ? ` (${product.brand})` : "";
    note.textContent = `${product.name}${brand} — ${round(product.energyPer100g)} kcal per 100 g`;
    renderNutrition(nutritionFromProduct(product, mass), null);
  });

  shoot.addEventListener("click", async () => {
    await analyse(video);
    (video.srcObject as MediaStream | null)?.getTracks().forEach((track) => track.stop());
    video.hidden = true;
    shoot.hidden = true;
  });
};

void main();
