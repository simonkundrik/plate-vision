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
  type NutritionEstimate,
} from "@plate-vision/client";

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
  const rows = (["protein", "fat", "carbohydrate", "mass"] as const)
    .map(
      (key) =>
        `<tr><th>${key[0].toUpperCase()}${key.slice(1)}</th>
         <td>${round(scaled[key].low)}&ndash;${round(scaled[key].high)} g</td></tr>`,
    )
    .join("");

  container.innerHTML = `
    <p class="range-label">Estimated energy</p>
    <p class="range">${round(scaled.energy.low)}&ndash;${round(scaled.energy.high)}
      <span class="unit">kcal</span></p>
    <p class="muted">most likely ${round(scaled.energy.median)} kcal</p>
    <table class="macros">${rows}</table>`;
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
      render(await pv.analyse(image));
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

  shoot.addEventListener("click", async () => {
    await analyse(video);
    (video.srcObject as MediaStream | null)?.getTracks().forEach((track) => track.stop());
    video.hidden = true;
    shoot.hidden = true;
  });
};

void main();
