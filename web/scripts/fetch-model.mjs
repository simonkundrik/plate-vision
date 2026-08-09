// Fetch the published model into public/ so the demo serves it from its own origin.
//
// GitHub Release assets carry no Access-Control-Allow-Origin header, so a browser cannot
// fetch them cross-origin. React Native is not subject to CORS, which is why the app can
// use the release URLs directly and this cannot.
//
// The artifact stays out of the repository either way: this runs before dev and before the
// Pages deploy, and public/model is gitignored.

import { createWriteStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, "..", "public", "model");

const TAG = process.env.MODEL_TAG ?? "model-v0.2.0";
const base = `https://github.com/simonkundrik/plate-vision/releases/download/${TAG}`;

const get = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} fetching ${url}`);
  return response;
};

await mkdir(outDir, { recursive: true });

const manifest = await (await get(`${base}/bundle.json`)).json();
await writeFile(join(outDir, "bundle.json"), JSON.stringify(manifest, null, 2), "utf8");

const { name, bytes, sha256 } = manifest.artifact;
const target = join(outDir, name);

console.log(`fetching ${name} (${(bytes / 1024 ** 2).toFixed(2)} MB)`);
await pipeline(Readable.fromWeb((await get(`${base}/${name}`)).body), createWriteStream(target));

// The manifest declares a digest so a truncated download is caught. Checking it here means
// a bad artifact fails at build time rather than as a confusing runtime error in a browser.
const downloaded = await readFile(target);
const digest = createHash("sha256").update(downloaded).digest("hex");

if (downloaded.length !== bytes || digest !== sha256) {
  throw new Error(
    `downloaded artifact does not match the manifest:\n` +
      `  manifest ${bytes} bytes, sha256 ${sha256.slice(0, 12)}…\n` +
      `  file     ${downloaded.length} bytes, sha256 ${digest.slice(0, 12)}…`,
  );
}

console.log(`verified ${name}: ${downloaded.length} bytes, sha256 ${digest.slice(0, 12)}…`);
