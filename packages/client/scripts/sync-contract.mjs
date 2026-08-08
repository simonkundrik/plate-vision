// Copies the shared contract into the package so it ships inside the published tarball.
//
// A published package cannot reference files outside its own directory, so the contract
// has to be copied. The copies are committed and a test asserts they still match
// ../../shared, which is what stops the two from drifting apart silently. A drifted label
// ordering does not crash anything: it renames every prediction.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoShared = join(here, "..", "..", "..", "shared");
const generated = join(here, "..", "src", "generated");

mkdirSync(generated, { recursive: true });

for (const name of ["model_meta.json", "food101_labels.json"]) {
  const source = readFileSync(join(repoShared, name), "utf8");
  // Reserialised rather than copied verbatim so line endings and trailing whitespace
  // cannot make the drift test fail for reasons that do not matter.
  const normalised = `${JSON.stringify(JSON.parse(source), null, 2)}\n`;
  writeFileSync(join(generated, name), normalised, "utf8");
  console.log(`synced ${name}`);
}
