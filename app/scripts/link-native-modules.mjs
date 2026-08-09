/**
 * Put a link to the hoisted native dependencies where CocoaPods will look for them.
 *
 * The Podfile that `expo prebuild` generates carries podspec paths **relative to the app
 * directory**, but CocoaPods resolves them relative to the Podfile, which lives in
 * `app/ios`. In a single-package project those are the same thing. In this npm workspace
 * they are not: npm hoists dependencies to the repository root, so the Podfile asks for
 * `../node_modules/onnxruntime-react-native` and CocoaPods reads that as
 * `app/node_modules/onnxruntime-react-native`, which does not exist.
 *
 *     [!] No podspec found for `onnxruntime-react-native` in `../node_modules/onnxruntime-react-native`
 *
 * The message names a package rather than a layout, so it reads like a broken dependency.
 * Autolinking itself is fine and reports the correct absolute path; only the relative form
 * written into the Podfile is off by one directory.
 *
 * A link inside `app/node_modules` makes the path CocoaPods computes land on the package.
 * This runs from the app's postinstall so it is already true after `npm ci`, rather than
 * being a step in the instructions that everyone building on a Mac has to remember.
 *
 * Windows is skipped: symlinks there need elevation or developer mode, and no iOS build
 * happens on Windows anyway, so failing the install would cost something and buy nothing.
 */

import { createRequire } from "node:module";
import { existsSync, lstatSync, mkdirSync, symlinkSync, unlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Only packages with an iOS podspec that autolinking picks up. Expo's own modules resolve
// through a different path that already handles hoisting, so they are not affected.
const PACKAGES = ["onnxruntime-react-native"];

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const localModules = path.join(appDir, "node_modules");
const require = createRequire(path.join(appDir, "package.json"));

if (process.platform === "win32") {
  process.exit(0);
}

mkdirSync(localModules, { recursive: true });

for (const name of PACKAGES) {
  let target;
  try {
    target = path.dirname(require.resolve(`${name}/package.json`));
  } catch {
    // Not installed. That is a dependency problem for npm to report, not something this
    // script should turn into a confusing link failure.
    continue;
  }

  const link = path.join(localModules, name);

  // Already where CocoaPods expects it, either really or through an earlier run.
  if (path.resolve(target) === path.resolve(link)) continue;
  if (existsSync(link)) {
    const existing = lstatSync(link);
    if (!existing.isSymbolicLink()) continue;
    unlinkSync(link);
  }

  symlinkSync(path.relative(localModules, target), link, "dir");
  console.log(`linked ${name} into app/node_modules`);
}
