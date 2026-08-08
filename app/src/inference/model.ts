/**
 * Acquiring the model artifact.
 *
 * Fetches the manifest, decides whether the cached file can be used, downloads it when it
 * cannot, and verifies what arrived. The decisions live in `cache.ts` and are tested; this
 * module is the filesystem and network around them and is not.
 */

import { artifactOf, parseBundle, type ModelBundle } from "@plate-vision/client";
import { Directory, File, Paths } from "expo-file-system";

import { MISSING_MODEL_SOURCE, modelSource } from "../config";
import { cacheFileName, decideCache, NO_ARTIFACT_DIGEST, verifyDownload } from "./cache";

export type ResolvedModel = {
  /** Local file URI the ONNX session loads. */
  path: string;
  bundle: ModelBundle;
};

export class ModelUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

/** Models live in cache, not documents: they are re-downloadable and should be evictable. */
const modelDirectory = () => new Directory(Paths.cache, "models");

const statOf = (file: File) => ({
  exists: file.exists,
  size: file.exists ? (file.size ?? 0) : 0,
});

export const resolveModel = async (): Promise<ResolvedModel> => {
  const source = modelSource();
  if (!source) throw new ModelUnavailableError(MISSING_MODEL_SOURCE);

  const response = await fetch(source.bundleUrl);
  if (!response.ok) {
    throw new ModelUnavailableError(
      `could not fetch the model manifest (${response.status} from ${source.bundleUrl})`,
    );
  }

  const raw = await response.json();
  // parseBundle throws on a manifest it cannot read rather than filling in defaults. A
  // model whose provenance is unknown is exactly what this project refuses to run.
  const bundle = parseBundle(raw);
  const descriptor = artifactOf(raw);

  const directory = modelDirectory();
  if (!directory.exists) directory.create({ intermediates: true });

  if (descriptor === null) throw new ModelUnavailableError(NO_ARTIFACT_DIGEST);

  const file = new File(directory, cacheFileName(descriptor));
  const decision = decideCache(descriptor, statOf(file));

  if (decision.action === "refuse") throw new ModelUnavailableError(decision.reason);

  if (decision.action === "download") {
    // Delete first: downloading over a partial file is how a short read becomes a
    // permanently poisoned cache entry.
    if (file.exists) file.delete();
    await File.downloadFileAsync(source.modelUrl, file);

    const verified = verifyDownload(descriptor, statOf(file));
    if (!verified.ok) {
      file.delete();
      throw new ModelUnavailableError(`model download failed verification: ${verified.reason}`);
    }
  }

  return { path: file.uri, bundle };
};
