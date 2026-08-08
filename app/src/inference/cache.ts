/**
 * Deciding whether a cached model file can be used.
 *
 * Kept free of expo-file-system so it can be tested. The parts of model acquisition that
 * actually go wrong are the decisions, not the I/O: caching under a name that ignores the
 * model's identity, or trusting a file whose download was cut short.
 */

import type { ArtifactDescriptor } from "@plate-vision/client";

export type CachedFile = {
  exists: boolean;
  /** Size in bytes. Meaningless when `exists` is false. */
  size: number;
};

export type CacheDecision =
  | { action: "use" }
  | { action: "download"; reason: string }
  | { action: "refuse"; reason: string };

/**
 * Where a given artifact is cached.
 *
 * The hash prefix is in the filename on purpose. Caching by artifact name alone means a
 * re-exported model published under the same name is never picked up: the app finds a file
 * of that name, uses it, and serves the previous model forever. Nothing about that failure
 * is visible, because the stale model still returns confident predictions.
 */
export const cacheFileName = (descriptor: ArtifactDescriptor): string => {
  const dot = descriptor.name.lastIndexOf(".");
  const stem = dot === -1 ? descriptor.name : descriptor.name.slice(0, dot);
  const extension = dot === -1 ? "" : descriptor.name.slice(dot);
  return `${stem}-${descriptor.sha256.slice(0, 12)}${extension}`;
};

/**
 * Whether the cached file at that path can be used as-is.
 *
 * Size is checked rather than assumed. A download interrupted partway leaves a shorter
 * file that ONNX Runtime may still parse as a valid graph, and the resulting predictions
 * are wrong in a way that looks like a bad model.
 */
/**
 * A manifest with no artifact block predates the digest, so there is nothing to check a
 * download against. Refusing is the honest response: silently accepting an unverifiable
 * model is how a truncated file becomes a modelling mystery weeks later.
 */
export const NO_ARTIFACT_DIGEST =
  "The model manifest does not declare an artifact digest, so a downloaded file cannot be " +
  "verified. Re-export with a current version of the exporter.";

export const decideCache = (
  descriptor: ArtifactDescriptor | null,
  cached: CachedFile,
): CacheDecision => {
  if (descriptor === null) {
    return { action: "refuse", reason: NO_ARTIFACT_DIGEST };
  }

  if (descriptor.bytes <= 0) {
    return { action: "refuse", reason: "The manifest declares an artifact of zero bytes." };
  }

  if (!cached.exists) {
    return { action: "download", reason: "not cached yet" };
  }

  if (cached.size !== descriptor.bytes) {
    return {
      action: "download",
      reason:
        `cached file is ${cached.size} bytes, the manifest declares ${descriptor.bytes}; ` +
        "treating it as an interrupted download",
    };
  }

  return { action: "use" };
};

/**
 * Whether a freshly downloaded file matches what the manifest promised.
 *
 * `sha256` is optional and the caller decides whether to compute it. Size alone catches a
 * truncated transfer, which is the common failure; it does not catch corruption or a
 * substituted file, which is what the hash is for. Passing `undefined` is a deliberate
 * choice to skip that, not a default that quietly weakens the check.
 */
export const verifyDownload = (
  descriptor: ArtifactDescriptor,
  downloaded: CachedFile,
  sha256?: string,
): { ok: true } | { ok: false; reason: string } => {
  if (!downloaded.exists) {
    return { ok: false, reason: "the download produced no file" };
  }
  if (downloaded.size !== descriptor.bytes) {
    return {
      ok: false,
      reason: `downloaded ${downloaded.size} bytes, expected ${descriptor.bytes}`,
    };
  }
  if (sha256 !== undefined && sha256.toLowerCase() !== descriptor.sha256.toLowerCase()) {
    return {
      ok: false,
      reason: `downloaded file hashes to ${sha256.slice(0, 12)}…, manifest declares ${descriptor.sha256.slice(0, 12)}…`,
    };
  }
  return { ok: true };
};
