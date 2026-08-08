import { describe, expect, it } from "vitest";
import type { ArtifactDescriptor } from "@plate-vision/client";

import { cacheFileName, decideCache, verifyDownload } from "../cache";

const descriptor = (overrides: Partial<ArtifactDescriptor> = {}): ArtifactDescriptor => ({
  name: "plate-vision-fp32.onnx",
  bytes: 17280512,
  sha256: "abcdef0123456789".repeat(4),
  ...overrides,
});

describe("cacheFileName", () => {
  it("includes a hash prefix so a re-export cannot reuse the old file", () => {
    // The failure this prevents is silent. Caching by artifact name alone means a model
    // republished under the same name is never picked up, and the app keeps serving the
    // previous one while returning perfectly confident predictions.
    const first = cacheFileName(descriptor());
    const second = cacheFileName(descriptor({ sha256: "f".repeat(64) }));

    expect(first).not.toBe(second);
  });

  it("keeps the extension so the runtime still recognises the file", () => {
    expect(cacheFileName(descriptor())).toMatch(/\.onnx$/);
  });

  it("handles a name with no extension", () => {
    expect(cacheFileName(descriptor({ name: "model" }))).toBe(
      `model-${descriptor().sha256.slice(0, 12)}`,
    );
  });

  it("does not confuse a dot in a directory-like name for an extension", () => {
    expect(cacheFileName(descriptor({ name: "v1.2/model.onnx" }))).toMatch(/\.onnx$/);
  });
});

describe("decideCache", () => {
  it("downloads when nothing is cached", () => {
    expect(decideCache(descriptor(), { exists: false, size: 0 })).toEqual({
      action: "download",
      reason: "not cached yet",
    });
  });

  it("uses a cached file whose size matches the manifest", () => {
    expect(decideCache(descriptor(), { exists: true, size: 17280512 })).toEqual({
      action: "use",
    });
  });

  it("re-downloads a short file rather than loading it", () => {
    // A transfer cut partway leaves a file ONNX Runtime may still parse as a valid,
    // shorter graph. The predictions are then wrong in a way that looks like a bad model.
    const decision = decideCache(descriptor(), { exists: true, size: 9_000_000 });

    expect(decision.action).toBe("download");
    expect(decision).toMatchObject({ reason: expect.stringMatching(/interrupted/) });
  });

  it("re-downloads a file that is longer than promised", () => {
    expect(decideCache(descriptor(), { exists: true, size: 20_000_000 }).action).toBe(
      "download",
    );
  });

  it("refuses a manifest with no artifact digest", () => {
    // Accepting an unverifiable model is how a truncated download becomes a modelling
    // mystery weeks later.
    const decision = decideCache(null, { exists: true, size: 17280512 });

    expect(decision.action).toBe("refuse");
    expect(decision).toMatchObject({ reason: expect.stringMatching(/digest/) });
  });

  it("refuses a manifest declaring zero bytes", () => {
    expect(decideCache(descriptor({ bytes: 0 }), { exists: false, size: 0 }).action).toBe(
      "refuse",
    );
  });
});

describe("verifyDownload", () => {
  it("accepts a file of the promised size", () => {
    expect(verifyDownload(descriptor(), { exists: true, size: 17280512 })).toEqual({ ok: true });
  });

  it("rejects a download that produced nothing", () => {
    expect(verifyDownload(descriptor(), { exists: false, size: 0 }).ok).toBe(false);
  });

  it("rejects a size mismatch", () => {
    const result = verifyDownload(descriptor(), { exists: true, size: 1024 });
    expect(result).toMatchObject({ ok: false, reason: expect.stringContaining("1024") });
  });

  it("checks the hash when one is supplied", () => {
    const result = verifyDownload(descriptor(), { exists: true, size: 17280512 }, "0".repeat(64));
    expect(result.ok).toBe(false);
  });

  it("accepts a matching hash regardless of case", () => {
    const d = descriptor();
    const result = verifyDownload(d, { exists: true, size: d.bytes }, d.sha256.toUpperCase());
    expect(result).toEqual({ ok: true });
  });

  it("skips the hash check when none is supplied", () => {
    // Deliberate: size alone catches truncation, which is the common failure. Skipping the
    // hash is the caller's choice, not a default that silently weakens the check.
    const d = descriptor();
    expect(verifyDownload(d, { exists: true, size: d.bytes }, undefined)).toEqual({ ok: true });
  });
});
