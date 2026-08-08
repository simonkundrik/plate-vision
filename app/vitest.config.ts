import { defineConfig } from "vitest/config";

/**
 * Tests cover the parts of the app that are plain TypeScript: the model cache decisions
 * and the JPEG decode. Nothing here renders a component or touches a native module.
 *
 * That boundary is the point. The ONNX session, the filesystem, and the camera cannot run
 * outside a device build, so the logic they surround is kept in files that can, and the
 * untestable remainder is thin enough to read.
 */
export default defineConfig({
  test: {
    include: ["src/**/__tests__/**/*.test.ts"],
    environment: "node",
  },
});
