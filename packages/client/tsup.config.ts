import { defineConfig } from "tsup";

export default defineConfig({
  // Two entry points, one API. package.json exports resolves `index.native` through the
  // `react-native` condition so integrators import the same specifier on both platforms.
  entry: {
    index: "src/index.ts",
    "index.native": "src/index.native.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  // Both runtimes are optional peers. Bundling either would force every consumer to ship
  // a runtime they may not use, and would break React Native by pulling in the web build.
  external: ["onnxruntime-web", "onnxruntime-react-native"],
});
