import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// Deployed under /plate-vision/ on GitHub Pages, served from / in dev.
const base = process.env.PAGES_BASE ?? "/";

export default defineConfig({
  base,
  plugins: [
    // onnxruntime-web loads its WebAssembly at runtime rather than through the bundler, so
    // the .wasm and .mjs files have to exist as real assets next to the page.
    viteStaticCopy({
      targets: [
        {
          src: "../node_modules/onnxruntime-web/dist/*.{wasm,mjs}",
          dest: "ort",
        },
      ],
    }),
  ],
  build: {
    // The model is a public/ asset of tens of megabytes. Warning about it on every build
    // trains you to ignore the warning that matters.
    chunkSizeWarningLimit: 2000,
  },
});
