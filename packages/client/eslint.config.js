import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "node_modules/**", "src/generated/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // The ONNX runtime modules are injected as `never` at the boundary, which is the
      // price of supporting two runtimes without depending on either.
      "@typescript-eslint/no-explicit-any": "error",

      // A leading underscore marks a parameter that exists for its type rather than its
      // value. Test doubles need declared parameters even when the body ignores them, or
      // vi.fn types the call record as an empty tuple and the request cannot be asserted on.
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    // Build scripts run under Node, not in the bundle. Declared here rather than pulling
    // in the `globals` package for two identifiers.
    files: ["scripts/**/*.mjs", "*.config.{js,ts}"],
    languageOptions: {
      globals: { console: "readonly", process: "readonly" },
    },
  },
);
