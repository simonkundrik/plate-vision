// Flat config. `expo lint` would scaffold this interactively; writing it out keeps the
// setup reproducible from the repository rather than from a prompt someone answered once.
const expo = require("eslint-config-expo/flat");

module.exports = [
  ...expo,
  {
    ignores: ["node_modules/**", "dist/**", ".expo/**", "android/**", "ios/**"],
  },
];
