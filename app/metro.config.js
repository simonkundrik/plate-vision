// Metro only watches the project directory by default, so `shared/` above it is invisible
// and importing the contract would fail at bundle time.
//
// The alternative is copying model_meta.json into the app at build time, which creates a
// second copy that can go stale. The whole point of the contract is that there is exactly
// one of it, so Metro is pointed at the real file instead.
const path = require("path");
const { getDefaultConfig } = require("expo/metro-config");

const projectRoot = __dirname;
const repoRoot = path.resolve(projectRoot, "..");

const config = getDefaultConfig(projectRoot);

config.watchFolders = [path.resolve(repoRoot, "shared")];
config.resolver.nodeModulesPaths = [path.resolve(projectRoot, "node_modules")];

module.exports = config;
