// Monorepo Metro config.
//
// The app depends on @plate-vision/client, which npm workspaces links as a symlink into
// the root node_modules. Metro only watches the project directory by default, so without
// this the symlink resolves to a path it refuses to read and the bundle fails at import.
//
// The previous version of this file pointed Metro at ../shared so the app could import the
// contract directly. It no longer does: the contract now arrives inside the client package,
// which is the only copy the app sees.
const path = require("path");
const { getDefaultConfig } = require("expo/metro-config");

const projectRoot = __dirname;
const repoRoot = path.resolve(projectRoot, "..");

const config = getDefaultConfig(projectRoot);

// Watch the whole repo so edits inside packages/client trigger a rebuild rather than
// silently serving a stale bundle.
config.watchFolders = [repoRoot];

// Hoisted dependencies live in the root node_modules. Both paths are listed, nearest
// first, because npm can leave a package unhoisted when versions conflict.
config.resolver.nodeModulesPaths = [
  path.resolve(projectRoot, "node_modules"),
  path.resolve(repoRoot, "node_modules"),
];

// A workspace symlink can otherwise be walked twice, giving two copies of React and the
// invalid-hook-call error that follows from it.
config.resolver.disableHierarchicalLookup = true;

module.exports = config;
