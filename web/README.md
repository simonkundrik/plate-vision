# web

Browser demo. Scaffolded in PR #16.

Runs the exact same `.onnx` artifact as the Android app, via `onnxruntime-web` with the WASM
backend and WebGPU where available. Camera through `getUserMedia`, plus drag-and-drop upload for
desktop.

This exists so the project has a link that works instantly on any device with no install step. It
is not a second implementation: the model, the preprocessing (compiled into the graph), and the
contract in `shared/model_meta.json` are shared with the app.

Deployed to GitHub Pages by a workflow. Model binaries are fetched from GitHub Releases rather than
committed, to avoid Git LFS bandwidth limits.
