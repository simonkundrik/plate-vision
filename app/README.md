# app

Expo React Native app for Android. Scaffolded in PR #13.

Planned stack:

- `expo-camera` for capture
- `onnxruntime-react-native` for on-device inference, pinned to the same ORT minor as the web demo
- `expo-dev-client`, because ONNX requires native modules and will not run inside Expo Go
- `expo-sqlite` for local meal history and portion-correction logging

Builds run on EAS cloud (15 Android builds per month on the free tier, 45 minute timeout). The
native shell is built once and JavaScript is iterated on top of it, so the quota is not a
constraint in practice.

The model ships as a bundled app asset and reads its contract from `shared/model_meta.json`.
