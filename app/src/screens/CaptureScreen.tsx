import { CameraView, useCameraPermissions } from "expo-camera";
import { useRef, useState } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { colour, radius, space, type } from "../theme";

type Props = {
  onCaptured: (uri: string) => void;
  /**
   * Opens the experimental depth lab.
   *
   * Undefined unless the build was made with the flag set, and the control is not rendered
   * at all in that case. An always-present entry point that reports "not available" would
   * put an experiment in front of every user of a shipped build, which is the opposite of
   * what a flag is for.
   */
  onOpenDepthLab?: () => void;
};

export const CaptureScreen = ({ onCaptured, onOpenDepthLab }: Props) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [busy, setBusy] = useState(false);
  const camera = useRef<CameraView>(null);
  const insets = useSafeAreaInsets();

  if (!permission) {
    return (
      <View style={styles.centred}>
        <ActivityIndicator color={colour.accent} />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={[styles.centred, { paddingHorizontal: space(4) }]}>
        <Text style={styles.headline}>Camera access</Text>
        <Text style={styles.body}>
          The photo is analysed on this device and is not uploaded anywhere.
        </Text>
        <Pressable style={styles.primary} onPress={requestPermission} accessibilityRole="button">
          <Text style={styles.primaryLabel}>Allow camera</Text>
        </Pressable>
      </View>
    );
  }

  const capture = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const photo = await camera.current?.takePictureAsync({ quality: 0.9 });
      if (photo?.uri) onCaptured(photo.uri);
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={styles.container}>
      <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" />

      <View style={[styles.guidance, { paddingTop: insets.top + space(2) }]}>
        <Text style={styles.guidanceText}>Fill the frame with the plate</Text>
        {onOpenDepthLab && (
          <Pressable
            onPress={onOpenDepthLab}
            accessibilityRole="button"
            accessibilityLabel="Open the experimental depth lab"
            style={styles.depthEntry}
          >
            <Text style={styles.depthEntryText}>Depth lab (experimental)</Text>
          </Pressable>
        )}
      </View>

      {/* Controls sit above the home indicator rather than under it. Ignoring safe-area
          insets is the most common way a mobile layout looks almost right. */}
      <View style={[styles.controls, { paddingBottom: insets.bottom + space(3) }]}>
        <Pressable
          onPress={capture}
          disabled={busy}
          accessibilityRole="button"
          accessibilityLabel="Take photo"
          style={({ pressed }) => [styles.shutter, pressed && styles.shutterPressed]}
        >
          {busy ? <ActivityIndicator color={colour.background} /> : <View style={styles.shutterCore} />}
        </Pressable>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  centred: {
    flex: 1,
    backgroundColor: colour.background,
    alignItems: "center",
    justifyContent: "center",
    gap: space(2),
  },
  headline: { color: colour.text, fontSize: type.title, fontWeight: "600" },
  body: { color: colour.muted, fontSize: type.body, textAlign: "center", lineHeight: 24 },
  primary: {
    marginTop: space(2),
    minHeight: 48,
    paddingHorizontal: space(4),
    justifyContent: "center",
    borderRadius: radius.small,
    backgroundColor: colour.accent,
  },
  primaryLabel: { color: colour.background, fontSize: type.body, fontWeight: "600" },
  guidance: { paddingHorizontal: space(3), alignItems: "center", gap: space(1) },
  depthEntry: {
    minHeight: 44,
    justifyContent: "center",
    paddingHorizontal: space(2),
    borderRadius: radius.pill,
    backgroundColor: "rgba(0,0,0,0.45)",
  },
  depthEntryText: { color: colour.accent, fontSize: type.label, fontWeight: "600" },
  // A shadow rather than a translucent chip. A chip would be another floating card on a
  // screen whose subject is the camera feed.
  //
  // The typed props, not the `textShadow` shorthand. The web preview logs a deprecation
  // warning for `textShadow*`, but that warning also fires with this style deleted
  // entirely: it comes from react-native-web internals, which the Android build does not
  // use. Switching to the shorthand would require a cast, because the shipped types
  // reject it, in exchange for silencing a warning this code does not cause.
  guidanceText: {
    color: colour.text,
    fontSize: type.label,
    letterSpacing: 0.6,
    textTransform: "uppercase",
    textShadowColor: "rgba(0,0,0,0.75)",
    textShadowRadius: 6,
  },
  controls: { marginTop: "auto", alignItems: "center" },
  shutter: {
    width: 76,
    height: 76,
    borderRadius: radius.pill,
    borderWidth: 3,
    borderColor: "rgba(255,255,255,0.9)",
    alignItems: "center",
    justifyContent: "center",
  },
  shutterPressed: { opacity: 0.7 },
  shutterCore: {
    width: 60,
    height: 60,
    borderRadius: radius.pill,
    backgroundColor: colour.text,
  },
});
