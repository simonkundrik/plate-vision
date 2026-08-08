import { StatusBar } from "expo-status-bar";
import { useCallback, useState } from "react";
import { ActivityIndicator, StyleSheet, View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { analyse } from "./src/inference";
import { CaptureScreen } from "./src/screens/CaptureScreen";
import { ResultScreen } from "./src/screens/ResultScreen";
import { colour } from "./src/theme";
import type { MealAnalysis } from "./src/types";

type Stage =
  | { name: "capture" }
  | { name: "working" }
  | { name: "result"; analysis: MealAnalysis };

export default function App() {
  const [stage, setStage] = useState<Stage>({ name: "capture" });

  const onCaptured = useCallback(async (uri: string) => {
    setStage({ name: "working" });
    const analysis = await analyse(uri);
    setStage({ name: "result", analysis });
  }, []);

  const onRetake = useCallback(() => setStage({ name: "capture" }), []);

  return (
    <SafeAreaProvider>
      <StatusBar style="light" />
      <View style={styles.root}>
        {stage.name === "capture" && <CaptureScreen onCaptured={onCaptured} />}
        {stage.name === "working" && (
          <View style={styles.centred}>
            <ActivityIndicator color={colour.accent} size="large" />
          </View>
        )}
        {stage.name === "result" && (
          <ResultScreen analysis={stage.analysis} onRetake={onRetake} />
        )}
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colour.background },
  centred: { flex: 1, alignItems: "center", justifyContent: "center" },
});
