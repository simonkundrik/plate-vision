import { StatusBar } from "expo-status-bar";
import { useCallback, useState } from "react";
import { ActivityIndicator, StyleSheet, View } from "react-native";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { analyse, isModelUnavailable } from "./src/inference";
import { CaptureScreen } from "./src/screens/CaptureScreen";
import { ProblemScreen } from "./src/screens/ProblemScreen";
import { ResultScreen } from "./src/screens/ResultScreen";
import { colour } from "./src/theme";
import type { MealAnalysis } from "./src/types";

type Stage =
  | { name: "capture" }
  | { name: "working" }
  | { name: "result"; analysis: MealAnalysis }
  | { name: "problem"; title: string; detail: string };

export default function App() {
  const [stage, setStage] = useState<Stage>({ name: "capture" });

  const onCaptured = useCallback(async (uri: string) => {
    setStage({ name: "working" });
    try {
      setStage({ name: "result", analysis: await analyse(uri) });
    } catch (error) {
      // A model that could not be loaded is a state the app understands and can explain.
      // Anything else is a bug, and saying so is more useful than dressing it up as a
      // configuration problem the user could act on.
      setStage({
        name: "problem",
        title: isModelUnavailable(error) ? "No model available" : "Analysis failed",
        detail: error instanceof Error ? error.message : String(error),
      });
    }
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
        {stage.name === "problem" && (
          <ProblemScreen title={stage.title} detail={stage.detail} onRetry={onRetake} />
        )}
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colour.background },
  centred: { flex: 1, alignItems: "center", justifyContent: "center" },
});
