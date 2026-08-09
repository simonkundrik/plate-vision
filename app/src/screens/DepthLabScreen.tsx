import { compareToTraining, describeDepth } from "@plate-vision/client/depth";
import type { DepthDescription, DistributionCheck } from "@plate-vision/client/depth";
import Constants from "expo-constants";
import { useCallback, useState } from "react";
import {
  ActivityIndicator,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { captureDepth, depthSupport } from "../../modules/depth-capture";
import { buildIssueUrl, formatMeasurements, type CaptureContext } from "../depth/report";
import { colour, radius, space, type } from "../theme";

type State =
  | { name: "idle" }
  | { name: "capturing" }
  | { name: "measured"; description: DepthDescription; checks: DistributionCheck[]; report: string }
  | { name: "failed"; detail: string };

/**
 * The depth lab: an experiment that measures a sensor, not a feature that estimates calories.
 *
 * It exists because of a gap nobody working on this project can close. Nutrition5k's own
 * result is that feeding depth to the network as a fourth channel takes calorie MAE from
 * 70.6 to 47.6 kcal, and it is what the market leader ships via iPhone LiDAR. But that depth
 * comes from a fixed overhead rig at about 3.5 m, and a phone is held over a plate at arm's
 * length. Whether the two are close enough for a model trained on the first to work on the
 * second is an empirical question that needs a LiDAR device to answer, and this project has
 * none.
 *
 * So the screen takes one depth frame, measures it against the training distribution, shows
 * every number, and offers to file them. It does not touch the calorie estimate, and it says
 * so on screen rather than leaving a user to infer it from an absence.
 */
export const DepthLabScreen = ({ onBack }: { onBack: () => void }) => {
  const [state, setState] = useState<State>({ name: "idle" });
  const insets = useSafeAreaInsets();
  const support = depthSupport();

  const run = useCallback(async () => {
    setState({ name: "capturing" });
    try {
      const capture = await captureDepth();
      const description = describeDepth(capture.map);
      const checks = compareToTraining(description);
      const context: CaptureContext = {
        sensor: capture.sensor,
        filtered: capture.filtered,
        accuracy: capture.accuracy,
        platform: `${Platform.OS} ${Platform.Version}`,
        appVersion: Constants.expoConfig?.version ?? "unknown",
      };
      setState({
        name: "measured",
        description,
        checks,
        report: formatMeasurements(description, checks, context),
      });
    } catch (error) {
      // A failure here is as much of a result as a success, and more likely to be the
      // interesting one. It gets the same report path rather than a dead end.
      setState({ name: "failed", detail: error instanceof Error ? error.message : String(error) });
    }
  }, []);

  const report = useCallback((body: string, title: string) => {
    void Linking.openURL(buildIssueUrl(body, title));
  }, []);

  return (
    <ScrollView
      style={styles.root}
      contentContainerStyle={[
        styles.content,
        { paddingTop: insets.top + space(2), paddingBottom: insets.bottom + space(4) },
      ]}
    >
      <Text style={styles.badge}>Experimental</Text>
      <Text style={styles.title}>Depth lab</Text>
      <Text style={styles.body}>
        This measures your device&apos;s depth sensor against the depth data the model was
        trained on. It does not change any calorie estimate, and nothing it produces is used
        by the rest of the app.
      </Text>
      <Text style={styles.body}>
        The training data came from a fixed camera about 3.5 m above the food. A phone is held
        far closer, so the distance check below is expected to fail. That result is the point:
        it is what tells us whether the depth channel can be made to work on a phone at all.
      </Text>
      <Text style={styles.body}>
        Worth saying plainly: training a model on that fixed-camera depth has already been
        tried, and it made calorie estimates worse rather than better. What you measure here
        is still unmeasured elsewhere, but it is a smaller question than it looks.
      </Text>
      <Text style={styles.body}>
        No photo and no depth map leave your device. Only the summary shown below is sent, and
        only if you tap through to file it.
      </Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>{support.supported ? "Sensor found" : "No depth sensor"}</Text>
        <Text style={styles.detail}>{support.reason}</Text>
        {support.sensor ? <Text style={styles.detail}>{support.sensor}</Text> : null}
      </View>

      {support.supported && (
        <Pressable
          style={[styles.primary, state.name === "capturing" && styles.disabled]}
          onPress={run}
          disabled={state.name === "capturing"}
          accessibilityRole="button"
        >
          {state.name === "capturing" ? (
            <ActivityIndicator color={colour.background} />
          ) : (
            <Text style={styles.primaryLabel}>Capture a depth frame</Text>
          )}
        </Pressable>
      )}

      {state.name === "measured" && (
        <>
          <View style={styles.card}>
            {state.checks.map((check) => (
              <View key={check.name} style={styles.check}>
                <View style={styles.checkHead}>
                  <Text style={styles.checkName}>{check.name}</Text>
                  <Text style={[styles.checkValue, !check.inDistribution && styles.out]}>
                    {check.value}
                  </Text>
                </View>
                <Text style={styles.detail}>Training data: {check.expected}</Text>
                <Text style={styles.detail}>{check.note}</Text>
              </View>
            ))}
          </View>

          <Text style={styles.sectionTitle}>What would be sent</Text>
          <Text style={styles.mono}>{state.report}</Text>

          <Pressable
            style={styles.secondary}
            onPress={() => report(state.report, "Depth capture result")}
            accessibilityRole="button"
          >
            <Text style={styles.secondaryLabel}>Report this result</Text>
          </Pressable>
        </>
      )}

      {state.name === "failed" && (
        <>
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Capture failed</Text>
            <Text style={styles.detail}>{state.detail}</Text>
          </View>
          <Pressable
            style={styles.secondary}
            onPress={() =>
              report(
                `platform:      ${Platform.OS} ${Platform.Version}\n` +
                  `app:           ${Constants.expoConfig?.version ?? "unknown"}\n` +
                  `sensor:        ${support.sensor || "unknown"}\n\n` +
                  `capture failed: ${state.detail}`,
                "Depth capture failed",
              )
            }
            accessibilityRole="button"
          >
            <Text style={styles.secondaryLabel}>Report this failure</Text>
          </Pressable>
        </>
      )}

      <Pressable style={styles.tertiary} onPress={onBack} accessibilityRole="button">
        <Text style={styles.secondaryLabel}>Back to camera</Text>
      </Pressable>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colour.background },
  content: { paddingHorizontal: space(3), gap: space(2) },
  badge: {
    color: colour.accent,
    fontSize: type.label,
    letterSpacing: 1,
    textTransform: "uppercase",
    fontWeight: "600",
  },
  title: { color: colour.text, fontSize: type.title, fontWeight: "600" },
  sectionTitle: { color: colour.text, fontSize: type.body, fontWeight: "600", marginTop: space(1) },
  body: { color: colour.muted, fontSize: type.body, lineHeight: 24 },
  detail: { color: colour.muted, fontSize: type.label, lineHeight: 19 },
  card: {
    backgroundColor: colour.surface,
    borderRadius: radius.medium,
    padding: space(2),
    gap: space(1),
  },
  cardTitle: { color: colour.text, fontSize: type.body, fontWeight: "600" },
  check: { gap: space(0.5) },
  checkHead: { flexDirection: "row", justifyContent: "space-between", alignItems: "baseline" },
  checkName: { color: colour.text, fontSize: type.body },
  checkValue: { color: colour.text, fontSize: type.body, fontWeight: "600" },
  // Amber rather than red. An out-of-distribution reading here is the expected finding on a
  // phone, not an error, and colouring it as a fault would teach testers to discard exactly
  // the captures worth having.
  out: { color: colour.accent },
  mono: {
    color: colour.muted,
    fontSize: type.label,
    lineHeight: 18,
    fontFamily: Platform.select({ ios: "Menlo", default: "monospace" }),
    backgroundColor: colour.surface,
    borderRadius: radius.small,
    padding: space(2),
  },
  primary: {
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.small,
    backgroundColor: colour.accent,
  },
  disabled: { opacity: 0.6 },
  primaryLabel: { color: colour.background, fontSize: type.body, fontWeight: "600" },
  secondary: {
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.small,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.border,
  },
  tertiary: { minHeight: 48, alignItems: "center", justifyContent: "center", marginTop: space(1) },
  secondaryLabel: { color: colour.text, fontSize: type.body },
});
