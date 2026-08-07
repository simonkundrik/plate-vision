import { StyleSheet, View } from "react-native";

import { colour, radius } from "../theme";
import type { Interval } from "../types";

type Props = {
  interval: Interval;
  /** Upper bound of the drawn axis. Fixed across renders so bars stay comparable. */
  scaleMax: number;
};

/**
 * Draws the predicted range as a band with the median marked inside it.
 *
 * The band is the point. A single dot would say "620 kcal" and imply a precision the model
 * does not have; showing the span makes the uncertainty the first thing read rather than a
 * footnote. Width is proportional to a fixed axis maximum so a wide interval looks wide.
 */
export const IntervalBar = ({ interval, scaleMax }: Props) => {
  const clamp = (value: number) => Math.max(0, Math.min(1, value / scaleMax));

  const low = clamp(interval.low);
  const high = clamp(interval.high);
  const median = clamp(interval.median);

  return (
    <View style={styles.track} accessibilityRole="progressbar">
      <View
        style={[styles.band, { left: `${low * 100}%`, width: `${Math.max(high - low, 0.01) * 100}%` }]}
      />
      <View style={[styles.median, { left: `${median * 100}%` }]} />
    </View>
  );
};

const styles = StyleSheet.create({
  track: {
    height: 10,
    borderRadius: radius.pill,
    backgroundColor: colour.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.border,
    overflow: "hidden",
    justifyContent: "center",
  },
  band: {
    position: "absolute",
    top: 0,
    bottom: 0,
    backgroundColor: colour.accent,
    opacity: 0.35,
  },
  median: {
    position: "absolute",
    width: 2,
    top: 0,
    bottom: 0,
    backgroundColor: colour.accent,
  },
});
