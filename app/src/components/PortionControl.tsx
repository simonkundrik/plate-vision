import { Pressable, StyleSheet, Text, View } from "react-native";

import { colour, radius, space, type } from "../theme";

export const PORTION_STEPS = [0.5, 0.75, 1, 1.5, 2] as const;
export type PortionStep = (typeof PORTION_STEPS)[number];

type Props = {
  value: number;
  onChange: (value: PortionStep) => void;
};

/**
 * Lets the user correct the portion the model assumed.
 *
 * This is the most important control on the screen. Portion size is the dominant source of
 * calorie error and cannot be recovered from a photograph, so the honest design is to
 * predict a range and let the person holding the plate fix the part the model genuinely
 * cannot know.
 *
 * Discrete steps rather than a continuous slider, for two reasons. Nobody can judge "1.37
 * portions" by eye, so the extra resolution is false precision of a different kind. And a
 * slider means another native module and another rebuild for no gain in accuracy.
 */
export const PortionControl = ({ value, onChange }: Props) => (
  <View style={styles.row} accessibilityRole="radiogroup">
    {PORTION_STEPS.map((step) => {
      const selected = step === value;
      return (
        <Pressable
          key={step}
          onPress={() => onChange(step)}
          accessibilityRole="radio"
          accessibilityState={{ selected }}
          accessibilityLabel={`${step} portions`}
          style={[styles.step, selected && styles.stepSelected]}
        >
          <Text style={[styles.label, selected && styles.labelSelected]}>
            {step === 1 ? "1×" : `${step}×`}
          </Text>
        </Pressable>
      );
    })}
  </View>
);

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    gap: space(1),
  },
  step: {
    flex: 1,
    // 44 is the smallest reliably tappable target. Anything less and the control is a
    // decoration that happens to respond sometimes.
    minHeight: 44,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.small,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.border,
    backgroundColor: colour.surface,
  },
  stepSelected: {
    backgroundColor: colour.accent,
    borderColor: colour.accent,
  },
  label: {
    color: colour.muted,
    fontSize: type.body,
    fontVariant: ["tabular-nums"],
  },
  labelSelected: {
    color: colour.background,
    fontWeight: "600",
  },
});
