import { useState } from "react";
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { IntervalBar } from "../components/IntervalBar";
import { PORTION_STEPS, PortionControl, type PortionStep } from "../components/PortionControl";
import { colour, radius, space, type } from "../theme";
import { scaleInterval, type Interval, type MealAnalysis } from "../types";

type Props = {
  analysis: MealAnalysis;
  onRetake: () => void;
};

const round = (value: number) => Math.round(value);

/** Axis maximum for the energy bar. Fixed so a wide interval visibly reads as wide. */
const ENERGY_AXIS_MAX = 1600;

export const ResultScreen = ({ analysis, onRetake }: Props) => {
  const [portion, setPortion] = useState<PortionStep>(1);
  const insets = useSafeAreaInsets();

  const top = analysis.dishes[0];

  // The library returns null nutrition when the loaded artifact's nutrition head is not
  // trained, rather than handing back numbers from random weights. The screen has to render
  // that state, because the alternative is an app that shows a calorie figure whenever one
  // is technically available regardless of whether it means anything.
  const nutrition = analysis.nutrition;
  const energy = nutrition ? scaleInterval(nutrition.energy, portion) : null;

  const macros: [string, Interval, string][] = nutrition
    ? [
        ["Protein", scaleInterval(nutrition.protein, portion), "g"],
        ["Fat", scaleInterval(nutrition.fat, portion), "g"],
        ["Carbohydrate", scaleInterval(nutrition.carbohydrate, portion), "g"],
        ["Mass", scaleInterval(nutrition.mass, portion), "g"],
      ]
    : [];

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={{ paddingBottom: insets.bottom + space(4) }}
    >
      <Image source={{ uri: analysis.photoUri }} style={styles.photo} resizeMode="cover" />

      <View style={styles.body}>
        {analysis.placeholder && (
          <View style={styles.notice}>
            <Text style={styles.noticeText}>
              Placeholder values. The model is not wired in yet, so these numbers describe
              nothing.
            </Text>
          </View>
        )}

        <Text style={styles.dish}>{top?.label ?? "Unrecognised"}</Text>
        <Text style={styles.confidence}>
          {top ? `${Math.round(top.confidence * 100)}% confidence` : "no confident match"}
          {analysis.dishes[1] ? `  ·  or ${analysis.dishes[1].label}` : ""}
        </Text>

        {energy ? (
          <>
            {/* The range is the headline, not a single number. A photo carries no scale
                reference, so a point estimate would be precision the model does not have. */}
            <Text style={styles.rangeLabel}>Estimated energy</Text>
            <Text style={styles.range}>
              {round(energy.low)}–{round(energy.high)}
              <Text style={styles.unit}> kcal</Text>
            </Text>
            <Text style={styles.median}>most likely {round(energy.median)} kcal</Text>

            <View style={styles.bar}>
              <IntervalBar interval={energy} scaleMax={ENERGY_AXIS_MAX} />
            </View>

            <Text style={styles.sectionLabel}>Portion</Text>
            <PortionControl value={portion} onChange={setPortion} />
            <Text style={styles.hint}>
              Portion size is the largest source of error and cannot be read from a
              photograph. If this looks wrong, it probably is. Correct it here.
            </Text>

            <View style={styles.macros}>
              {macros.map(([label, interval, unit]) => (
                <View key={label} style={styles.macroRow}>
                  <Text style={styles.macroLabel}>{label}</Text>
                  <Text style={styles.macroValue}>
                    {round(interval.low)}–{round(interval.high)} {unit}
                  </Text>
                </View>
              ))}
            </View>
          </>
        ) : (
          // No blank figures and no zeroes. An empty calorie row invites the reading that
          // the meal has no calories; saying why there is no number does not.
          <View style={styles.withheld}>
            <Text style={styles.withheldTitle}>No calorie estimate</Text>
            <Text style={styles.withheldBody}>
              {analysis.nutritionUnavailableReason ??
                "This model cannot produce nutrition figures."}
            </Text>
          </View>
        )}

        <Pressable style={styles.secondary} onPress={onRetake} accessibilityRole="button">
          <Text style={styles.secondaryLabel}>Take another photo</Text>
        </Pressable>

        <Text style={styles.footer}>
          Analysed on this device in {analysis.inferenceMs} ms. Nothing was uploaded.
        </Text>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colour.background },
  // The photo runs edge to edge and is the only saturated thing on screen.
  photo: { width: "100%", aspectRatio: 4 / 3, backgroundColor: colour.surface },
  body: { padding: space(3), gap: space(1) },
  notice: {
    padding: space(2),
    marginBottom: space(1),
    borderRadius: radius.small,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.warning,
  },
  noticeText: { color: colour.warning, fontSize: type.label, lineHeight: 19 },
  dish: { color: colour.text, fontSize: type.title, fontWeight: "600" },
  confidence: { color: colour.muted, fontSize: type.label, marginBottom: space(2) },
  rangeLabel: {
    color: colour.muted,
    fontSize: type.label,
    letterSpacing: 0.6,
    textTransform: "uppercase",
  },
  range: {
    color: colour.text,
    fontSize: type.display,
    fontWeight: "700",
    fontVariant: ["tabular-nums"],
  },
  unit: { fontSize: type.title, fontWeight: "500", color: colour.muted },
  median: { color: colour.muted, fontSize: type.body },
  bar: { marginTop: space(2), marginBottom: space(3) },
  sectionLabel: {
    color: colour.muted,
    fontSize: type.label,
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginBottom: space(1),
  },
  hint: { color: colour.muted, fontSize: type.label, lineHeight: 19, marginTop: space(1) },
  // Deliberately quiet rather than alarming. Nothing has gone wrong; the model simply
  // cannot answer this part, and shouting about it would read as an error state.
  withheld: {
    marginTop: space(2),
    padding: space(2),
    borderRadius: radius.small,
    backgroundColor: colour.surface,
  },
  withheldTitle: {
    color: colour.text,
    fontSize: type.body,
    fontWeight: "600",
    marginBottom: space(0.5),
  },
  withheldBody: { color: colour.muted, fontSize: type.label, lineHeight: 19 },
  macros: {
    marginTop: space(3),
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colour.border,
  },
  // A plain list with rules, not four floating cards. Cards would add chrome without
  // adding information.
  macroRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: space(1.5),
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colour.border,
  },
  macroLabel: { color: colour.muted, fontSize: type.body },
  macroValue: { color: colour.text, fontSize: type.body, fontVariant: ["tabular-nums"] },
  secondary: {
    marginTop: space(4),
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.small,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.border,
  },
  secondaryLabel: { color: colour.text, fontSize: type.body },
  footer: {
    color: colour.muted,
    fontSize: type.label,
    textAlign: "center",
    marginTop: space(3),
  },
});

export { PORTION_STEPS };
