import { Pressable, StyleSheet, Text, View } from "react-native";

import { colour, radius, space, type } from "../theme";

type Props = {
  title: string;
  detail: string;
  onRetry: () => void;
};

/**
 * Shown when there is no usable model.
 *
 * It states the actual reason rather than "something went wrong". Every failure this
 * screen renders is one the app diagnosed precisely: no model configured, a manifest it
 * could not read, a download whose size disagreed with what the manifest promised. Hiding
 * that behind a generic message would throw away the only information worth having.
 */
export const ProblemScreen = ({ title, detail, onRetry }: Props) => (
  <View style={styles.root}>
    <Text style={styles.title}>{title}</Text>
    <Text style={styles.detail}>{detail}</Text>
    <Pressable style={styles.button} onPress={onRetry} accessibilityRole="button">
      <Text style={styles.label}>Try again</Text>
    </Pressable>
  </View>
);

const styles = StyleSheet.create({
  root: { flex: 1, justifyContent: "center", padding: space(3), gap: space(2) },
  title: { color: colour.text, fontSize: type.title, fontWeight: "600" },
  detail: { color: colour.muted, fontSize: type.body, lineHeight: 22 },
  button: {
    marginTop: space(2),
    minHeight: 48,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: radius.small,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colour.border,
  },
  label: { color: colour.text, fontSize: type.body },
});
