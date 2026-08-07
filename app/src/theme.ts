/**
 * Visual language for the app.
 *
 * Deliberately restrained. The photograph of the food is the only colourful thing on
 * screen; everything around it is near-neutral so the picture reads as the subject rather
 * than competing with the interface.
 *
 * The accent is a warm amber rather than the blue-violet that mobile mockups default to.
 * It suits food, and it is not the colour every generated app arrives wearing.
 */

export const colour = {
  background: "#0B0B0C",
  surface: "#17181A",
  border: "#26282B",
  text: "#F2F2F0",
  muted: "#9A9A98",
  accent: "#E8A33D",
  warning: "#D9614C",
} as const;

/** 8pt grid. Every margin and padding in the app is a multiple of this. */
export const space = (steps: number): number => steps * 8;

/**
 * Body text is 16, not 13. Small type is the most common failure in generated mobile UI,
 * and this screen is read at arm's length while holding a plate of food.
 */
export const type = {
  display: 36,
  title: 22,
  body: 16,
  label: 13,
} as const;

export const radius = {
  small: 8,
  medium: 14,
  pill: 999,
} as const;
