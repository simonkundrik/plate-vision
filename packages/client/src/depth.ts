/**
 * Depth handling for the experimental LiDAR route.
 *
 * A TypeScript mirror of `model/platevision/depth.py`, plus the diagnostics that answer the
 * one question a device owner can settle and this machine cannot: **does a depth map from a
 * phone resemble the depth maps the model was trained on?**
 *
 * That question gates the whole depth track. The published result being chased is
 * Nutrition5k's own: handing depth to the network as a fourth input channel takes calorie
 * MAE from 70.6 to 47.6 kcal, a 33% reduction, and it is what the market leader ships via
 * iPhone LiDAR. But Nutrition5k depth comes from a **fixed overhead rig at roughly 3.5 m**,
 * and a phone is held at arm's length over a plate. A model trained on the first is not
 * automatically a model that works on the second, and nothing in this repository can tell
 * which it is without a capture from real hardware.
 *
 * The arithmetic says to expect trouble, which is why this module reports it rather than
 * hiding it. `normaliseDepth` divides by {@link MAX_DEPTH_MM}, so the rig's 3,562 mm median
 * lands at 0.594 and training standardises around a mean of 0.612 with a standard deviation
 * of 0.052. A phone at 300 mm normalises to 0.05, which is about **eleven standard
 * deviations** below anything the model has seen. Raw absolute depth from a phone is
 * therefore not expected to transfer at all. {@link heightAbove}, which re-expresses the map
 * as height above the surface behind the food, removes the camera distance and keeps the
 * shape, and it is the channel with a chance of surviving the move to a handheld device.
 *
 * Everything here is pure and unit tested. The native capture that feeds it is not, and
 * cannot be from a machine with no depth sensor.
 *
 * @module
 */

/**
 * Nothing in a Nutrition5k frame is further than this, and the rig sits near 3.5 m.
 * Clipping here turns saturated pixels into "far" rather than letting a single 65535
 * dominate the normalisation of an entire image. Mirrors `depth.MAX_DEPTH_MM`.
 */
export const MAX_DEPTH_MM = 6000;

/** Sensor dropouts, encoded as zero. Mirrors `depth.DROPOUT`. */
export const DROPOUT = 0;

/** The other end of the same failure: a pixel that saturated rather than one that dropped out. */
export const SATURATED = 65535;

/**
 * What the training data looks like, so a capture can be compared against it rather than
 * against an intuition.
 *
 * `mean` and `std` are `transforms.DEPTH_MEAN` and `DEPTH_STD`, measured over 300 maps in
 * the normalised [0, 1] space. The millimetre figures come from the same measurement pass
 * recorded in `depth.py`. `worstDropout` is the worst single map seen, not a limit.
 */
export const TRAINING_DEPTH = {
  /** Mean of the normalised map the model was standardised against. */
  mean: 0.612,
  /** Standard deviation of the same. Tight, because the rig never moves. */
  std: 0.052,
  medianMm: 3562,
  p5Mm: 3415,
  p95Mm: 4054,
  /** Roughly 16% of pixels are dropouts across the dataset. */
  meanDropout: 0.16,
  /** The worst map measured reached this. */
  worstDropout: 0.39,
} as const;

/**
 * A depth map in the encoding this project uses everywhere: **unsigned millimetres from the
 * camera**, with zero meaning the sensor returned nothing.
 *
 * The unit is not incidental. Nutrition5k stores uint16 millimetres, ARKit reports metres as
 * float32, and ARCore's `acquireDepthImage16Bits` reports uint16 millimetres. Converting at
 * the platform boundary means exactly one representation reaches this module and the Python
 * that mirrors it, rather than two conventions meeting somewhere in the middle.
 */
export type DepthMap = {
  /** Row-major depth in millimetres, length `width * height`. */
  data: Uint16Array;
  width: number;
  height: number;
};

const assertShape = (map: DepthMap): void => {
  const expected = map.width * map.height;
  if (map.data.length !== expected) {
    throw new Error(
      `depth map is ${map.width}x${map.height} but carries ${map.data.length} values, expected ${expected}`,
    );
  }
};

/** Valid depths, ascending. Everything below reads percentiles off this. */
const validSorted = (map: DepthMap): Float64Array => {
  const kept: number[] = [];
  for (let i = 0; i < map.data.length; i += 1) {
    if (map.data[i] > DROPOUT) kept.push(map.data[i]);
  }
  return Float64Array.from(kept).sort();
};

/**
 * Linear-interpolated percentile, matching numpy's default so this module and `depth.py`
 * cannot disagree about what the 90th percentile of the same map is.
 */
const percentile = (sorted: Float64Array, q: number): number => {
  if (sorted.length === 0) return Number.NaN;
  const position = (q / 100) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
};

/**
 * A raw depth map as float32 in [0, 1], dropouts filled.
 *
 * Dropouts are replaced with the median of the valid pixels rather than zero or the maximum.
 * Zero reads as "against the lens" and the maximum as "infinitely far", and both are strong,
 * wrong signals in a region where the sensor simply had nothing to say. The median is the
 * least informative value available, which is the point.
 *
 * A map with no valid pixels at all returns the midpoint, since there is nothing to take a
 * median of and the alternative is a division by zero.
 */
export const normaliseDepth = (map: DepthMap): Float32Array => {
  assertShape(map);
  const out = new Float32Array(map.data.length);
  const sorted = validSorted(map);

  if (sorted.length === 0) {
    out.fill(0.5);
    return out;
  }

  const fill = percentile(sorted, 50);
  for (let i = 0; i < map.data.length; i += 1) {
    const value = map.data[i] > DROPOUT ? map.data[i] : fill;
    out[i] = Math.min(Math.max(value, 0), MAX_DEPTH_MM) / MAX_DEPTH_MM;
  }
  return out;
};

/** Share of pixels the sensor did not report. Worth logging: it reaches 39% on the rig. */
export const dropoutFraction = (map: DepthMap): number => {
  assertShape(map);
  if (map.data.length === 0) return 0;
  let dropped = 0;
  for (let i = 0; i < map.data.length; i += 1) {
    if (map.data[i] === DROPOUT) dropped += 1;
  }
  return dropped / map.data.length;
};

/**
 * Depth re-expressed as height above the surface behind the food, in [0, 1].
 *
 * The four-channel model follows the paper and consumes the raw map, where absolute distance
 * on a fixed rig is nearly constant and the network can read the table's distance as a free
 * scale cue. A handheld phone reproduces none of that, so this is the transform that has a
 * chance of transferring: it subtracts the camera distance and keeps the shape.
 *
 * The reference surface is a high percentile of the valid depths rather than the maximum,
 * which would be a saturated pixel on most frames.
 */
export const heightAbove = (map: DepthMap, percentileRank = 90): Float32Array => {
  assertShape(map);
  const out = new Float32Array(map.data.length);
  const sorted = validSorted(map);
  if (sorted.length === 0) return out;

  const surface = percentile(sorted, percentileRank);
  for (let i = 0; i < map.data.length; i += 1) {
    const height = map.data[i] > DROPOUT ? surface - map.data[i] : 0;
    out[i] = Math.min(Math.max(height, 0), MAX_DEPTH_MM) / MAX_DEPTH_MM;
  }
  return out;
};

/** Everything a tester's capture can say about itself without the photograph leaving the device. */
export type DepthDescription = {
  width: number;
  height: number;
  pixels: number;
  validPixels: number;
  /** Share of pixels the sensor did not report. */
  dropoutFraction: number;
  /** Share that saturated instead, the same failure at the other end. */
  saturatedFraction: number;
  minMm: number;
  p5Mm: number;
  medianMm: number;
  p95Mm: number;
  maxMm: number;
  /** `medianMm / MAX_DEPTH_MM`: the value a four-channel model would actually receive. */
  normalisedMedian: number;
  /**
   * How many training standard deviations that sits from what the model was standardised
   * against. Large is expected on a phone and is the finding, not a fault in the capture.
   */
  sigmaFromTraining: number;
  /**
   * How far the food stands above the surface behind it, in millimetres: the 90th percentile
   * of valid depths minus the closest valid pixel. This is the signal depth is meant to add,
   * and it is scale-free in a way absolute distance is not.
   */
  reliefMm: number;
};

/** Measure a capture. Pure, and the numbers a tester reports come straight from here. */
export const describeDepth = (map: DepthMap): DepthDescription => {
  assertShape(map);
  const sorted = validSorted(map);

  let saturated = 0;
  for (let i = 0; i < map.data.length; i += 1) {
    if (map.data[i] >= SATURATED) saturated += 1;
  }

  const medianMm = percentile(sorted, 50);
  const normalisedMedian = medianMm / MAX_DEPTH_MM;
  const surface = percentile(sorted, 90);

  return {
    width: map.width,
    height: map.height,
    pixels: map.data.length,
    validPixels: sorted.length,
    dropoutFraction: dropoutFraction(map),
    saturatedFraction: map.data.length === 0 ? 0 : saturated / map.data.length,
    minMm: sorted.length === 0 ? Number.NaN : sorted[0],
    p5Mm: percentile(sorted, 5),
    medianMm,
    p95Mm: percentile(sorted, 95),
    maxMm: sorted.length === 0 ? Number.NaN : sorted[sorted.length - 1],
    normalisedMedian,
    sigmaFromTraining: (normalisedMedian - TRAINING_DEPTH.mean) / TRAINING_DEPTH.std,
    reliefMm: sorted.length === 0 ? Number.NaN : surface - sorted[0],
  };
};

/** One comparison between a capture and the training distribution. */
export type DistributionCheck = {
  name: string;
  /** Rendered value, already carrying its unit. */
  value: string;
  /** What the training data does, for the same quantity. */
  expected: string;
  /** Whether the capture falls where the model was fitted. */
  inDistribution: boolean;
  /**
   * Why it matters, in one line. Present on every check rather than only the failing ones,
   * because a tester reading a red row needs to know whether it is a broken capture or the
   * predicted and interesting result.
   */
  note: string;
};

/**
 * Compare a capture against the rig the model was trained on.
 *
 * A phone is **expected** to fail the absolute-distance check. That failure is the measurement
 * this feature exists to collect, not a defect in the capture, and the note says so, so nobody
 * files it as a bug or, worse, quietly discards a perfectly good capture for looking wrong.
 */
export const compareToTraining = (description: DepthDescription): DistributionCheck[] => {
  const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;
  const mm = (value: number): string => (Number.isNaN(value) ? "n/a" : `${Math.round(value)} mm`);

  return [
    {
      name: "Camera distance",
      value: mm(description.medianMm),
      expected: `${TRAINING_DEPTH.p5Mm} to ${TRAINING_DEPTH.p95Mm} mm (fixed rig)`,
      inDistribution:
        description.medianMm >= TRAINING_DEPTH.p5Mm && description.medianMm <= TRAINING_DEPTH.p95Mm,
      note:
        "A phone is held far closer than the rig, so this is expected to be out of range. " +
        "It is the reason the raw depth channel cannot be used as trained.",
    },
    {
      name: "Normalised median",
      value: description.normalisedMedian.toFixed(3),
      expected: `${TRAINING_DEPTH.mean} +/- ${TRAINING_DEPTH.std}`,
      inDistribution: Math.abs(description.sigmaFromTraining) <= 3,
      note: `${description.sigmaFromTraining.toFixed(1)} standard deviations from what the model was standardised against.`,
    },
    {
      name: "Sensor dropouts",
      value: pct(description.dropoutFraction),
      expected: `about ${pct(TRAINING_DEPTH.meanDropout)}, worst ${pct(TRAINING_DEPTH.worstDropout)}`,
      inDistribution: description.dropoutFraction <= TRAINING_DEPTH.worstDropout,
      note:
        "This one should match. Far more holes than the rig means the sensor is struggling " +
        "with the surface, and that would be a real obstacle rather than a rescalable one.",
    },
    {
      name: "Food relief",
      value: mm(description.reliefMm),
      expected: "non-zero, tens of mm for a plated meal",
      inDistribution: description.reliefMm > 5,
      note:
        "Height of the food above the surface behind it. This is the signal depth adds, and " +
        "unlike absolute distance it does not change with how the phone is held.",
    },
  ];
};
