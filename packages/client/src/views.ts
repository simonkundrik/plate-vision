/**
 * Test-time augmentation: analysing several views of one photo and averaging.
 *
 * Costs one forward pass per view and no training at all. Measured on the Nutrition5k test
 * split it is the only change in this project so far that improved accuracy *and*
 * calibration at once, rather than trading one for the other:
 *
 *     views   kcal MAE   median APE   coverage
 *         1       56.7        19.4%      82.2%
 *         3       54.0        18.7%      84.4%
 *         4       53.8        18.9%      85.0%
 *
 * Views are made by cropping and flipping the raw RGB. Nothing resamples here, because the
 * model resizes internally: a centre crop handed to the graph *is* a zoom, and doing the
 * scaling in client code would put an interpolation the model never trained against in
 * front of it.
 */

import type { RgbImage } from "./types";

/** How many views to average. One is plain inference. */
export const DEFAULT_VIEWS = 3;

export const MAX_VIEWS = 4;

/** Mirror an image horizontally, row by row. */
export const flipHorizontal = (image: RgbImage): RgbImage => {
  const { data, width, height } = image;
  const out = new Uint8Array(data.length);

  for (let y = 0; y < height; y += 1) {
    const row = y * width * 3;
    for (let x = 0; x < width; x += 1) {
      const from = row + x * 3;
      const to = row + (width - 1 - x) * 3;
      out[to] = data[from];
      out[to + 1] = data[from + 1];
      out[to + 2] = data[from + 2];
    }
  }

  return { data: out, width, height };
};

/**
 * Centre crop, keeping `fraction` of each dimension.
 *
 * Handing a smaller crop to a graph that resizes internally is a zoom in, and cropping is
 * the only way to do it without interpolating. A fraction above 1 is meaningless: there are
 * no pixels outside the photograph to include.
 */
export const centreCrop = (image: RgbImage, fraction: number): RgbImage => {
  if (!(fraction > 0 && fraction <= 1)) {
    throw new Error(`crop fraction must be in (0, 1], got ${fraction}`);
  }
  if (fraction === 1) return image;

  const { data, width, height } = image;
  const cropWidth = Math.max(1, Math.round(width * fraction));
  const cropHeight = Math.max(1, Math.round(height * fraction));
  const left = Math.floor((width - cropWidth) / 2);
  const top = Math.floor((height - cropHeight) / 2);

  const out = new Uint8Array(cropWidth * cropHeight * 3);
  for (let y = 0; y < cropHeight; y += 1) {
    const from = ((top + y) * width + left) * 3;
    out.set(data.subarray(from, from + cropWidth * 3), y * cropWidth * 3);
  }

  return { data: out, width: cropWidth, height: cropHeight };
};

/**
 * The views to average, most useful first.
 *
 * Ordered so that asking for two gets the flip and asking for three adds the zoom, which is
 * the ordering the measurement supports: the third view carried most of the gain.
 */
export const buildViews = (image: RgbImage, count: number): RgbImage[] => {
  if (!Number.isInteger(count) || count < 1) {
    throw new Error(`views must be a positive integer, got ${count}`);
  }

  const makers: ((source: RgbImage) => RgbImage)[] = [
    (source) => source,
    flipHorizontal,
    (source) => centreCrop(source, 0.91),
    (source) => flipHorizontal(centreCrop(source, 0.91)),
  ];

  return makers.slice(0, Math.min(count, makers.length)).map((make) => make(image));
};
