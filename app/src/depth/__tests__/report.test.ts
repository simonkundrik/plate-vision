import { compareToTraining, describeDepth } from "@plate-vision/client/depth";
import { describe, expect, it } from "vitest";

import { buildIssueUrl, formatMeasurements, ISSUE_FIELDS, type CaptureContext } from "../report";

const context: CaptureContext = {
  sensor: "Back LiDAR Camera",
  filtered: false,
  accuracy: "absolute",
  platform: "ios 18.2",
  appVersion: "0.1.0",
};

/** A plate 40 mm proud of a table 300 mm from the lens, with one dropout. */
const phone = describeDepth({
  data: Uint16Array.from([260, 300, 300, 0]),
  width: 4,
  height: 1,
});

describe("formatMeasurements", () => {
  const body = formatMeasurements(phone, compareToTraining(phone), context);

  it("carries the numbers rather than a description of them", () => {
    expect(body).toContain("depth median:  300 mm");
    expect(body).toContain("relief:        40 mm");
    expect(body).toContain("dropouts:      25.0%");
  });

  it("states whether the values are metric at all", () => {
    // A `relative` capture is a different quantity, not a worse one, and every millimetre
    // above would be meaningless without this line.
    expect(body).toContain("accuracy:      absolute");
    expect(body).toContain("filtered:      false");
  });

  it("ties the report to the code that produced it", () => {
    expect(body).toContain("platform:      ios 18.2");
    expect(body).toContain("app:           0.1.0");
  });

  it("marks the distance check as out of distribution", () => {
    expect(body).toContain("out  Camera distance: 300 mm");
  });

  it("contains no pixel data", () => {
    // The standing promise is that the photo and the depth map stay on the device. A report
    // that grew a base64 blob would break it in a public issue tracker.
    expect(body).not.toMatch(/[A-Za-z0-9+/]{64,}={0,2}/);
    expect(body.length).toBeLessThan(1200);
  });
});

describe("buildIssueUrl", () => {
  const body = formatMeasurements(phone, compareToTraining(phone), context);
  const url = buildIssueUrl(body, "Depth capture result");

  it("targets the issue form the repository actually has", () => {
    expect(url.startsWith("https://github.com/simonkundrik/plate-vision/issues/new?")).toBe(true);
    expect(url).toContain("template=depth-capture-report.yml");
  });

  it("prefills the field the template declares", () => {
    const query = new URLSearchParams(url.split("?")[1]);
    expect(query.get(ISSUE_FIELDS.measurements)).toBe(body);
    expect(query.get("title")).toBe("Depth capture result");
    expect(query.get("labels")).toBe("depth-capture");
  });

  it("stays well inside what a URL bar will carry", () => {
    expect(url.length).toBeLessThan(4000);
  });
});
