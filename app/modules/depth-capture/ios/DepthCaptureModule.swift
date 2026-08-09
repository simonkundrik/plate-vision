import AVFoundation
import ExpoModulesCore

/// One-shot LiDAR depth capture.
///
/// **This compiles, and it has never run.** It was written on a Windows machine with no
/// Apple hardware attached, so every behavioural claim below is about what the AVFoundation
/// API is documented to do, not about what was observed. The `ios module` workflow builds it
/// for the simulator on every pull request, which rules out the syntax and the API surface
/// and rules out nothing else: a session that never delivers a depth map compiles perfectly.
/// That is why the feature ships behind a flag with an issue template attached.
///
/// A single photo capture is used rather than a streaming depth output. Streaming means
/// managing a session lifecycle across a screen that also renders a preview, and the thing
/// wanted here is one frame with its depth map, which `AVCapturePhotoOutput` delivers
/// directly.
///
/// Two settings below are deliberate and easy to get backwards:
///
/// - `isDepthDataFiltered = false`. Filtering interpolates over the holes where the sensor
///   returned nothing. Nutrition5k encodes those holes as zero and `platevision/depth.py`
///   fills them with the median of the valid pixels on purpose, so filtering here would hide
///   the one property most worth measuring: how much of a real plate a phone actually sees.
/// - Depth is converted to `kCVPixelFormatType_DepthFloat32`, which is metres, and then to
///   unsigned millimetres. Millimetres are what Nutrition5k stores and what ARCore reports,
///   so exactly one unit crosses into JavaScript.
public class DepthCaptureModule: Module {
  private let shot = DepthShot()

  public func definition() -> ModuleDefinition {
    Name("DepthCapture")

    Function("support") { () -> [String: Any] in
      DepthShot.support()
    }

    AsyncFunction("capture") { (promise: Promise) in
      self.shot.capture(promise: promise)
    }
  }
}

private final class DepthShot: NSObject, AVCapturePhotoCaptureDelegate {
  private let session = AVCaptureSession()
  private let output = AVCapturePhotoOutput()
  private let queue = DispatchQueue(label: "dev.simonkundrik.platevision.depth")

  private var pending: Promise?
  private var configured = false

  /// The LiDAR camera specifically, not whatever the back camera happens to be.
  ///
  /// `builtInDualCamera` can also produce depth, by disparity between two lenses, but that
  /// depth is relative rather than metric unless the device calibrates it. Asking for the
  /// LiDAR device means a `nil` here is an honest "this phone cannot do it" rather than a
  /// capture that succeeds and returns numbers in no particular unit.
  static func device() -> AVCaptureDevice? {
    AVCaptureDevice.default(.builtInLiDARDepthCamera, for: .video, position: .back)
  }

  static func support() -> [String: Any] {
    guard let device = device() else {
      return [
        "supported": false,
        "reason":
          "No LiDAR depth camera on this device. LiDAR is on iPhone 12 Pro and later Pro "
          + "models, and on iPad Pro from 2020.",
        "sensor": "",
      ]
    }
    return [
      "supported": true,
      "reason": "LiDAR depth camera available.",
      "sensor": device.localizedName,
    ]
  }

  func capture(promise: Promise) {
    queue.async {
      // One capture at a time. The delegate holds a single promise, and a second request
      // arriving mid-flight would either overwrite it, leaving the first caller waiting
      // forever, or resolve the wrong one.
      guard self.pending == nil else {
        promise.reject("E_BUSY", "A depth capture is already in progress.")
        return
      }

      do {
        try self.configure()
      } catch {
        promise.reject("E_CONFIGURE", error.localizedDescription)
        return
      }

      self.pending = promise

      if !self.session.isRunning {
        self.session.startRunning()
      }

      let settings = AVCapturePhotoSettings()
      settings.isDepthDataDeliveryEnabled = true
      settings.embedsDepthDataInPhoto = false
      settings.isDepthDataFiltered = false
      self.output.capturePhoto(with: settings, delegate: self)
    }
  }

  private func configure() throws {
    guard !configured else { return }

    guard let device = Self.device() else {
      throw DepthError.unsupported("No LiDAR depth camera on this device.")
    }

    session.beginConfiguration()
    defer { session.commitConfiguration() }

    session.sessionPreset = .photo

    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else {
      throw DepthError.unsupported("The camera could not be added to a capture session.")
    }
    session.addInput(input)

    guard session.canAddOutput(output) else {
      throw DepthError.unsupported("A photo output could not be added to the session.")
    }
    session.addOutput(output)

    // Order matters: this property only reads as supported once the output is attached to a
    // session that has the depth-capable device as its input.
    guard output.isDepthDataDeliverySupported else {
      throw DepthError.unsupported(
        "The camera is present but this session cannot deliver depth data.")
    }
    output.isDepthDataDeliveryEnabled = true

    configured = true
  }

  func photoOutput(
    _ output: AVCapturePhotoOutput,
    didFinishProcessingPhoto photo: AVCapturePhoto,
    error: Error?
  ) {
    guard let promise = pending else { return }
    pending = nil

    // The session is stopped either way. Leaving the LiDAR emitter running after a single
    // diagnostic capture is a battery cost with nothing to show for it.
    defer {
      queue.async { if self.session.isRunning { self.session.stopRunning() } }
    }

    if let error {
      promise.reject("E_CAPTURE", error.localizedDescription)
      return
    }

    guard let depth = photo.depthData else {
      promise.reject(
        "E_NO_DEPTH",
        "The photo arrived without a depth map. Depth delivery was requested and the device "
          + "reported it as supported, so this is worth reporting.")
      return
    }

    let converted = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32)
    guard let payload = Self.millimetres(from: converted.depthDataMap) else {
      promise.reject("E_READ", "The depth buffer could not be read.")
      return
    }

    promise.resolve([
      "width": payload.width,
      "height": payload.height,
      "base64": payload.data.base64EncodedString(),
      "filtered": converted.isDepthDataFiltered,
      // `relative` means the values carry no metric unit, which would make every millimetre
      // figure downstream meaningless. Reported rather than assumed.
      "accuracy": converted.depthDataAccuracy == .absolute ? "absolute" : "relative",
      "sensor": Self.device()?.localizedName ?? "",
    ])
  }

  private struct Payload {
    let width: Int
    let height: Int
    let data: Data
  }

  /// Float32 metres to unsigned millimetres, holes encoded as zero.
  ///
  /// The buffer is read row by row against `bytesPerRow` rather than as one flat array.
  /// CoreVideo pads rows to an alignment boundary, so treating the base address as
  /// `width * height` contiguous floats shears the image whenever the width is not aligned.
  private static func millimetres(from buffer: CVPixelBuffer) -> Payload? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    var values = [UInt16](repeating: 0, count: width * height)
    for y in 0..<height {
      let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
      for x in 0..<width {
        let metres = row[x]
        // A hole arrives as NaN with filtering off. Zero is what this project means by "the
        // sensor said nothing", everywhere, so it is left at its initialised zero.
        guard metres.isFinite, metres > 0 else { continue }
        values[y * width + x] = UInt16(min(metres * 1000, 65535))
      }
    }

    // Native byte order, which is little-endian on every device this can run on. The
    // JavaScript side decodes little-endian explicitly rather than inheriting whatever the
    // JavaScript engine's platform happens to be.
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    return Payload(width: width, height: height, data: data)
  }
}

private enum DepthError: Error, LocalizedError {
  case unsupported(String)

  var errorDescription: String? {
    switch self {
    case .unsupported(let message): return message
    }
  }
}
