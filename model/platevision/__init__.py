"""Training, evaluation, and ONNX export for plate-vision.

The model contract (input shape, output names, preprocessing constants) lives in
``shared/model_meta.json`` and is loaded via :func:`platevision.meta.load_meta`.
Nothing in this package should hardcode those values.
"""

__version__ = "0.0.0"
