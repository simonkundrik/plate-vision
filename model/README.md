# model

PyTorch training, evaluation, and ONNX export.

## Layout

```
platevision/     package: contract loader, datasets, transforms, models, losses, training, export
data/            dataset acquisition scripts (downloaded data is gitignored)
notebooks/       Kaggle training notebooks
tests/           unit tests, run in CI on every PR
```

## Setup

Core install is deliberately light so contract and data tests run fast in CI:

```bash
pip install -e ".[dev]"
```

Training and export pull heavier wheels and are opt-in:

```bash
pip install -e ".[dev,train,export]"
```

## Where training runs

There is no local NVIDIA GPU, so training runs on Kaggle notebooks (30 GPU hours per week on the
free tier). `notebooks/` holds the runnable notebooks; `platevision/` holds the logic they import so
the notebooks stay thin and the real code stays testable.

Local machine handles data preparation, ONNX export, quantization, and CPU latency benchmarking. CPU
benchmarks are the relevant ones for the web demo anyway.

## The contract

`shared/model_meta.json` at the repo root defines input shape, output names, and preprocessing
constants for all three surfaces. Load it through `platevision.meta`, never by reading the file
directly and never by hardcoding the values.
