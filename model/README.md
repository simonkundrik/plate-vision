# model

PyTorch training, evaluation, and ONNX export.

## Layout

```
platevision/     package: contract, datasets, transforms, models, engine, checkpoints, export
data/            dataset acquisition scripts (downloaded data is gitignored)
scripts/         training entry points
notebooks/       Kaggle notebooks, kept thin: they orchestrate, they do not implement
tests/           unit tests, run in CI on every PR
```

Notebooks call into `platevision` rather than defining logic inline. Notebook code cannot be
linted usefully, cannot be unit tested, and invites editing the very thing you are trying to
measure, so anything worth trusting lives in the package.

## Setup

There is no NVIDIA GPU on the development machine, so install the CPU-only torch wheels
first. The default PyPI wheel pulls CUDA libraries that are several gigabytes of dead
weight here:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,train,export]"
```

The contract-only subset, if you just want to run the fast tests:

```bash
pip install -e ".[dev]"
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
