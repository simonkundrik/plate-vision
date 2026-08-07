#!/usr/bin/env python
"""Push a notebook to Kaggle, run it headless on a GPU, and fetch the output.

There is no NVIDIA GPU on this machine, so training runs on Kaggle's free tier. Driving it
through the API rather than the web UI means the run is reproducible from the repository
and does not depend on remembering which toggles were set in a browser.

Authenticate once, interactively, before using this:

    kaggle auth login

Usage:
    python scripts/run_on_kaggle.py --slug plate-vision-baseline
    python scripts/run_on_kaggle.py --slug plate-vision-baseline --watch
    python scripts/run_on_kaggle.py --slug plate-vision-baseline --fetch-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from platevision import kaggle_run

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = MODEL_DIR / "notebooks" / "01_food101_baseline.ipynb"
POLL_SECONDS = 60


def kaggle_cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Invoke the Kaggle CLI as a subprocess.

    Not `import kaggle`: that package authenticates at import time and raises without
    credentials, which would make this module unimportable in CI.
    """
    command = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(command, capture_output=True, text=True, check=check)


def resolve_username(explicit: str | None) -> str:
    if explicit:
        return explicit
    if env := os.environ.get("KAGGLE_USERNAME"):
        return env

    token_file = Path.home() / ".kaggle" / "kaggle.json"
    if token_file.is_file():
        try:
            return json.loads(token_file.read_text(encoding="utf-8"))["username"]
        except (json.JSONDecodeError, KeyError):
            pass

    raise SystemExit(
        "Could not determine your Kaggle username. Pass --user, or set KAGGLE_USERNAME.\n"
        "If you have not authenticated yet, run:  kaggle auth login"
    )


def stage(notebook: Path, metadata: dict, staging: Path) -> None:
    """Assemble the folder `kaggle kernels push` expects: notebook plus metadata."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(notebook, staging / notebook.name)
    kaggle_run.write_kernel_metadata(staging, metadata)


def watch(kernel_id: str, poll_seconds: int) -> str:
    print(
        f"Watching {kernel_id} (polling every {poll_seconds}s). Ctrl-C is safe, "
        f"the run continues on Kaggle."
    )
    while True:
        result = kaggle_cli("kernels", "status", kernel_id, check=False)
        status = kaggle_run.parse_status(result.stdout + result.stderr)
        stamp = time.strftime("%H:%M:%S")
        print(f"  [{stamp}] {status}")
        if kaggle_run.is_terminal(status):
            return status
        time.sleep(poll_seconds)


def fetch(kernel_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching output into {out_dir}")
    result = kaggle_cli("kernels", "output", kernel_id, "-p", str(out_dir), check=False)
    print(result.stdout.strip() or result.stderr.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--slug", required=True, help="kernel slug, lowercase with hyphens")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--user", help="Kaggle username; auto-detected when possible")
    parser.add_argument("--title")
    parser.add_argument(
        "--accelerator",
        default=kaggle_run.DEFAULT_ACCELERATOR,
        choices=kaggle_run.ACCELERATORS,
    )
    parser.add_argument("--public", action="store_true", help="publish the kernel publicly")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--watch", action="store_true", help="poll until the run finishes")
    parser.add_argument("--fetch-only", action="store_true", help="skip the push, fetch output")
    parser.add_argument("--out", type=Path, default=MODEL_DIR / "runs" / "kaggle")
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    args = parser.parse_args(argv)

    username = resolve_username(args.user)
    kernel_id = f"{username}/{args.slug}"

    if args.fetch_only:
        fetch(kernel_id, args.out)
        return 0

    if not args.notebook.is_file():
        raise SystemExit(f"notebook not found: {args.notebook}")

    metadata = kaggle_run.build_kernel_metadata(
        username=username,
        slug=args.slug,
        code_file=args.notebook.name,
        title=args.title,
        enable_gpu=not args.no_gpu,
        is_private=not args.public,
    )

    staging = MODEL_DIR / ".kaggle-staging" / args.slug
    stage(args.notebook, metadata, staging)
    print(f"Staged {args.notebook.name} for {kernel_id}")
    print(f"  accelerator: {args.accelerator}")
    print("  internet:    enabled (the notebook clones the repo and fetches Food-101)")

    push = kaggle_cli(
        "kernels", "push", "-p", str(staging), "--accelerator", args.accelerator, check=False
    )
    output = (push.stdout + push.stderr).strip()
    print(output)
    if push.returncode != 0:
        if "authentication" in output.lower():
            print("\nRun `kaggle auth login` first.", file=sys.stderr)
        return push.returncode

    print(f"\nRunning at https://www.kaggle.com/code/{kernel_id}")
    if not args.watch:
        print(f"Poll with:  python scripts/run_on_kaggle.py --slug {args.slug} --watch")
        return 0

    status = watch(kernel_id, args.poll_seconds)
    if status != "complete":
        print(f"run finished with status: {status}", file=sys.stderr)
        return 1

    fetch(kernel_id, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
