"""Finding the shared contract, in a checkout and in an installed package.

``shared/model_meta.json`` sits above this package in the repository, which is fine while
working from a checkout and wrong the moment the package is installed: nothing above
``site-packages/platevision`` is a repository, so the file simply is not there and every
function that touches the contract raises FileNotFoundError.

The build copies the contract into ``platevision/_data/`` (see ``force-include`` in
pyproject.toml), so an installed package carries its own copy. A checkout has no such
directory and reads the repository file directly, which is what keeps the two from
drifting: there is only ever one contract in version control.

This is the same failure the TypeScript client had, and the same shape of fix. A published
package cannot reference files outside itself.
"""

from __future__ import annotations

from pathlib import Path

# platevision/_contract.py -> platevision -> model -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Written by the build, absent in a checkout.
PACKAGED_DIR = Path(__file__).resolve().parent / "_data"

# Present in a checkout, absent once installed.
REPO_SHARED_DIR = REPO_ROOT / "shared"


def contract_path(name: str) -> Path:
    """Locate a contract file, preferring the repository copy when one exists.

    The checkout is preferred deliberately. Editing ``shared/model_meta.json`` and having
    an editable install keep serving a stale build-time copy is exactly the confusion this
    project spends its time avoiding.

    Never raises. These paths are resolved at import time, and a package that cannot be
    imported at all is a worse failure than one whose loader reports a missing file. When
    neither copy exists the packaged location is returned, so the eventual error names the
    place an installed package should have carried the contract.
    """
    in_repo = REPO_SHARED_DIR / name
    return in_repo if in_repo.exists() else PACKAGED_DIR / name
