"""Runtime provenance capture.

Rule: every provenance value is measured at the moment of use. Nothing here
is typed in by hand. If a value cannot be measured (for example, no container
image ID outside Modal), it is recorded as None rather than guessed.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata
from typing import Any

ANALYSIS_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn")


def package_versions(names: tuple[str, ...] = ANALYSIS_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(),
        # Set by Modal inside its containers; absent elsewhere, recorded as None.
        "container_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "container_task_id": os.environ.get("MODAL_TASK_ID"),
    }
