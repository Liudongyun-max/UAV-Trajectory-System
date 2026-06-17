from __future__ import annotations

import re
from pathlib import Path

from .config import ROOT
from .layout import RUN_SCAN_DIRS

RUN_RE = re.compile(r"^run(\d{3})$")


def existing_run_indices(root: Path = ROOT) -> list[int]:
    indices: set[int] = set()
    for rel in RUN_SCAN_DIRS:
        base = root / rel
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                match = RUN_RE.fullmatch(child.name)
                if match:
                    indices.add(int(match.group(1)))
    return sorted(indices)


def resolve_run_id(requested: str, root: Path = ROOT) -> str:
    if requested == "auto":
        indices = existing_run_indices(root)
        return f"run{(max(indices) + 1 if indices else 1):03d}"
    if not RUN_RE.fullmatch(requested):
        raise ValueError(f"RUN_ID must be auto or runXXX: {requested}")
    return requested


def assert_can_create_run(run_id: str, resume: bool = False, root: Path = ROOT) -> None:
    exists = any((root / rel / run_id).exists() for rel in RUN_SCAN_DIRS)
    if exists and not resume:
        raise FileExistsError(f"{run_id} already exists; set RESUME=true to continue it")

