from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def open_atomic(filepath: Path) -> Generator[BinaryIO]:
    """Open a temporary file next to filepath; atomically replace filepath on clean exit.

    The destination keeps its old content if the body raises before exit. Data is
    fsynced before the rename so a completed write survives power loss; the rename
    swaps the name only after the data reached disk.
    Based on https://stackoverflow.com/a/29491523
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        descriptor, raw_name = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp",
        )
        temp_path = Path(raw_name)
        # os.fdopen cannot fail for a fresh descriptor opened by mkstemp.
        with os.fdopen(descriptor, "wb") as file:
            yield file
            file.flush()
            os.fsync(file.fileno())
        _ = temp_path.replace(filepath)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
