"""Dependency pin contract: wheel consumers get exactly the locked versions.

The wheel's Requires-Dist is the reproducibility surface for `uv tool install`
consumers, which resolve fresh from metadata (uv#5815: they do not read
uv.lock) — so every runtime dependency must have a `==` pin.
Verification is static and universal: the built
wheel's METADATA requirements must equal `uv export` of uv.lock (name + full
version specifiers; markers intentionally not compared — the two artifacts
normalize them differently). known blind spot: a loosened or removed pin
marker (e.g. dropping `sys_platform == 'linux'` from the CUDA pins) stays
green here — markers must be reviewed manually when editing pins.
Dev-group packages never enter either side.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import tomllib
import zipfile
from functools import partial
from pathlib import Path
from typing import NamedTuple

from const import REPO_DIR
from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.utils import NormalizedName
from packaging.utils import canonicalize_name as _canon_name_nonstrict

canonicalize_name = partial(_canon_name_nonstrict, validate=True)

GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class _Dep(NamedTuple):
    name: NormalizedName
    url: str | None
    specifiers: tuple[Specifier, ...]


def _parse_deps(raw_deps: list[str]) -> set[_Dep]:
    infos = set[_Dep]()
    for raw in raw_deps:
        req = Requirement(raw)
        infos.add(_Dep(name=canonicalize_name(req.name), url=req.url, specifiers=tuple(req.specifier)))
    return infos


def _declared_deps() -> set[_Dep]:
    with (REPO_DIR / "pyproject.toml").open("rb") as file:
        res: list[str] = tomllib.load(file)["project"]["dependencies"]
        return _parse_deps(res)


def _excluded_deps() -> set[_Dep]:
    with (REPO_DIR / "pyproject.toml").open("rb") as file:
        res: list[str] = tomllib.load(file)["tool"]["uv"]["exclude-dependencies"]
        return _parse_deps(res)


def _deps_from_wheel() -> set[_Dep]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        build = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out)],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        assert build.returncode == 0, f"uv build failed:\n{build.stderr}"
        (wheel,) = out.glob("*.whl")
        with zipfile.ZipFile(wheel) as zf:
            metadata_name = next(name for name in zf.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = zf.read(metadata_name).decode()
    return _parse_deps(re.findall(r"^Requires-Dist: (.+)$", metadata, re.MULTILINE))


def _deps_from_lock() -> set[_Dep]:
    result = subprocess.run(
        [
            "uv",
            "export",
            "--locked",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--no-annotate",
            "--no-header",
        ],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [
        stripped
        for line in result.stdout.splitlines()
        if len(stripped := line.strip()) > 0 and not stripped.startswith("#")
    ]
    return _parse_deps(lines)


def test_pins_are_exact() -> None:
    deps = _declared_deps()
    assert len(deps) > 0
    for dep in deps:
        if dep.url is not None:  # git dep
            ref = dep.url.rsplit("@", 1)[-1]
            assert dep.url.startswith("git+"), f"non-git URL dep: {dep.name}"
            assert GIT_SHA.fullmatch(ref) is not None, f"git dep not pinned by commit: {dep.name}"
        else:
            assert len(dep.specifiers) == 1, f"expected exactly one specifier: {dep.name}"
            spec = dep.specifiers[0]
            assert spec.operator == "==", f"unpinned dependency: {dep.name}"
            assert "*" not in spec.version, f"wildcard pin: {dep.name}"


def test_wheel_metadata_matches_lock() -> None:
    """Consumer-visible requirements (wheel) == locked runtime requirements (uv export)."""
    wheel = _deps_from_wheel()
    lock = _deps_from_lock()
    assert wheel == lock, (
        f"wheel-only requirements (stale pins): {sorted(wheel - lock)}; "
        f"lock-only requirements (consumer gets these unpinned): {sorted(lock - wheel)}"
    )


def test_no_pins_target_excluded_deps() -> None:
    excluded = _excluded_deps()
    pinned = _declared_deps()
    overlap = excluded & pinned
    assert len(overlap) == 0, f"pins target excluded deps: {sorted(overlap)}"


def test_uv_lock_is_fresh() -> None:
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"uv.lock is stale:\n{result.stderr}{result.stdout}"
