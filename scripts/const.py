from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

assert (REPO_DIR / "pyproject.toml").is_file(), (
    f"{REPO_DIR} is not a repo checkout (expected {REPO_DIR / 'pyproject.toml'}); "
    "REPO_DIR only exists for repo-local tooling and does not resolve under a non-editable install"
)
