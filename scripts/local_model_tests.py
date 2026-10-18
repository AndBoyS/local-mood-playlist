import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

from const import REPO_DIR

CONTEXT = "local/model-tests"


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)


def sh(*cmd: str) -> str:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as error:
        die(f"could not run {cmd[0]}: {error}")
    if result.returncode != 0:
        die(f"command failed (exit {result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def main() -> None:
    repo = sh("gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    host = sh("hostname", "-s")

    sha = sh("git", "rev-parse", "HEAD")
    sh("git", "fetch", "--quiet")
    if not sh("git", "branch", "-r", "--contains", sha):
        die(f"HEAD ({sha}) is not on any remote-tracking branch; push first (or fetch if pushed elsewhere).")

    fd, log_name = tempfile.mkstemp(prefix="model-tests-", suffix=".log")
    os.close(fd)
    log = Path(log_name)

    pytest_proc: subprocess.Popen[str] | None
    code: int | None = None
    log_written = False
    try:
        pytest_proc = subprocess.Popen(
            ["uv", "run", "pytest", "-m", "model", *sys.argv[1:]],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except FileNotFoundError:
        # Shell semantics don't apply to list-form Popen: a missing `uv` binary
        # raises ENOENT instead of yielding the shell's exit 127.
        log.unlink()
        code = 127
    except OSError:
        log.unlink()
    else:
        stream = pytest_proc.stdout
        assert stream is not None, "stdout=PIPE was requested"
        with log.open("w", encoding="utf-8") as log_file:
            try:
                for line in stream:
                    sys.stdout.write(line)
                    log_file.write(line)
            except BaseException:
                pytest_proc.kill()
                pytest_proc.wait()
                raise
        code = pytest_proc.wait()
        log_written = True

    if code == 0:
        state, reason = "success", f"model tests passed on {host}"
    elif code == 4:
        state, reason = "failure", f"pytest usage error on {host} (exit 4; check extra args)"
    elif code == 5:
        state, reason = "failure", f"no model tests collected on {host} (mark tests with pytest.mark.model)"
    elif code == 127:
        state, reason = "failure", f"uv not found on {host}"
    elif code is None:
        state, reason = "failure", f"model tests failed on {host} (could not start pytest)"
    else:
        state, reason = "failure", f"model tests failed on {host} (exit {code})"
    # Log filename (not full path) keeps the description within GitHub's 140-char
    # commit-status limit; the absolute path is printed locally below.
    log_suffix = f" — log: {log.name}" if log_written else ""
    description = f"{reason}{log_suffix}"[:140]

    head_moved = sh("git", "rev-parse", "HEAD") != sha
    if head_moved:
        die(f"HEAD moved while tests ran ({sha}); not posting a stale status. Re-run.")

    sh(
        "gh",
        "api",
        f"repos/{repo}/statuses/{sha}",
        "-f",
        f"state={state}",
        "-f",
        f"context={CONTEXT}",
        "-f",
        f"description={description}",
    )
    print(f"reported {state} on {sha}: {description}")
    if log_written:
        print(f"output: {log}")


if __name__ == "__main__":
    main()
