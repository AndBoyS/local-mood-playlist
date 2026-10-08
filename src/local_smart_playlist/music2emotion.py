"""Process adapter for vendored Music2Emotion inference."""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import override

from local_smart_playlist.predictions import TrackPrediction

MODEL_VERSION = "m2e-c5f7984-j95926289-btc11c6ed-tags-c53cee67-mert12af15fe-patch2"


class Music2EmotionWorkerError(RuntimeError):
    """Inference worker reported an error while processing audio."""


class Music2EmotionWorker(AbstractContextManager["Music2EmotionWorker"]):
    """Keep one Python 3.10 model process alive across a library indexing run."""

    def __init__(self, python_executable: Path | None = None) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        if python_executable is None:
            configured_python = os.environ.get("SP_MUSIC2EMOTION_PYTHON")
            if configured_python is not None:
                python_executable = Path(configured_python).expanduser()
            else:
                executable_name = "python.exe" if os.name == "nt" else "python"
                environment_dir = ".venv-m2e/Scripts" if os.name == "nt" else ".venv-m2e/bin"
                python_executable = repository_root / environment_dir / executable_name
        if not python_executable.is_file():
            raise FileNotFoundError(
                f"Music2Emotion environment missing: {python_executable}. "
                "Create it with `uv venv --python 3.10 .venv-m2e` and install "
                "`inference/requirements.txt`, or set SP_MUSIC2EMOTION_PYTHON."
            )

        worker_path = repository_root / "inference" / "music2emotion_worker.py"
        self._process: subprocess.Popen[str] = subprocess.Popen(
            [str(python_executable), "-u", str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("failed to open Music2Emotion worker pipes")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

    def __enter__(self) -> Music2EmotionWorker:
        return self

    @override
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def predict(self, audio_path: Path) -> TrackPrediction:
        if self._process.poll() is not None:
            raise RuntimeError(f"Music2Emotion worker exited with code {self._process.returncode}")
        request = json.dumps({"audio_path": str(audio_path.resolve(strict=True))})
        _ = self._stdin.write(f"{request}\n")
        self._stdin.flush()
        response_line = self._stdout.readline()
        if len(response_line) == 0:
            raise RuntimeError(
                f"Music2Emotion worker stopped without a response "
                f"(exit code {self._process.poll()})"
            )
        response: object = json.loads(response_line)
        if not isinstance(response, dict):
            raise TypeError("Music2Emotion worker returned invalid response")
        error = response.get("error")
        if isinstance(error, str):
            raise Music2EmotionWorkerError(f"Music2Emotion prediction failed: {error}")

        raw_scores = response.get("mood_scores")
        raw_valence = response.get("valence")
        raw_arousal = response.get("arousal")
        if not isinstance(raw_scores, dict):
            raise TypeError("Music2Emotion worker response has no mood score mapping")
        if (
            not isinstance(raw_valence, (int, float))
            or isinstance(raw_valence, bool)
            or not isinstance(raw_arousal, (int, float))
            or isinstance(raw_arousal, bool)
        ):
            raise TypeError("Music2Emotion worker response has invalid valence/arousal")
        scores: dict[str, float] = {}
        for tag, score in raw_scores.items():
            if not isinstance(tag, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
                raise TypeError("Music2Emotion worker response has invalid mood score")
            scores[tag] = float(score)
        return TrackPrediction(scores, float(raw_valence), float(raw_arousal))

    def close(self) -> None:
        if not self._stdin.closed:
            self._stdin.close()
        if self._process.poll() is None:
            try:
                _ = self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    _ = self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    _ = self._process.wait()
        if not self._stdout.closed:
            self._stdout.close()

