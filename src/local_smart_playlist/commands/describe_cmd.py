"""`sp describe` command."""

from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.table import Table
from shape_extensions import IntVar

from local_smart_playlist.config import default_db_path
from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder, cap_torch_threads, load_model
from local_smart_playlist.index.store import MetaKey, Store
from local_smart_playlist.query import prompts
from local_smart_playlist.query.phrases import track_documents
from local_smart_playlist.type_utils import NonEmptyTuple

DEFAULT_TOP_N = 10


def resolve_rel_path(raw: str | Path, root: Path, *, indexed_paths: set[str] | None = None) -> str:
    """Normalize *raw* to a POSIX rel path from the library root.

    Accepts an absolute path, including one reached through a symlink or an
    alternate root whose trailing path matches an indexed track, a cwd-relative
    path inside the root, or an already-indexed rel path. Falls back to the
    input as-is when no indexed suffix matches.
    """
    raw_path = Path(raw).expanduser()
    absolute = raw_path
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        if raw_path.is_absolute() and indexed_paths is not None:
            absolute_text = absolute.as_posix().rstrip("/")
            matches = [rel_path for rel_path in indexed_paths if absolute_text.endswith(f"/{rel_path}")]
            if len(matches) > 0:
                return max(matches, key=len)
        return raw_path.as_posix()


def rank_captions[M: IntVar](
    model: MuLanEmbedder[M], *, mean_vec: np.ndarray[[M]], vocab: NonEmptyTuple[str], top_n: int
) -> NonEmptyTuple[tuple[str, float]]:
    """Top captions by cosine similarity between the unit-norm track mean vector and vocab embeddings."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    embeds = model.embed_texts(vocab)
    docs = track_documents(mean_vec, embeds, top_n=top_n)
    return NonEmptyTuple((vocab[i], sim) for i, sim in docs)


class DescribeArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(*, path: str, db: Path | None, n: int) -> None:
        if n <= 0:
            raise SystemExit("number of captions must be positive")
        cap_torch_threads()
        db_path = db if db is not None else default_db_path()
        if not db_path.is_file():
            raise SystemExit(f"no index at {db_path}; run `sp index` first")

        with Store(db_path) as store:
            library_root = store.get_meta(MetaKey.LIBRARY_ROOT)
            store.require_model(MODEL_ID)
            if library_root is None:
                raise SystemExit("index has no library root recorded; re-run `sp index`")
            root = Path(library_root).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
            indexed_paths = store.track_rel_paths() if Path(path).expanduser().is_absolute() else None
            rel_path = resolve_rel_path(path, root, indexed_paths=indexed_paths)
            track = store.get_track(rel_path)
            if track is None:
                raise SystemExit(f"track not in index: {path}")
            vectors = track.vectors
            n_windows = track.n_windows
            if vectors is None or n_windows is None:
                raise RuntimeError(f"stored track has incomplete data: {rel_path}")
            model = load_model(device="cpu", text_only=True)
            captions = rank_captions(model, mean_vec=vectors.mean_vec, vocab=prompts.caption_vocab(), top_n=n)

        rprint(f"[bold]{track.title}[/bold]  {track.duration:.1f}s  {n_windows} windows  {rel_path}")
        table = Table(title="caption-vocab neighbors")
        table.add_column("sim", justify="right", style="cyan")
        table.add_column("caption")
        for caption, sim in captions:
            table.add_row(f"{sim:.3f}", caption)
        rprint(table)
