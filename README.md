# local-smart-playlist

Smart playlists for a local music library. MuQ-MuLan audio embeddings rank
tracks for mood, genre, instrument, vocal, rhythm, arrangement, timbre, and
recording queries. Describe what you want in plain words — a single word,
a comma list of attributes, or a free-text vibe — and `sp play` turns it
into a playlist; with `--llm`, any OpenAI-compatible endpoint aligns your
phrasing with the embedding model's caption style before ranking.

```console
uv sync
uv run sp index "/Users/user/Library/CloudStorage/Dropbox/Music/Selected"
uv run sp play "melancholic" -n 30
uv run sp status
```

- CLI binary: `sp`
- `sp index <root> [--rescan] [--quiet] [--db PATH]`
- `sp play "<query>" [-n 30] [-o out.m3u8|-] [--llm] [--context TEXT] [--seed-track PATH] [--dry] [--min-score 0.75]`
- `--context` adds supplementary context to the LLM adaptation message; used with `--llm`.
- `--min-score` is an upper-cluster fit cutoff (default 0.9; floor 0.5 neutral). With `--llm`, every decomposed query axis must clear it.
- `uv run python scripts/eval_rank.py` — production upper-cluster ranking probe over the live index (no playlist written)
- `sp status [--db PATH]`
- `sp describe PATH [-n 3] [--db PATH]` — top prompt matches within each attribute axis for an indexed track (ranked independently; offline; no LLM; `-n` sets matches per axis)

Playlists are written to `<library root>/[Playlists]/<query slug>.m3u8` with
paths relative to the playlist file, so they stay portable across synced
copies of the library (e.g. Dropbox → Android via FolderSync, consumed by
AIMP). Use `-o` for a custom destination, `-o -` to print the m3u8 to stdout.

Text queries rank by adaptive upper-cluster fit against axis reference prompts.
For each track, query and anchor similarities are averaged across stored windows;
the scorer separates the top anchor cluster from lower anchors and maps relative
fit to [0, 1]. A score above 0.5 favors the upper cluster. Without `--llm`,
the query uses mood anchors, matching the evaluated production recipe. With
`--llm`, each adapted query is decomposed into axis-labelled components and
scored against that axis's 32 prompts; every component must clear
`--min-score`, and weakest-axis fit determines rank. Axes cover mood,
genre/style, instruments/sources, vocals, tempo/rhythm, arrangement/dynamics,
timbre/production, and recording/context. Full-library scan uses stored
windows; tracks without them must be re-indexed.

`--seed-track` uses two-stage audio ranking:
exact NumPy track-mean cosine prefilter, then peak-window rescore. Seed score =
alpha * max cos(query, window) + (1-alpha) * cos(query, track mean);
`--alpha` tunes peak-mood vs whole-track mood (default 0.7).

Index DB defaults to `.sp-index/index.db` relative to the current working
directory; override with `--db`. After `sp play` closes its DB connection, it
starts a detached `VACUUM` worker when free pages exceed both 50 MiB and 25%
of DB size. Worker output goes to `<db path>.vacuum.log`; SQLite reuses free
pages when vacuum threshold is not met.

LLM query adaptation (`--llm`) uses any OpenAI-compatible endpoint. For free
text, the first LLM step generates distinct MuQ-MuLan-style attribute-query
variants using readout-caption examples. A second LLM step decomposes **each
variant** into axis-labelled components before ranking; CLI shows both stages.
Each variant ranks tracks separately and must pass every axis gate; results
are unioned by track using its highest weakest-axis score. Already-comma-separated
attribute lists pass through as one variant, then still receive axis decomposition.
Variants infer qualities implied by intent but avoid inventing specific
instruments, genres, or vocal styles. Add situation details with `--context`,
for example `sp play "something calm" --llm --context "for reading before bed"`.
Recognized components are
axis-gated; if results feel too narrow, lower `--min-score` toward 0.5 or
refine the original request. A failed LLM call shows and uses the raw phrase:

```
SP_LLM_BASE_URL  # default https://opencode.ai/zen/go/v1 (OpenCode Go)
SP_LLM_API_KEY   # default: $OPENCODE_API_KEY
SP_LLM_MODEL     # default deepseek-v4-flash
```

Model test suite (MuQ-MuLan download) is marked `model` and skipped by default:
`uv run pytest -m model` to run locally.
