"""Compare two prediction caches produced by old vs new indexing paths.

Entries are keyed canonically but stored under two key types across
generations (plain ``Path`` before the ResolvedPath refactor, wrapped
afterwards); both are normalized to their string form here so the script can
compare cache files of any generation.
"""

import os
import pickle
import sys
from pathlib import Path

old_path, new_path = (Path(p) for p in sys.argv[1:3])
with old_path.open("rb") as f:
    old = pickle.load(f)
with new_path.open("rb") as f:
    new = pickle.load(f)

old_entries = {str(key): value for key, value in old["entries"].items()}
new_entries = {str(key): value for key, value in new["entries"].items()}
assert old["model_version"] == new["model_version"]

assert set(old_entries) == set(new_entries), (
    f"key mismatch\n only old: {sorted(set(old_entries) - set(new_entries))}\n"
    f" only new: {sorted(set(new_entries) - set(old_entries))}"
)

max_diff, worst = 0.0, ""
for key in sorted(old_entries):
    o, n = old_entries[key], new_entries[key]
    assert o.size_bytes == n.size_bytes, f"size drift: {key}"
    assert o.modified_ns == n.modified_ns, f"mtime drift: {key}"
    assert set(o.prediction.mood_scores) == set(n.prediction.mood_scores), f"mood key mismatch: {key}"
    assert o.prediction.valence == n.prediction.valence, f"valence diff: {key}"
    assert o.prediction.arousal == n.prediction.arousal, f"arousal diff: {key}"
    for mood in o.prediction.mood_scores:
        diff = abs(o.prediction.mood_scores[mood] - n.prediction.mood_scores[mood])
        if diff > max_diff:
            max_diff, worst = diff, f"{os.path.basename(key)}:{mood}"

print(f"entries: {len(old_entries)}")
print(f"max mood-score diff: {max_diff:.3e} ({worst})")
print("IDENTICAL" if max_diff == 0.0 else ("CLOSE (fp-level)" if max_diff < 1e-5 else "MISMATCH"))
