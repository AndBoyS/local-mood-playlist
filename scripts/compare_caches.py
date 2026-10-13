"""Compare two prediction caches produced by old vs new indexing paths."""

import pickle
import sys
from pathlib import Path

old_path, new_path = (Path(p) for p in sys.argv[1:3])
with old_path.open("rb") as f:
    old = pickle.load(f)
with new_path.open("rb") as f:
    new = pickle.load(f)

old_entries, new_entries = old["entries"], new["entries"]
assert old["model_version"] == new["model_version"]

assert set(old_entries) == set(new_entries), (
    f"key mismatch\n only old: {sorted(set(old_entries) - set(new_entries))}\n"
    f" only new: {sorted(set(new_entries) - set(old_entries))}"
)

max_diff, worst = 0.0, ""
for path in sorted(old_entries):
    o, n = old_entries[path], new_entries[path]
    assert o.size_bytes == n.size_bytes, f"size drift: {path}"
    assert o.modified_ns == n.modified_ns, f"mtime drift: {path}"
    assert set(o.prediction.mood_scores) == set(n.prediction.mood_scores), f"mood key mismatch: {path}"
    assert o.prediction.valence == n.prediction.valence, f"valence diff: {path}"
    assert o.prediction.arousal == n.prediction.arousal, f"arousal diff: {path}"
    for mood in o.prediction.mood_scores:
        diff = abs(o.prediction.mood_scores[mood] - n.prediction.mood_scores[mood])
        if diff > max_diff:
            max_diff, worst = diff, f"{path.name}:{mood}"

print(f"entries: {len(old_entries)}")
print(f"max mood-score diff: {max_diff:.3e} ({worst})")
print("IDENTICAL" if max_diff == 0.0 else ("CLOSE (fp-level)" if max_diff < 1e-5 else "MISMATCH"))
