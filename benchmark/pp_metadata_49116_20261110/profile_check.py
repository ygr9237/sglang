import json
from pathlib import Path

p = Path(__file__).parent
rows = json.loads((p / "profiles.json").read_text())
selected = {r["arm"]: r for r in rows if r["case"] == "cuda_view32k_b25"}
assert set(selected) == {"base", "fixed"}
out = {}
for arm, row in selected.items():
    events = row["events"]
    clone_count = sum(e["count"] for e in events if e["key"] == "aten::clone")
    d2h_count = sum(e["count"] for e in events if "DtoH" in e["key"])
    assert clone_count == (50 if arm == "base" else 0), (arm, clone_count)
    assert d2h_count == 50, (arm, d2h_count)
    out[arm] = {"cpu_clone_count": clone_count, "d2h_count": d2h_count,
                "pickle_bytes": row["pickle_bytes"]}
assert out["base"]["pickle_bytes"] == out["fixed"]["pickle_bytes"]
(p / "profile-check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
