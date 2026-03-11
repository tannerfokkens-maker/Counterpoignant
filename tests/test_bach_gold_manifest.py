from __future__ import annotations

import json
from pathlib import Path


def test_bach_gold_manifest_patterns_resolve_expected_counts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "data/benchmarks/bach_gold.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    assert manifest["groups"], "benchmark manifest should define at least one group"

    for group in manifest["groups"]:
        files = set()
        for pattern in group.get("patterns", []):
            files.update(repo_root.glob(pattern))
        for explicit in group.get("files", []):
            files.add(repo_root / explicit)
        assert len(files) == group["expected_count"], group["name"]
