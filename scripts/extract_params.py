"""Extract the parameter inventory from the pinned upstream config.h.

Parses the structured comment blocks that upstream itself uses to generate
docs/Parameters.rst (see upstream .ci/parameter-generator.py), so the inventory
is derived from source rather than memory.

Usage:
    uv run python scripts/extract_params.py

Writes docs/compat/params.json and an identical copy embedded by the Rust core
(crates/lgbm-core/data/params.json) for alias resolution and defaults.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_H = ROOT / "third_party" / "LightGBM" / "include" / "LightGBM" / "config.h"
OUT = ROOT / "docs" / "compat" / "params.json"
CORE_COPY = ROOT / "crates" / "lgbm-core" / "data" / "params.json"

DECL_RE = re.compile(r"^\s*([\w:<>, ]+?)\s+(\w+)\s*(?:=\s*(.*?))?;\s*$")
REGION_RE = re.compile(r"#pragma region (.+)$")


def upstream_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(CONFIG_H.parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()


def parse() -> list[dict]:
    lines = CONFIG_H.read_text(encoding="utf-8").splitlines()
    params: list[dict] = []
    region_stack: list[str] = []
    block: list[str] = []
    in_params = False
    for line in lines:
        stripped = line.strip()
        m = REGION_RE.search(stripped)
        if m:
            name = m.group(1).strip()
            region_stack.append(name)
            if name == "Parameters":
                in_params = True
            block = []
            continue
        if stripped.startswith("#pragma endregion"):
            if region_stack:
                ended = region_stack.pop()
                if ended == "Parameters":
                    in_params = False
            block = []
            continue
        if not in_params:
            continue
        if stripped.startswith("//"):
            block.append(stripped[2:].strip())
            continue
        if not stripped or stripped.startswith("#"):
            if not stripped:
                block = []
            continue
        m = DECL_RE.match(stripped)
        if m and block:
            ctype, name, default = m.group(1).strip(), m.group(2), m.group(3)
            entry = {
                "name": name,
                "section": region_stack[-1] if region_stack else "",
                "cpp_type": ctype,
                "cpp_default": default.strip() if default else None,
                "aliases": [],
                "checks": [],
                "options": [],
                "desc": [],
                "flags": [],
            }
            for c in block:
                if c.startswith("[") and c.endswith("]"):
                    entry["flags"].append(c.strip("[]"))
                    continue
                key, sep, val = c.partition("=")
                key, val = key.strip(), val.strip()
                if not sep:
                    entry["desc"].append(c)
                elif key == "alias":
                    entry["aliases"] += [a.strip() for a in val.split(",") if a.strip()]
                elif key == "check":
                    entry["checks"].append(val)
                elif key == "options":
                    entry["options"] += [o.strip() for o in val.split(",") if o.strip()]
                elif key == "default":
                    entry["doc_default"] = val
                elif key == "type":
                    entry["doc_type"] = val
                elif key in ("desc", "descl2"):
                    entry["desc"].append(val)
                else:
                    entry.setdefault("other", {})[key] = val
            entry["desc"] = " ".join(entry["desc"])
            params.append(entry)
        block = []
    return params


def main() -> None:
    params = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "upstream_tag": "v4.7.0",
        "upstream_commit": upstream_commit(),
        "source": "include/LightGBM/config.h",
        "num_parameters": len(params),
        "parameters": params,
    }
    text = json.dumps(payload, indent=2) + "\n"
    for out in (OUT, CORE_COPY):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {len(params)} parameters to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
