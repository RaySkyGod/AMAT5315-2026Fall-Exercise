"""Run `field ... | fluid ...` pipelines for the evidence scripts.

Binaries are resolved from PATH (after `cargo install --path .`), then from
the crate's target/ directories. Run from week4/scripts/.
"""

import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _binary(name: str) -> str:
    for cand in [shutil.which(name), BASE / "target/release" / name, BASE / "target/debug" / name]:
        if cand and Path(cand).exists():
            return str(cand)
    raise SystemExit(
        f"{name} not found: run `cargo install --path .` from week4/ (or build with cargo build)"
    )


def field(args, stdout_path):
    """Run `field` and write its JSON to stdout_path."""
    with open(stdout_path, "w") as fh:
        subprocess.run([_binary("field")] + args, stdout=fh, check=True)
    return stdout_path


def pipeline(field_args, fluid_args, tsv_path):
    """Run `field ... | fluid ...`; fluid's stdout goes to tsv_path.

    Returns fluid's exit code (1 means the run stopped at non-finite energy).
    """
    f = subprocess.run([_binary("field")] + field_args, capture_output=True, check=True)
    with open(tsv_path, "wb") as fh:
        r = subprocess.run(
            [_binary("fluid")] + fluid_args, input=f.stdout,
            stdout=fh, stderr=subprocess.PIPE,
        )
    if r.returncode not in (0, 1):
        raise SystemExit(f"fluid failed ({r.returncode}): {r.stderr.decode()}")
    return r.returncode


def read_frames(folder):
    """Parse <out>/fields.jsonl into a list of dicts."""
    import json

    with open(Path(folder) / "fields.jsonl") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_tsv(path):
    """Parse a fluid tsv: list of (t, E, Z); non-finite values stay NaN."""
    import math

    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts or not parts[0][0].isdigit() and parts[0][0] not in "-n":
                continue  # header
            try:
                t = float(parts[0])
            except ValueError:
                continue
            vals = [float(x) if not {"NaN", "nan", "inf", "-inf", "Infinity"} & {x} else math.nan for x in parts[1:]]
            if any(math.isnan(v) for v in vals) or "NaN" in line or "nan" in line or "inf" in line:
                rows.append((t, math.nan, math.nan))
            else:
                rows.append((t, *vals))
    return rows
