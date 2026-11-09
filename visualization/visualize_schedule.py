"""Create a standalone placement viewer (no server or extra packages needed).

    python3 visualization/visualize_schedule.py
    python3 visualization/visualize_schedule.py outputs/schedule_synthetic_four_prefixes_mean.json
    python3 visualization/visualize_schedule.py outputs/schedule.json --trace trace.jsonl --output view.html

With no paths, includes all schedule_*.json files in the repository's outputs/.
The default HTML output is visualization/placements.html.
Synthetic prefix_group metadata is joined by request id from the matching trace.
Other schedules show chosen reuse families, not inferred prefix equivalence.
"""

import argparse
import json
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parent


def prepare(path, trace_path=None):
    data = json.loads(path.read_text())
    if not isinstance(data.get("requests"), list) or not data["requests"]:
        raise ValueError(f"{path}: expected a nonempty requests list")
    data["name"] = path.stem.removeprefix("schedule_")
    if trace_path is None:
        for case in ("same_prefix", "distinct_prefixes", "four_prefixes"):
            if path.stem.startswith(f"schedule_synthetic_{case}_"):
                candidate = REPO_ROOT / "traces" / "synthetic" / f"{case}.jsonl"
                if candidate.exists():
                    trace_path = candidate
    if trace_path:
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        for request in data["requests"]:
            i = request["id"]
            if not isinstance(i, int) or not 0 <= i < len(rows):
                raise ValueError(f"{trace_path}: no trace row for request {i}")
            row = rows[i]
            if row.get("input_length") != request["isl"] or row.get("output_length") != request["osl"]:
                raise ValueError(f"{trace_path}: token lengths disagree for request {i}")
            if "prefix_group" in row:
                request["prefix_group"] = row["prefix_group"]
        data["trace_source"] = str(trace_path)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("schedules", type=Path, nargs="*")
    parser.add_argument("--trace", type=Path, help="matching Mooncake JSONL; only with one schedule")
    parser.add_argument("--output", type=Path, default=VIEWER_DIR / "placements.html")
    args = parser.parse_args()
    paths = args.schedules or sorted((REPO_ROOT / "outputs").glob("schedule_*.json"))
    if not paths:
        parser.error(f"No schedule_*.json files found in {REPO_ROOT / 'outputs'}")
    if args.trace and len(paths) != 1:
        parser.error("--trace requires exactly one schedule")
    datasets = [prepare(path, args.trace) for path in paths]
    # Prefer the controlled four-prefix example as the landing view.
    datasets.sort(key=lambda d: (d["name"] != "synthetic_four_prefixes_mean", d["name"]))
    payload = json.dumps(datasets).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("placement_viewer.html").read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__SCHEDULE_DATA__", payload))
    print(f"Wrote {args.output.resolve()} with {len(datasets)} schedules. Open it in your browser.")


if __name__ == "__main__":
    main()
