#!/usr/bin/env python3
"""Atomically accept or reject both rows of one recorded recovery pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SAFE_REASON = re.compile(r"[a-z0-9][a-z0-9_.-]{2,80}")


def append_note(existing: str, note: str) -> str:
    parts = [part for part in existing.split("; ") if part and not part.startswith("pair_qc=")]
    parts.append(note)
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--accept", action="store_true")
    action.add_argument("--reject", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests/vla_recovery_episode_schedule.csv",
    )
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("refuse manifest edit without --confirm")
    if args.accept and args.reason:
        raise SystemExit("--reason is only valid with --reject")
    if args.reject and (not args.reason or not SAFE_REASON.fullmatch(args.reason)):
        raise SystemExit("--reject requires a 3-81 character lowercase machine-readable --reason")

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise SystemExit("manifest has no header")
        fieldnames = reader.fieldnames
        rows = list(reader)
    pair = [row for row in rows if row["pair_id"] == args.pair_id]
    if len(pair) != 2 or {row["order_type"] for row in pair} != {
        "pink_then_cyan",
        "cyan_then_pink",
    }:
        raise SystemExit(f"expected one complete opposite-order pair for {args.pair_id}")

    statuses = {row["status"] for row in pair}
    if args.accept:
        if statuses != {"recorded_pending_qc"}:
            raise SystemExit(f"accept requires both rows recorded_pending_qc, got {statuses}")
        for row in pair:
            row["status"] = "accepted"
            row["accepted"] = "true"
            row["initial_both_visible"] = "true"
            row["initial_pose_match"] = "true"
            row["smooth_motion_qc"] = "true"
            row["notes"] = append_note(row.get("notes", ""), "pair_qc=accepted")
        result = "accepted"
    else:
        if not statuses <= {"recorded_pending_qc", "accepted", "rejected_needs_retry"}:
            raise SystemExit(f"reject requires two already recorded rows, got {statuses}")
        for row in pair:
            row["status"] = "rejected_needs_retry"
            row["accepted"] = "false"
            row["initial_both_visible"] = "false"
            row["initial_pose_match"] = "false"
            row["smooth_motion_qc"] = "false"
            row["notes"] = append_note(
                row.get("notes", ""), f"pair_qc=rejected:{args.reason}"
            )
        result = f"rejected:{args.reason}"

    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=args.manifest.parent, delete=False
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(stream.name)
    temporary.replace(args.manifest)
    print(f"pair_id={args.pair_id} rows=2 result={result}")
    print("raw episodes were preserved; no dataset, upload, training, camera, or robot was touched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
