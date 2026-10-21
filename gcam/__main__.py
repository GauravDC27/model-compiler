"""Command line for GCAM v1.

    python -m gcam run "Review expense reports from the last 7 days"
    python -m gcam run 'Handle request from C-1007: "Cancel order A-8391."' --approve cancel_order:A-8391
    python -m gcam train        # hub capacity (goal experts + LoRA), then System-1 fit and calibration
    python -m gcam eval         # per-op ladders vs baselines, end-to-end goals, replay check
"""

from __future__ import annotations

import argparse
import json


def _print(res) -> None:
    print(f"\ngoal     {res.text}")
    print(f"family   {res.family}    status {res.status}" + (f"    ({res.reason})" if res.reason else ""))
    if res.clarify is not None and res.clarify.candidates:
        print("options  " + "\n         ".join(res.clarify.candidates))
    if res.schedule_hash:
        print(f"schedule {res.schedule_hash[:23]}…   worst case {res.worst_case} tok-eq   realized {res.cost}"
              f"   {res.seconds:.1f}s")
    for n in res.nodes:
        path = " > ".join(t["rung"] + ("" if t["answered"] else " (abstained)") for t in n.tried) or "replayed from receipt"
        value = json.dumps(n.value, default=str)
        print(f"  {n.node:<10} {n.op:<24} {path}")
        print(f"  {'':<10} -> {value[:150]}{'…' if len(value) > 150 else ''}   [{n.cost} tok-eq]")
    for e in res.events:
        print(f"  event: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="gcam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("goal")
    run.add_argument("--approve", action="append", default=[])
    run.add_argument("--autonomy", default="confirm", choices=["read_only", "confirm", "auto"])
    sub.add_parser("train")
    sub.add_parser("eval")
    args = ap.parse_args()
    if args.cmd == "run":
        from .runtime import Runtime
        from .system import System

        rt = Runtime(System())
        _print(rt.run(args.goal, autonomy=args.autonomy, approvals=tuple(args.approve)))
    elif args.cmd == "train":
        from .train import calibrate, capacity

        capacity.main()
        calibrate.main()
    else:
        from . import evaluate

        evaluate.main()


if __name__ == "__main__":
    main()
