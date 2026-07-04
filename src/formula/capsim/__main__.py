"""CLI: python3 -m formula.capsim [config.yaml] -o out/ [--stages 4,5] [--quick N]
[--replay rays.jsonl]"""

import argparse
import sys

from .simulation import ALL_STAGES, Simulation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m formula.capsim",
        description="Source → (Lloyd mirror wall | capillaries) → screen → |μ| and I: "
                    "images, reflections.jsonl and report.md into the output directory.")
    parser.add_argument("config", nargs="?", default=None,
                        help="YAML with parameters (built-in defaults when omitted)")
    parser.add_argument("-o", "--out", default="capsim-out",
                        help="output directory (default ./capsim-out)")
    parser.add_argument("--stages", default=None,
                        help="which stages to run, e.g. 1,2,3 (default all; "
                             "3 requires 2, 5 requires 4 — added automatically)")
    parser.add_argument("--quick", type=int, default=1, metavar="N",
                        help="divisor for the mode/ray counts for a quick estimate")
    parser.add_argument("--replay", metavar="RAYS_JSONL", default=None,
                        help="re-evaluate recorded rays on the spectrum/material from "
                             "the config, without tracing")
    args = parser.parse_args(argv)

    stages = None
    if args.stages:
        stages = sorted({int(s) for s in args.stages.replace(" ", "").split(",")})
        bad = [s for s in stages if s not in ALL_STAGES]
        if bad:
            parser.error(f"no such stages: {bad}; available {list(ALL_STAGES)}")

    sim = (Simulation.from_yaml(args.config) if args.config
           else Simulation.from_dict({}))
    result = (sim.replay(args.replay, args.out) if args.replay
              else sim.run(args.out, stages=stages, quick=max(1, args.quick)))
    print(f"{result['out_dir']}: " + ", ".join(result["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
