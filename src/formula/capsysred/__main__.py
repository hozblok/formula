"""CLI: python3 -m formula.capsysred [config.yaml] -o out/ [--stages 4,5] [--quick N]
[--trace] [--replay rays.jsonl.gz] [--force]"""

import argparse
import sys

from .simulation import KNOWN_STAGES, Simulation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m formula.capsysred",
        description="Source → (Lloyd mirror wall | capillaries) → screen → |μ| and I: "
                    "images, rays.jsonl.gz and a timestamped report-*.md into the output directory.")
    parser.add_argument("config", nargs="?", default=None,
                        help="YAML with parameters (built-in defaults when omitted)")
    parser.add_argument("-o", "--out", default="capsysred-out",
                        help="output directory (default ./capsysred-out)")
    parser.add_argument("--stages", default=None,
                        help="which stages to run, e.g. 1,2,3 (default all; "
                             "3 requires 2, 5 requires 4 — added automatically)")
    parser.add_argument("--quick", type=int, default=1, metavar="N",
                        help="divisor for the mode/ray counts for a quick estimate")
    parser.add_argument("--trace", action="store_true",
                        help="trace-only: record every scene into the rays file and "
                             "exit; a later run with the same config, output "
                             "directory and --quick reuses it instead of tracing")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing an existing incompatible, incomplete, "
                             "or unreadable rays.jsonl.gz (without this flag it is "
                             "never overwritten)")
    parser.add_argument("--replay", metavar="RAYS_JSONL", default=None,
                        nargs="+",
                        help="re-evaluate recorded rays on the spectrum/material from "
                             "the config, without tracing; several files stream as one "
                             "recording (mode ids offset, config n_modes = their sum)")
    parser.add_argument("--no-jackknife", action="store_true",
                        help="stage 10 totals-only: |mu| map without per-mode rows "
                             "(no sigma_jack/loo; O(pixels) memory for huge grids)")
    args = parser.parse_args(argv)

    if args.force and args.replay:
        parser.error("--force cannot be used with --replay "
                     "(replay never writes the rays file)")

    stages = None
    if args.stages:
        stages = sorted({int(s) for s in args.stages.replace(" ", "").split(",")})
        bad = [s for s in stages if s not in KNOWN_STAGES]
        if bad:
            parser.error(f"no such stages: {bad}; available {list(KNOWN_STAGES)}")

    sim = (Simulation.from_yaml(args.config) if args.config
           else Simulation.from_dict({}))
    sim.no_jackknife = args.no_jackknife   # read by run_jack_stage
    result = (sim.replay(args.replay, args.out, stages=stages,
                         quick=max(1, args.quick)) if args.replay
              else sim.trace(args.out, quick=max(1, args.quick),
                             force=args.force) if args.trace
              else sim.run(args.out, stages=stages, quick=max(1, args.quick),
                           force=args.force))
    print(f"{result['out_dir']}: " + ", ".join(result["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
