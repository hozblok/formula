"""CLI: python3 -m formula.capsysred config.yaml -o out/ [--stages 1,14]
[--replay ARCHIVE...]. Stages only read rays: out/rays-modes, out/rays.jsonl.gz
or the --replay paths; recordings come from python -m formula.capsysred.trace_v3."""

import argparse
import sys

from .simulation import KNOWN_STAGES, Simulation


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m formula.capsysred",
        description="Source → capillaries → screen → |μ| and I: "
                    "images and a timestamped report-*.md into the output directory (rays are recorded "
                    "separately by python -m formula.capsysred.trace_v3).")
    parser.add_argument("config", help="YAML with simulation parameters")
    parser.add_argument("-o", "--out", default="capsysred-out",
                        help="output directory (default ./capsysred-out)")
    parser.add_argument("--stages", default=None,
                        help="which stages to run, e.g. 1,2,3 (default core "
                             "stages of the configured scenes; 3 requires 2, "
                             "which is added automatically; 10 and 14 are "
                             "separate opt-in jackknife estimators)")
    parser.add_argument("--replay", metavar="ARCHIVE", default=None,
                        nargs="+",
                        help="rays recording(s) to re-evaluate with the spectrum and "
                             "material from the config (no tracing). ARCHIVE is a "
                             "v3 per-mode directory (trace_v3 / convert_rays_v3) or "
                             "a v2 rays.jsonl.gz; several archives are read as one "
                             "recording whose n_modes is their sum (Stage 14 keeps "
                             "one disk cache per archive). Without --replay the run "
                             "reads out/rays-modes or out/rays.jsonl.gz; it never "
                             "traces")
    args = parser.parse_args(argv)

    stages = None
    if args.stages:
        stages = sorted({int(s) for s in args.stages.replace(" ", "").split(",")})
        bad = [s for s in stages if s not in KNOWN_STAGES]
        if bad:
            parser.error(f"no such stages: {bad}; available {list(KNOWN_STAGES)}")

    sim = Simulation.from_yaml(args.config)
    result = (sim.replay(args.replay, args.out, stages=stages) if args.replay
              else sim.run(args.out, stages=stages))
    print(f"{result['out_dir']}: " + ", ".join(result["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
