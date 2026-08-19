"""Capillary X-ray coherence simulator on the Number/Solver engine (no numpy).

trace_v3 records rays; the stages consume the recording:

    python3 -m formula.capsysred.trace_v3 config.yaml --archive out/rays-modes
    python3 -m formula.capsysred config.yaml -o out/ --stages 1,14
"""

from .config import Config, load
from .simulation import KNOWN_STAGES, Simulation

__all__ = ["KNOWN_STAGES", "Config", "Simulation", "load"]
