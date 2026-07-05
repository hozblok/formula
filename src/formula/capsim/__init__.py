"""Capillary X-ray coherence simulator on the Number/Solver engine (no numpy).

One command runs the whole project (see plan-cap.ru.md): setup scheme, degree
of coherence with no optics (MC + van Cittert-Zernike analytics), the Lloyd
single-wall experiment (MC + two-path analytics, wall = capillary surface in
the same tracer), and the capillary run — images, rays.jsonl, report.md:

    python3 -m formula.capsim config.yaml -o out/
"""

from .config import Config, load
from .simulation import ALL_STAGES, Simulation

__all__ = ["ALL_STAGES", "Config", "Simulation", "load"]
