"""Human-readable length labels and report file names."""

import os
import time

from .units import m_to_mm, m_to_um


def mm(x) -> str:
    return f"{m_to_mm(x):g} mm"


def um(x) -> str:
    return f"{m_to_um(x):g} µm"


def report_name(out_dir: str, base: str) -> str:
    """Timestamped report name, never colliding with an existing file."""
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    name, n = f"{base}-{stamp}.md", 2
    while os.path.exists(os.path.join(out_dir, name)):
        name = f"{base}-{stamp}-{n}.md"
        n += 1
    return name
