"""Human-readable length/duration labels and report file names."""

import os
import time

from .units import m_to_mm, m_to_um


def hms(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


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
