"""Single-line console progress with rate and ETA (stderr)."""

import sys
import time

from .shared.format import hms as _hms


class Progress:
    def __init__(self, label: str, total: int, stream=sys.stderr):
        self.label, self.total, self.stream = label, max(total, 1), stream
        self.done = 0
        self.t0 = time.time()
        self._last = 0.0
        self._render(force=True)

    def step(self, n: int = 1, extra: str = ""):
        self.done += n
        self._render(extra=extra)

    def _render(self, extra: str = "", force: bool = False):
        now = time.time()
        if not force and now - self._last < 0.2 and self.done < self.total:
            return
        self._last = now
        pct = 100.0 * self.done / self.total
        elapsed = now - self.t0
        eta = elapsed / self.done * (self.total - self.done) if self.done else 0.0
        line = (f"\r  {self.label:<22} {self.done:>9,}/{self.total:<9,} "
                f"{pct:5.1f}%  elapsed {_hms(elapsed)}  left ~{_hms(eta)}  {extra}")
        self.stream.write(line.ljust(100))
        self.stream.flush()

    def finish(self, extra: str = ""):
        elapsed = time.time() - self.t0
        self.stream.write(
            f"\r  {self.label:<22} {self.done:>9,} rays in {_hms(elapsed)}  {extra}".ljust(100) + "\n")
        self.stream.flush()
