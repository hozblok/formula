"""Extended incoherent source: mutually incoherent coherent point modes.

Random draws are plain floats (not precision-critical); every emitted point and
direction is lifted to Number, and directions are unit-normalized in Number so
the absolute phase k*L stays exact downstream.
"""

import math
import random

from .nums import lift, vunit


class Source:
    def __init__(self, cfg, rng: random.Random):
        self.shape, self.size, self.position = cfg.shape, cfg.size, cfg.position
        self.n_modes, self.n_rays = cfg.n_modes, cfg.n_rays
        self.rng = rng
        self._p = self.size.precision
        self._size_f = float(self.size)

    def mode_origin(self):
        """One source point = one coherent mode."""
        if self.shape == "point" or self._size_f <= 0.0:
            return self.position
        if self.shape == "gaussian":
            ox = self.rng.gauss(0.0, self._size_f)
            oy = self.rng.gauss(0.0, self._size_f)
        else:  # disk
            r = self._size_f * math.sqrt(self.rng.random())
            phi = 2.0 * math.pi * self.rng.random()
            ox, oy = r * math.cos(phi), r * math.sin(phi)
        return (self.position[0] + lift(ox, self._p),
                self.position[1] + lift(oy, self._p),
                self.position[2])


def slope_direction(rng: random.Random, mx_range, my_range, precision: int):
    """Unit direction from slopes (mx, my, 1) uniform in a rectangular window.

    For micro-radian windows the solid-angle jacobian is constant to O(theta^2):
    the constant statistical weight cancels in mu and only scales intensity.
    """
    mx = rng.uniform(*mx_range)
    my = rng.uniform(*my_range)
    return vunit((lift(mx, precision), lift(my, precision), lift(1.0, precision)))


def aim_disk_direction(rng: random.Random, origin, cx: float, cy: float,
                       radius: float, z_target: float):
    """Unit direction from `origin` to a uniform point of a disk at z_target."""
    p = origin[0].precision
    r = radius * math.sqrt(rng.random())
    phi = 2.0 * math.pi * rng.random()
    target = (lift(cx + r * math.cos(phi), p), lift(cy + r * math.sin(phi), p),
              lift(z_target, p))
    return vunit((target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]))
