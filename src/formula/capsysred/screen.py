"""Pixel grid on the screen plane z=const.

Binning is an integer decision, so it runs on floats; positions and phases stay
Number upstream. Pixel index = iy*nx + ix, row-major from the window corner.
"""


class ScreenGrid:
    def __init__(self, cfg):
        self.z = cfg.z
        self.nx, self.ny = cfg.nx, cfg.ny
        self.cxf, self.cyf = float(cfg.center[0]), float(cfg.center[1])
        self.exf, self.eyf = float(cfg.edge_x), float(cfg.edge_y)
        self.x0f, self.y0f = self.cxf - self.exf / 2, self.cyf - self.eyf / 2

    def pixel(self, point):
        """Pixel index for a screen point, None when outside the window."""
        fx = (float(point[0]) - self.x0f) / self.exf
        fy = (float(point[1]) - self.y0f) / self.eyf
        if not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        return int(fy * self.ny) * self.nx + int(fx * self.nx)

    def ref_pixel(self, reference):
        """Index of the reference point (window center when reference is None)."""
        xf, yf = reference if reference else (self.cxf, self.cyf)
        ix = min(self.nx - 1, max(0, int((xf - self.x0f) / self.exf * self.nx)))
        iy = min(self.ny - 1, max(0, int((yf - self.y0f) / self.eyf * self.ny)))
        return iy * self.nx + ix

    def xs(self):
        """Pixel-center x coordinates (floats, metres)."""
        return [self.x0f + (i + 0.5) * self.exf / self.nx for i in range(self.nx)]

    def ys(self):
        return [self.y0f + (j + 0.5) * self.eyf / self.ny for j in range(self.ny)]

    def pixel_xy(self, index):
        iy, ix = divmod(index, self.nx)
        return (self.x0f + (ix + 0.5) * self.exf / self.nx,
                self.y0f + (iy + 0.5) * self.eyf / self.ny)


class ScatterRaster:
    """Fixed-resolution ray-location raster over the screen window;
    square pixels, iy=0 at the bottom edge."""

    def __init__(self, cfg, px: int = 720):
        self.x0f = float(cfg.center[0]) - float(cfg.edge_x) / 2
        self.y0f = float(cfg.center[1]) - float(cfg.edge_y) / 2
        self.exf, self.eyf = float(cfg.edge_x), float(cfg.edge_y)
        step = max(self.exf, self.eyf) / px
        self.nx = max(1, round(self.exf / step))
        self.ny = max(1, round(self.eyf / step))
        self.counts = [[0] * self.nx for _ in range(self.ny)]

    def add(self, point):
        fx = (float(point[0]) - self.x0f) / self.exf
        fy = (float(point[1]) - self.y0f) / self.eyf
        if 0.0 <= fx < 1.0 and 0.0 <= fy < 1.0:
            self.counts[int(fy * self.ny)][int(fx * self.nx)] += 1

    def extent_um(self):
        return (1e6 * self.x0f, 1e6 * (self.x0f + self.exf),
                1e6 * self.y0f, 1e6 * (self.y0f + self.eyf))
