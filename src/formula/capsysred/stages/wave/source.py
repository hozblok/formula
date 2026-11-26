"""Source of Stage 16: its positive quadrature or the recorded origins of the rays archives."""

import math

import numpy as np

from ... import rays_v3


def source_rule(src, wave):
    """Positive quadrature of the source: (nodes (N, 2) [m], weights (N,), label)."""
    shape, size = src.shape, float(src.size)
    cx, cy = float(src.position[0]), float(src.position[1])
    n_req = int(wave["source_nodes"])
    if shape == "grid":                     # before the degenerate-size branch: size 0 = equal weights
        n, s = src.grid_n, float(src.grid_step)
        half = (n - 1) / 2.0
        rot = math.radians(getattr(src, "grid_rot_deg", 0.0))
        c, q = math.cos(rot), math.sin(rot)
        sites = np.array([((i - half) * s * c - (j - half) * s * q,
                           (i - half) * s * q + (j - half) * s * c)
                          for i in range(n) for j in range(n)])
        r_max = getattr(src, "grid_r_max", None)
        if r_max is not None:
            sites = sites[np.hypot(sites[:, 0], sites[:, 1]) <= r_max]
        w = (np.exp(-(sites ** 2).sum(1) / (2.0 * size * size)) if size > 0.0
             else np.ones(len(sites)))
        return _checked(sites + np.array([cx, cy]), w, f"grid: {len(sites)} lattice nodes")
    if shape == "point" or size <= 0.0:
        return np.array([[cx, cy]]), np.array([1.0]), "point"
    if shape == "disk":
        n_r = max(1, math.ceil(math.sqrt(n_req / 4.0)))
        n_phi = 4 * n_r
        t, wt = np.polynomial.legendre.leggauss(n_r)      # uniform in r^2
        rad = size * np.sqrt((t + 1.0) / 2.0)
        phi = (np.arange(n_phi) + 0.5) * 2.0 * math.pi / n_phi
        nodes = np.stack([cx + rad[:, None] * np.cos(phi)[None, :],
                          cy + rad[:, None] * np.sin(phi)[None, :]], -1).reshape(-1, 2)
        w = np.repeat(wt / (2.0 * n_phi), n_phi)
        return _checked(nodes, w, f"disk: Gauss-Legendre(r^2) {n_r} x {n_phi} angles")
    if shape == "gaussian":
        n = max(1, math.ceil(math.sqrt(n_req)))
        t, wt = np.polynomial.hermite_e.hermegauss(n)
        gx, gy = np.meshgrid(t, t, indexing="ij")
        nodes = np.stack([cx + size * gx, cy + size * gy], -1).reshape(-1, 2)
        w = (wt[:, None] * wt[None, :]).reshape(-1)
        return _checked(nodes, w, f"gaussian: Gauss-Hermite {n} x {n}")
    raise ValueError(f"wave_estimator: unsupported source shape {shape!r}")


def _checked(nodes, w, label):
    w = np.asarray(w, dtype=float)
    if len(nodes) == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0) or w.sum() <= 0.0:
        raise ValueError(f"wave_estimator: source rule {label!r} has no nodes or invalid weights")
    return np.asarray(nodes, dtype=float), w / w.sum(), label


def recorded_rule(archives, scene, src):
    """Empirical source of rays v3 archives: every recorded origin of every archive, weight 1/N."""
    nodes = []
    for archive in archives:
        if not rays_v3.is_v3(archive):
            raise ValueError(f"{archive}: recorded_origins needs a v3 rays archive")
        index = rays_v3.load_index(archive)
        origins = rays_v3.origins(archive, index, scene)
        if not origins or any(o is None for o in origins):
            raise ValueError(f"{archive}: scene {scene!r} has no recorded origins")
        zs = {round(float(o[2]), 12) for o in origins}
        if zs != {round(float(src.position[2]), 12)}:
            raise ValueError(f"{archive}: origin z {sorted(zs)} differs from the "
                             f"source z {float(src.position[2])}")
        nodes += [[float(o[0]), float(o[1])] for o in origins]
    nodes = np.array(nodes)
    return nodes, np.full(len(nodes), 1.0 / len(nodes)), \
        f"recorded_origins: {len(nodes)} modes from {len(archives)} archive(s)"


def _rule(wave, scene, src, rays_paths):
    if wave["source_mode"] == "recorded_origins":
        if not rays_paths:
            raise ValueError("wave_estimator: source_mode recorded_origins needs a rays archive "
                             "(out/rays-modes or --replay)")
        return recorded_rule(list(rays_paths), scene, src)
    return source_rule(src, wave)
