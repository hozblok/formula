"""Float-array entrance-aperture membership and unique bore assignment."""

import math

import numpy as np

from .types import _APERTURE_REL_TOL


def is_polygon(bore):
    """Config dicts carry kind; raw YAML dicts may carry only sides."""
    return bore.get("kind") == "polygon" or bore.get("sides") is not None


def polygon_faces(bore):
    """Outward unit normals (n, 2), with face k at rotation + 2 pi k / n."""
    n = int(bore["sides"])
    rot = (float(bore["rotation"]) if bore.get("rotation") is not None
           else math.radians(float(bore.get("rotation_deg", 0.0))))
    phi = rot + 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(phi), np.sin(phi)])


def polygon_apothems(bore):
    """Per-face apothems (n,), defaulting to radius on every face."""
    aps = bore.get("apothems")
    if aps is None:
        return np.full(int(bore["sides"]), float(bore["radius"]))
    return np.array([float(a) for a in aps])


def bore_inside(points, bore, tol=_APERTURE_REL_TOL):
    """Transverse points (..., 2) inside a circular or polygon entrance."""
    points = np.asarray(points, dtype=float)
    center = np.asarray(bore["center"], dtype=float)
    radius = float(bore["radius"])
    if is_polygon(bore):
        aps = polygon_apothems(bore)
        return np.max((points - center) @ polygon_faces(bore).T - aps, axis=-1) <= tol * aps.max()
    return np.linalg.norm(points - center, axis=-1) <= radius * (1 + tol)


def entrance_bore_ids(points, bores, tol=_APERTURE_REL_TOL):
    """Unique containing-bore index, or -1 for outside/ambiguous points (..., 2)."""
    points = np.asarray(points, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("entrance points must have shape (..., 2)")
    count = np.zeros(points.shape[:-1], dtype=int)
    ids = np.full(points.shape[:-1], -1, dtype=int)
    for i, bore in enumerate(bores):
        inside = bore_inside(points, bore, tol)
        count += inside
        ids = np.where(inside, i, ids)
    return np.where(count == 1, ids, -1)
