"""Receiver grids, coherence statistics and map output shared by estimators."""

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from ..screen import ScreenGrid


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _map_grid(screen, stride):
    grid = ScreenGrid(screen)
    hx, hy = grid.exf/grid.nx, grid.eyf/grid.ny
    if not math.isclose(hx, hy, rel_tol=1e-12, abs_tol=0):
        raise ValueError("stage17 archive_canonical currently requires square receiver cells")
    ref = grid.ref_pixel(screen.reference)
    ref_y, ref_x = divmod(ref, grid.nx)
    ix, iy = np.arange(ref_x % stride, grid.nx, stride), np.arange(ref_y % stride, grid.ny, stride)
    if len(ix) < 2 or len(iy) < 2:
        raise ValueError("stage17 map_stride leaves fewer than two cells on an axis")
    x = grid.x0f+(ix+.5)*hx
    y = grid.y0f+(iy+.5)*hy
    return dict(x=x, y=y, native_ix=ix, native_iy=iy, cell_width=hx,
                nx=grid.nx, ny=grid.ny, x0=grid.x0f, y0=grid.y0f,
                edge_x=grid.exf, edge_y=grid.eyf,
                ref_index=(int(np.flatnonzero(iy == ref_y)[0]), int(np.flatnonzero(ix == ref_x)[0])),
                reference=grid.pixel_xy(ref))


def matched_ray_field(points, phase, fresnel, grid):
    """Native stage14 binning on exactly the selected receiver cells."""
    shape = (len(grid["y"]), len(grid["x"]))
    field = np.zeros(shape, complex)
    self_i = np.zeros(shape)
    count = np.zeros(shape, np.int64)
    lookup_x, lookup_y = np.full(grid["nx"], -1), np.full(grid["ny"], -1)
    lookup_x[grid["native_ix"]] = np.arange(shape[1])
    lookup_y[grid["native_iy"]] = np.arange(shape[0])
    fx = (points[:, 0]-grid["x0"])/grid["edge_x"]
    fy = (points[:, 1]-grid["y0"])/grid["edge_y"]
    inside = (fx >= 0) & (fx < 1) & (fy >= 0) & (fy < 1)
    rays = np.flatnonzero(inside)
    ix = lookup_x[(fx[inside]*grid["nx"]).astype(int)]
    iy = lookup_y[(fy[inside]*grid["ny"]).astype(int)]
    selected = (ix >= 0) & (iy >= 0)
    rays, ix, iy = rays[selected], ix[selected], iy[selected]
    values = fresnel[rays]*np.exp(1j*phase[rays])
    np.add.at(field, (iy, ix), values)
    np.add.at(self_i, (iy, ix), abs(values)**2)
    np.add.at(count, (iy, ix), 1)
    intensity = abs(field)**2-self_i
    cross = field*field[grid["ref_index"]].conjugate()
    cross[grid["ref_index"]] -= self_i[grid["ref_index"]]
    return dict(I=intensity, W=cross, ray_count=count)


def normalized_coherence(intensity, cross, reference):
    mu = np.full(intensity.shape, np.nan+1j*np.nan)
    ref_i = intensity[reference]
    valid = np.isfinite(intensity) & np.isfinite(cross) & (intensity > 0)
    if np.isfinite(ref_i) and ref_i > 0:
        mu[valid] = cross[valid]/np.sqrt(intensity[valid]*ref_i)
    return mu


def _jackknife(rows_i, rows_w, reference):
    count = len(rows_i)
    shape = rows_i[0].shape
    if count < 3:
        return np.full(shape, np.nan), np.zeros(shape, np.int32)
    total_i, total_w = np.sum(rows_i, axis=0), np.sum(rows_w, axis=0)
    mean, m2, valid_count = np.zeros(shape), np.zeros(shape), np.zeros(shape, np.int32)
    for intensity, cross in zip(rows_i, rows_w):
        mu = abs(normalized_coherence(total_i-intensity, total_w-cross, reference))
        valid = np.isfinite(mu)
        valid_count[valid] += 1
        difference = mu[valid]-mean[valid]
        mean[valid] += difference/valid_count[valid]
        m2[valid] += difference*(mu[valid]-mean[valid])
    error = np.full(shape, np.nan)
    complete = valid_count == count
    error[complete] = np.sqrt((count-1)/count*np.maximum(m2[complete], 0))
    return error, valid_count


def _array_counts(intensity, mu):
    return {"pixels": int(intensity.size), "positive_intensity": int(np.sum(np.isfinite(intensity) & (intensity > 0))),
            "nonpositive_intensity": int(np.sum(np.isfinite(intensity) & (intensity <= 0))),
            "nonfinite_intensity": int(np.sum(~np.isfinite(intensity))),
            "finite_mu": int(np.sum(np.isfinite(mu))),
            "over_one_mu": int(np.sum(np.isfinite(mu) & (abs(mu) > 1+1e-10))),
            "max_abs_mu": float(np.nanmax(abs(mu))) if np.any(np.isfinite(mu)) else None}


def _atomic_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix+".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)
