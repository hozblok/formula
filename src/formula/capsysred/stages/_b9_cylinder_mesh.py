"""Prescribed branch-fitted entrance triangles for a straight circular bore.

This controls the 0/1-reflection aperture boundary, not exit-plane caustics.
Every node is traced by the existing multiprecision tracer. No rejected face is
silently removed, and the finite geometric gaps are reported explicitly.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ..native import make_tracer
from ..shared.nums import lift, vsub, vunit
from ..surfaces import CapillaryBundle
from ..trace import trace_ray
from ._b9_archive import apply_tube_flux, exit_nodes, mesh_from_triangles
from .stage17 import _fresnel_product


def _integer(value, name, minimum):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _polygon_area(points):
    # Translate first, so a shifted bore does not lose its small area.
    p = points-points[0]
    return .5*abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))


def _entrance_mesh(source, bore, z0, z1, *, angles, inner_rings, outer_rings,
                   boundary_relative_gap, entrance_relative_inset):
    """Return explicit nodes, faces, family labels and measured entrance areas."""
    angles = _integer(angles, "angles", 8)
    inner_rings = _integer(inner_rings, "inner_rings", 1)
    outer_rings = _integer(outer_rings, "outer_rings", 1)
    if (not isinstance(bore, dict) or bore.get("kind", "cylinder") != "cylinder"
            or any(key not in {"kind", "center", "radius"} for key in bore)):
        raise ValueError("branch-fitted mesh supports only a straight circular cylinder (center, radius, kind)")
    source = np.asarray(source, float)
    center, radius = np.asarray(bore["center"], float), float(bore["radius"])
    gap, inset = float(boundary_relative_gap), float(entrance_relative_inset)
    if (source.shape != (3,) or center.shape != (2,) or not np.isfinite(source).all()
            or not np.isfinite(center).all() or not np.isfinite([radius, z0, z1, gap, inset]).all()
            or radius <= 0 or not source[2] < z0 < z1 or not 0 < gap < 1 or not 0 < inset < 1):
        raise ValueError("finite source before entrance, positive cylinder length/radius and gaps in (0,1) required")
    if np.linalg.norm(source[:2]-center) >= radius:
        raise ValueError("source projection must be strictly inside the bore for an unclipped 0/1 boundary circle")
    a, length = z0-source[2], z1-z0
    circle_center = center+length/(a+length)*(source[:2]-center)
    circle_radius = radius*a/(a+length)
    theta = np.arange(angles)*2*np.pi/angles
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    inner_radius = circle_radius*(1-gap)
    reflected_radius = circle_radius*(1+gap)/np.cos(np.pi/angles)
    offset = circle_center-center
    dot = unit@offset
    radical = dot**2+(radius*(1-inset))**2-offset@offset
    if np.any(radical <= 0):
        raise ValueError("inset aperture does not enclose the family boundary center")
    outer_radius = -dot+np.sqrt(radical)
    if np.any(outer_radius <= reflected_radius):
        raise ValueError("circumscribed family boundary does not fit aperture; increase angles or reduce gaps/inset")

    inner_radii = inner_radius*np.arange(1, inner_rings+1)/inner_rings
    disk = np.vstack((circle_center, (circle_center+inner_radii[:, None, None]*unit).reshape(-1, 2)))
    disk_index = lambda ring, angle: 1+(ring-1)*angles+angle % angles
    faces = [[0, disk_index(1, j), disk_index(1, j+1)] for j in range(angles)]
    for ring in range(1, inner_rings):
        for j in range(angles):
            lo, hi = disk_index(ring, j), disk_index(ring+1, j)
            faces.extend(([lo, hi, disk_index(ring+1, j+1)],
                          [lo, disk_index(ring+1, j+1), disk_index(ring, j+1)]))
    disk_faces = len(faces)
    annulus_radii = reflected_radius+(outer_radius-reflected_radius)*np.linspace(0, 1, outer_rings+1)[:, None]
    annulus = (circle_center+annulus_radii[..., None]*unit).reshape(-1, 2)
    annulus_index = lambda ring, angle: len(disk)+ring*angles+angle % angles
    for ring in range(outer_rings):
        for j in range(angles):
            lo, hi = annulus_index(ring, j), annulus_index(ring+1, j)
            faces.extend(([lo, hi, annulus_index(ring+1, j+1)],
                          [lo, annulus_index(ring+1, j+1), annulus_index(ring, j+1)]))
    points = np.vstack((disk, annulus))
    faces = np.asarray(faces, np.int64)
    families = np.r_[np.zeros(len(disk), np.int64), np.ones(len(annulus), np.int64)]
    direct_polygon = disk[-angles:]
    reflected_polygon = annulus[:angles]
    outer_polygon = annulus[-angles:]
    direct_area = _polygon_area(direct_polygon)
    reflected_hole_area = _polygon_area(reflected_polygon)
    outer_area = _polygon_area(outer_polygon)
    aperture_area = np.pi*radius**2
    band_area = reflected_hole_area-direct_area
    outer_deficit = aperture_area-outer_area
    geometry = dict(
        entrance_nodes=len(points), entrance_triangles=len(faces),
        triangles_by_reflections={"0": disk_faces, "1": len(faces)-disk_faces},
        nodes_by_reflections={"0": len(disk), "1": len(annulus)},
        family_circle_center_m=circle_center.tolist(), family_circle_radius_m=float(circle_radius),
        direct_boundary_radius_m=float(inner_radius), reflected_boundary_vertex_radius_m=float(reflected_radius),
        reflected_boundary_minimum_chord_radius_m=float(reflected_radius*np.cos(np.pi/angles)),
        aperture_area_m2=float(aperture_area), direct_area_m2=float(direct_area),
        reflected_area_m2=float(outer_area-reflected_hole_area),
        accepted_area_m2=float(outer_area-band_area), accepted_area_fraction=float((outer_area-band_area)/aperture_area),
        family_boundary_band_area_m2=float(band_area), family_boundary_band_fraction=float(band_area/aperture_area),
        outer_chord_and_inset_deficit_m2=float(outer_deficit), outer_chord_and_inset_deficit_fraction=float(outer_deficit/aperture_area),
        direct_domain_deficit_m2=float(np.pi*circle_radius**2-direct_area),
        reflected_domain_deficit_m2=float(outer_deficit+reflected_hole_area-np.pi*circle_radius**2),
        angles=angles, inner_rings=inner_rings, outer_rings=outer_rings,
        boundary_relative_gap=gap, entrance_relative_inset=inset,
        topology="Explicit nonoverlapping polar disk and annulus; matching edges within each domain; a measured unmeshed band separates families.")
    return points, faces, families, geometry


def cylinder_exit_mesh(source_origin_decimal, bore, z0, z1, k, delta, beta, *,
                       angles=256, inner_rings=16, outer_rings=8,
                       boundary_relative_gap=1e-6, entrance_relative_inset=2e-6,
                       precision=64, determinant_floor=1e-10, amplitude_mode="tube_flux"):
    """Trace a supplied 0/1-family mesh and return ``(mesh, diagnostics)``.

    Geometry must enclose the complete family circle and have only the prescribed
    histories at all nodes. Unsupported histories, invalid GO nodes and folded
    faces raise; no fallback Delaunay, ray retries, or hidden face removal occurs.
    Area deficits and local phase diagnostics are not bounds on field error.
    """
    started = perf_counter()
    precision = _integer(precision, "precision", 32)
    if amplitude_mode not in {"point_jacobian", "tube_flux"}:
        raise ValueError("amplitude_mode must be point_jacobian or tube_flux")
    k, delta, beta, determinant_floor = map(float, (k, delta, beta, determinant_floor))
    if not np.isfinite([k, delta, beta, determinant_floor]).all() or k <= 0 or beta < 0 or determinant_floor < 0:
        raise ValueError("invalid wavenumber, material or determinant floor")
    source_decimal = [str(value) for value in source_origin_decimal]
    source = np.asarray(source_decimal, float)
    z0_decimal, z1_decimal = str(z0), str(z1)
    z0, z1 = float(z0), float(z1)
    entrance, faces, expected, geometry = _entrance_mesh(source, bore, z0, z1,
        angles=angles, inner_rings=inner_rings, outer_rings=outer_rings,
        boundary_relative_gap=boundary_relative_gap, entrance_relative_inset=entrance_relative_inset)
    center, radius = [str(c) for c in bore["center"]], str(bore["radius"])
    optic = CapillaryBundle([dict(kind="cylinder", center=tuple(lift(c, precision) for c in center),
                                 radius=lift(radius, precision))], lift(z0_decimal, precision), lift(z1_decimal, precision))
    tracer = make_tracer(optic)
    origin = tuple(lift(c, precision) for c in source_decimal)
    screen_z = lift(z1_decimal, precision)
    cap = SimpleNamespace(z0=z0, z1=z1, bores=[dict(center=list(map(float, center)), radius=float(radius))])
    points, directions, phases, reflections, sins = [], [], [], [], []
    traced = perf_counter()
    with localcontext() as context:
        context.prec = max(100, precision+30)
        axial = Decimal(z1_decimal)-Decimal(source_decimal[2])
        for ray_id, q in enumerate(entrance):
            target = (lift(float(q[0]), precision), lift(float(q[1]), precision), optic.z0)
            ray = tracer(origin, vunit(vsub(target, origin)), optic, screen_z, 200)
            if ray.fate != "screen" or len(ray.reflections) != expected[ray_id]:
                raise ValueError(f"prescribed node {ray_id}: expected {expected[ray_id]} reflections and screen; "
                                 f"tracer returned {len(ray.reflections)} reflections, fate={ray.fate}; "
                                 "geometry is outside the verified 0/1-family model")
            points.append([float(v) for v in ray.point[:2]])
            directions.append([float(v) for v in ray.direction[:2]])
            phases.append(k*float(Decimal(str(ray.opl))-axial))
            reflections.append([[float(v) for v in hit] for hit, _ in ray.reflections])
            sins.append([float(sine) for _, sine in ray.reflections])
    trace_seconds = perf_counter()-traced
    mode = dict(origin=source, points=np.asarray(points), directions=np.asarray(directions),
                phase_opl=np.asarray(phases), refl=reflections, sins=sins, ray_ids=np.arange(len(entrance)))
    nodes = exit_nodes(mode, cap, mode_z=z1, exit_z=z1, k=k,
                       fresnel=_fresnel_product(sins, delta, beta), determinant_floor=determinant_floor)
    entrance_defect = np.max(np.linalg.norm(nodes["entrance"]-entrance, axis=1))
    if entrance_defect > 1e-9*float(radius):
        raise ValueError("traced entrance coordinates do not recover the prescribed mesh")
    mesh = mesh_from_triangles(nodes, cap, faces)
    pre_tube = dict(mesh["metadata"])
    if amplitude_mode == "tube_flux":
        mesh = apply_tube_flux(mesh, nodes)
    diagnostic = dict(
        method="branch-fitted straight-cylinder 0/1 entrance mesh through existing multiprecision tracer",
        source_origin_decimal=source_decimal, bore=dict(center=list(map(float, center)), radius=float(radius), kind="cylinder"),
        z0=z0, z1=z1, k=k, delta=delta, beta=beta, geometry=geometry,
        trace=dict(backend="python multiprecision" if tracer is trace_ray else "native multiprecision",
                   precision=precision, emitted_nodes=len(entrance), screen_nodes=len(entrance),
                   verified_reflection_families=geometry["nodes_by_reflections"],
                   maximum_recovered_entrance_defect_m=float(entrance_defect), seconds=trace_seconds),
        phase_convention="k*(OPL-(exit_z-source_z)); Decimal axial-carrier subtraction before conversion to float; Fresnel and Maslov in complex amplitude",
        amplitude_mode=amplitude_mode, pre_tube=pre_tube, mesh=mesh["metadata"],
        seconds=perf_counter()-started,
        limitations=["Straight circular bore and a complete, interior 0/1-reflection boundary only.",
                     "Geometric area deficits are measured; they do not bound oscillatory field or coherence error.",
                     "Finite polar triangles approximate the nonlinear exit map; angular/radial and field quadrature convergence remain required.",
                     "This does not regularize exit caustics or restore other bores' archive mesh holes.",
                     "Finite-tube branch flux excludes interference cross terms; it is not total coherent power."])
    mesh["metadata"]["prescribed_cylinder_geometry"] = geometry
    return mesh, diagnostic
