"""Stage 19: resumable 18LITE fields and ensemble coherence, without NumPy."""

from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
from pathlib import Path
import gzip
import importlib
import json
import lzma
import math
import os
import struct
import sys
import time
from xml.sax.saxutils import escape

from ...formula import Number
from .. import rays_v3, render
from ..rays import _scene_core, geometry_metadata, metadata_equal
from ._s19_config import validate


FORMAT = 1


def _mesh_suffix(options):
    compression = options.get("archive_compression", "gzip")
    if not isinstance(compression, str) or compression not in {"gzip", "xz"}:
        raise ValueError("stage19.archive_compression must be gzip or xz")
    return ".jsonl.xz" if compression == "xz" else ".jsonl.gz"


def _mesh_open(path, mode, *, archive_path=None):
    """Open a lossless mesh stream, using the final name for atomic writes.

    Compression changes only the container; JSONL headers and MP record text
    have the same representation in both formats. Coherence exports remain gzip.
    """
    suffix = Path(archive_path if archive_path is not None else path).suffix
    options = {"encoding": "utf-8", "newline": "\n"} if mode == "wt" else {"encoding": "utf-8"}
    if suffix == ".gz":
        if mode == "wt":
            options["compresslevel"] = 3
        return gzip.open(path, mode, **options)
    if suffix == ".xz":
        if mode == "wt":
            options["preset"] = 6
        return lzma.open(path, mode, **options)
    raise ValueError("Stage 19 mesh archive must have a .gz or .xz suffix")


def _dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _native():
    try:
        return importlib.import_module("formula._stage19")
    except ImportError as exc:
        raise ValueError("Stage 19 needs the formula._stage19 C++ extension; rebuild formula") from exc


def _screen_specs(sim, options):
    cap = sim.cfg.capillary
    screens = [cap.screen, *cap.screens]
    indices = options["screen_indices"]
    if indices is None:
        indices = list(range(len(screens)))
    result = []
    for index in indices:
        if index >= len(screens):
            raise ValueError("stage19.screen_indices refers to an absent screen")
        screen = screens[index]
        if screen.nx < 1 or screen.ny < 1 or float(screen.edge_x) <= 0 or float(screen.edge_y) <= 0:
            raise ValueError("Stage 19 observation grids need positive sizes")
        distance = screen.z - cap.z1
        if float(distance) < 0:
            raise ValueError("Stage 19 observation plane is before the capillary exit")
        result.append({
            "name": "exit" if float(distance) == 0 else f"screen-{index}",
            "screen_index": index,
            "z": str(screen.z), "distance": str(distance),
            "center": [str(v) for v in screen.center],
            "edge_x": str(screen.edge_x), "edge_y": str(screen.edge_y),
            "nx": screen.nx, "ny": screen.ny,
            "reference": [str(v) for v in (screen.reference or screen.center)],
        })
    names = [row["name"] for row in result]
    if len(names) != len(set(names)):
        raise ValueError("Stage 19 supports one exit observation grid per run")
    return result


def read_sources(sim, options, rays_paths=None):
    """Read only source headers; ray bodies are neither read nor certified."""
    archive = options["source_archive"]
    if archive is None:
        paths = list(rays_paths or ())
        if len(paths) != 1:
            raise ValueError("Stage 19 requires one source_archive or one --replay archive")
        archive = paths[0]
    archive = Path(archive).resolve()
    fingerprint = rays_v3.read_fingerprint(archive)
    expected = geometry_metadata(sim.cfg)
    if not metadata_equal(_scene_core(fingerprint["geometry"], "capillary"),
                          _scene_core(expected, "capillary")):
        raise ValueError("Stage 19 archive source/optic differs from the configured scene")
    index = rays_v3.load_index(archive)
    modes = index.modes("capillary")
    count = options["max_modes"] or sim.cfg.capillary.source.n_modes
    if count > len(modes):
        raise ValueError(f"Stage 19 needs {count} sources, archive contains {len(modes)}")
    sources = []
    for mode, sections in enumerate(modes[:count]):
        origin = rays_v3.section_header(archive, sections[0]).get("origin")
        if not isinstance(origin, (list, tuple)) or len(origin) != 3:
            raise ValueError(f"Stage 19 source header {mode} lacks a 3D origin")
        values = [str(Number(str(v), sim.cfg.precision)) for v in origin]
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f"Stage 19 source header {mode} is not finite")
        sources.append({"mode": mode, "origin": values, "weight": 1.0 / count})
    return sources, {
        "archive": str(archive), "index_sha256": rays_v3.index_digest(archive),
        "source_rule": "equal-weight recorded point-source realizations",
        "header_count": count, "ray_bodies_read": False,
        "ray_body_checksums_verified": False,
    }


def preflight_stage19_inputs(sim, options):
    if sim.cfg.capillary is None:
        raise ValueError("Stage 19 requires a capillary scene")
    if len(sim.lines) != 1:
        raise ValueError("Stage 19 prototype requires a monochromatic spectrum")
    for bore in sim.cfg.capillary.bores:
        if "radius" not in bore or "sides" in bore or bore.get("kind", "cylinder") not in {"cylinder", "torus"}:
            raise ValueError("Stage 19 prototype requires circular bores")
    for screen in _screen_specs(sim, options):
        _grid(screen)
    _native()
    if options["mesh"].get("transport_backend", "decimal") == "interval":
        importlib.import_module("formula._stage19_transport")


def preflight_stage19_output(out_dir, options):
    path = Path(out_dir) / "stage19"
    if path.exists() and not options["resume"]:
        raise ValueError("stage19 output exists and resume is disabled")


def _accepted_node_refs(triangles):
    """IDs are local to each bore's trace cache, never archive row numbers."""
    references = set()
    for triangle in triangles:
        bore = triangle.get("bore", 0)
        if "center_node_id" in triangle:
            references.add((bore, triangle["center_node_id"]))
        for key in ("probe_node_ids", "child_center_node_ids"):
            references.update((bore, node_id) for node_id in triangle.get(key, ()))
    return references


def save_mesh(path, source, mesh, *, node_policy="all"):
    """Save all MP field elements/omissions, with optional trace-cache records.

    The standalone default retains the historical full-cache behaviour. The
    stage driver explicitly selects its configured storage policy instead.
    """
    if not isinstance(node_policy, str) or node_policy not in {"none", "accepted", "all"}:
        raise ValueError("Stage 19 mesh node_policy must be none, accepted or all")
    nodes = mesh.get("nodes", [])
    if isinstance(nodes, dict):
        nodes = nodes.values()
    references = _accepted_node_refs(mesh["triangles"]) if node_policy == "accepted" else set()
    if node_policy == "none":
        saved_nodes = ()
    elif node_policy == "accepted":
        saved_nodes = [node for node in nodes if (node.get("bore", 0), node.get("id")) in references]
    else:
        saved_nodes = nodes
    traced_nodes = mesh["stats"].get("nodes", len(nodes))
    if traced_nodes < len(nodes):
        raise ValueError("Stage 19 mesh has more node records than traced nodes")
    storage = {
        "archive_nodes": node_policy,
        "node_records_traced": traced_nodes,
        "node_records_available": len(nodes),
        "node_records_saved": len(saved_nodes),
        "node_records_omitted": traced_nodes - len(saved_nodes),
        "accepted_elements_saved": len(mesh["triangles"]),
        "unresolved_elements_saved": len(mesh.get("unresolved", [])),
        "node_id_namespace": "per-source, per-bore trace-cache IDs; not archive row numbers or guaranteed archive pointers",
        "node_selection": {"none": "No trace-cache node records; all accepted field models and unresolved elements retained.",
                           "accepted": "Only node IDs explicitly referenced by accepted elements; not a complete record of every acceptance check.",
                           "all": "All supplied trace-cache node records, including rejected parent probes."}[node_policy],
    }
    if node_policy == "accepted":
        present = {(node.get("bore", 0), node.get("id")) for node in saved_nodes}
        storage["referenced_nodes_missing"] = len(references - present)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with _mesh_open(temporary, "wt", archive_path=path) as handle:
        handle.write(json.dumps({"header": FORMAT, "source": source,
                                 "stats": mesh["stats"], "archive_storage": storage}, allow_nan=False) + "\n")
        for kind, key in (("triangle", "triangles"), ("node", "nodes"), ("unresolved", "unresolved")):
            records = saved_nodes if key == "nodes" else mesh.get(key, [])
            if isinstance(records, dict):
                records = records.values()
            for record in records:
                handle.write(json.dumps({"kind": kind, "record": record},
                                        allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)
    return storage


def load_mesh(path, *, load_nodes=True):
    """Load field models; optionally avoid retaining trace-cache records in RAM."""
    mesh = {"triangles": [], "nodes": [], "unresolved": []}
    keys = {"triangle": "triangles", "node": "nodes", "unresolved": "unresolved"}
    with _mesh_open(path, "rt") as handle:
        header = json.loads(next(handle))
        if header.get("header") != FORMAT:
            raise ValueError("Unsupported Stage 19 mesh archive")
        mesh["stats"] = header["stats"]
        node_count = 0
        for line in handle:
            row = json.loads(line)
            if row["kind"] == "node":
                node_count += 1
                if not load_nodes:
                    continue
            mesh[keys[row["kind"]]].append(row["record"])
    storage = header.get("archive_storage")
    if storage is None:
        storage = {"archive_nodes": "legacy_unspecified", "node_records_saved": node_count,
                   "node_id_namespace": "per-source, per-bore trace-cache IDs; not archive row numbers",
                   "node_selection": "Legacy archive has no explicit node-storage policy."}
    elif storage["node_records_saved"] != node_count:
        raise ValueError("Stage 19 mesh archive node count differs from its header")
    mesh["archive_storage"] = storage
    return header["source"], mesh


def _mesh_header(path):
    with _mesh_open(path, "rt") as handle:
        header = json.loads(next(handle))
    if header.get("header") != FORMAT:
        raise ValueError("Unsupported Stage 19 mesh archive")
    return header


def _mesh_rows(path):
    """Stream original JSONL rows and verify the declared saved-record counts."""
    with _mesh_open(path, "rt") as handle:
        header = json.loads(next(handle))
        if header.get("header") != FORMAT:
            raise ValueError("Unsupported Stage 19 mesh archive")
        counts = {"triangle": 0, "node": 0, "unresolved": 0}
        for line in handle:
            row = json.loads(line)
            if row.get("kind") not in counts or "record" not in row:
                raise ValueError("Invalid Stage 19 mesh archive record")
            counts[row["kind"]] += 1
            yield row, line
    storage = header.get("archive_storage", {})
    for kind, key in (("triangle", "accepted_elements_saved"),
                      ("node", "node_records_saved"), ("unresolved", "unresolved_elements_saved")):
        if key in storage and storage[key] != counts[kind]:
            raise ValueError(f"Stage 19 mesh archive {kind} count differs from its header")


def _mesh_records(paths, kind):
    for path in paths:
        for row, _ in _mesh_rows(path):
            if row["kind"] == kind:
                yield row["record"]


def _combine_bore_archives(path, source, parts, mesh_options):
    """Publish all original MP rows without holding all bores' models in RAM."""
    headers = [header for _, header in parts]
    stats = dict(headers[0]["stats"])
    stats["bores"] = [bore for header in headers for bore in header["stats"]["bores"]]
    for key in ("seconds", "nodes", "accepted_triangles", "unresolved_triangles", "physics_pending_nodes"):
        stats[key] = sum(header["stats"].get(key, 0) for header in headers)
    stats["accepted_triangles"] = sum(header["archive_storage"]["accepted_elements_saved"] for header in headers)
    stats["unresolved_triangles"] = sum(header["archive_storage"]["unresolved_elements_saved"] for header in headers)
    stats["peak_bore_retained_nodes"] = max(header["stats"].get("retained_nodes", 0) for header in headers)
    stats["retained_nodes"] = 0
    stats["driver_storage"] = "MP models and trace nodes released after each bore; native field rows retained"
    stats["options"] = dict(stats.get("options", {}), **mesh_options)
    if "bores" not in mesh_options:
        stats["options"].pop("bores", None)
    storage = dict(headers[0]["archive_storage"])
    for key in ("node_records_traced", "node_records_available", "node_records_saved", "node_records_omitted",
                "accepted_elements_saved", "unresolved_elements_saved", "referenced_nodes_missing"):
        if any(key in header["archive_storage"] for header in headers):
            storage[key] = sum(header["archive_storage"].get(key, 0) for header in headers)
    if any(header["archive_storage"]["archive_nodes"] != storage["archive_nodes"] for header in headers):
        raise ValueError("Stage 19 bore archives have incompatible node policies")
    result = {"header": FORMAT, "source": source, "stats": stats, "archive_storage": storage}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with _mesh_open(temporary, "wt", archive_path=path) as handle:
        handle.write(json.dumps(result, allow_nan=False) + "\n")
        for part_path, expected in parts:
            if _mesh_header(part_path) != expected:
                raise ValueError("Stage 19 bore checkpoint changed during source assembly")
            for _, line in _mesh_rows(part_path):
                handle.write(line)
    os.replace(temporary, path)
    return result


def mesh_svg(path, triangles, spec, source_mode, unresolved=(), *, partial=False):
    """One realization's input partition; red fill marks unresolved elements."""
    width, height = spec["nx"], spec["ny"]
    cx, cy = map(float, spec["center"])
    ex, ey = float(spec["edge_x"]), float(spec["edge_y"])
    x0, y0 = cx - ex / 2, cy - ey / 2
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        title = f"Stage 19 input triangles, source {source_mode}"
        if partial:
            title += " — PARTIAL SOURCE: selected bores are missing or have not passed acceptance"
        handle.write(f'<title>{escape(title)}</title>\n')
        handle.write(f'<rect width="{width}" height="{height}" fill="white"/>\n')
        for triangle in triangles:
            points = " ".join(f"{(float(q[0])-x0)/ex*width:.9g},{height-(float(q[1])-y0)/ey*height:.9g}"
                              for q in triangle["input"])
            color = colors[int(triangle.get("bore", 0)) % len(colors)]
            handle.write(f'<polygon points="{points}" fill="none" stroke="{color}" stroke-width="0.35"/>\n')
        for triangle in unresolved:
            points = " ".join(f"{(float(q[0])-x0)/ex*width:.9g},{height-(float(q[1])-y0)/ey*height:.9g}"
                              for q in triangle["input"])
            handle.write(f'<polygon points="{points}" fill="#D55E00" fill-opacity="0.4" stroke="#D55E00" stroke-width="0.35"/>\n')
        handle.write("</svg>\n")


def _as_array(data):
    result = array("d")
    result.frombytes(data)
    return result


def _write_binary(path, data):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _run_lock(directory):
    handle = (directory / "run.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise ValueError("Stage 19 output is locked by another running invocation") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _records(sim, triangles):
    result = []
    with localcontext() as context:
        context.prec = max(32, sim.cfg.precision)
        k = Decimal(str(sim.lines[0].k))
        turn = Decimal(str(Number("2*pi", sim.cfg.precision)))
        for triangle in triangles:
            phase = float((k * Decimal(triangle["opl"])) % turn)
            vertices = [float(v) for q in triangle["exit"] for v in q[:2]]
            center = [float(v) for v in triangle["center_exit"][:2]]
            gradient = [float(k * Decimal(v)) for v in triangle["direction"][:2]]
            amplitude = [float(v) for v in triangle["amplitude"]]
            record = [*vertices, *center, *gradient, *amplitude, phase]
            if len(record) != 13 or any(not math.isfinite(v) for v in record):
                raise ValueError("Invalid native Stage 19 element")
            result.append(record)
    return result


def _grid(spec):
    nx, ny = spec["nx"], spec["ny"]
    cx, cy = map(float, spec["center"])
    ex, ey = float(spec["edge_x"]), float(spec["edge_y"])
    dx, dy = ex / nx, ey / ny
    if not math.isclose(dx, dy, rel_tol=1e-12, abs_tol=0):
        raise ValueError("Stage 19 prototype requires square observation cells")
    x0, y0 = cx - ex / 2, cy - ey / 2
    xs = [x0 + (i + .5) * dx for i in range(nx)]
    ys = [y0 + (j + .5) * dy for j in range(ny)]
    rx, ry = map(float, spec["reference"])
    if not (x0 <= rx < x0 + ex and y0 <= ry < y0 + ey):
        raise ValueError("Stage 19 prototype needs its reference inside the observation window")
    ix, iy = int((rx - x0) / dx), int((ry - y0) / dy)
    return xs, ys, dx, iy * nx + ix


def _evaluate_field(records, k, spec, observable, options):
    native = _native()
    xs, ys, width, _ = _grid(spec)
    cell = width if observable == "coherent_cell" else 0.0
    distance = float(spec["distance"])
    if distance == 0:
        return native.eval_exit(records, xs, ys, pixel_width=cell)
    kwargs = {key: value for key, value in options.items() if key != "backend"}
    backend = options.get("backend", "contour")
    if backend == "native_nufft":
        from ._s19_accelerated import evaluate_field
        return evaluate_field(records, k, spec, observable, options)
    if backend != "contour":
        raise ValueError(f"Unknown Stage 19 propagation backend {backend!r}")
    return native.eval_contour(records, k, distance, xs, ys, pixel_width=cell, **kwargs)


def _source_task(task):
    from ..simulation import Simulation
    from ._s19_mesh import build_source_mesh

    raw, options, screens, source, target = task
    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "result.json"
    if completed.exists():
        result = json.loads(completed.read_text(encoding="utf-8"))
        if all((directory / row["file"]).exists() for row in result["fields"]):
            return str(completed)
    sim = Simulation.from_dict(raw)
    mesh_suffix = _mesh_suffix(options)
    mesh_path = directory / ("mesh" + mesh_suffix)
    mesh_options = dict(options["mesh"], screens=screens)
    bore_ids = list(mesh_options.get("bores", range(len(sim.cfg.capillary.bores))))
    if (len(set(bore_ids)) != len(bore_ids)
            or any(isinstance(i, bool) or not isinstance(i, int)
                   or not 0 <= i < len(sim.cfg.capillary.bores) for i in bore_ids)):
        raise ValueError("Stage 19 checkpoint bores must be distinct valid integer indices")
    mesh_signature = _digest({"raw": raw, "options": options, "screens": screens,
                              "source": source, "code_sha256": _code_hashes(options)})
    bore_paths = [(bore, directory / f"mesh-bore-{bore:04d}{mesh_suffix}") for bore in bore_ids]
    started = time.perf_counter()
    if not bore_paths:
        raise ValueError("Stage 19 needs at least one selected bore")
    parts, records = [], []

    def gate(header, paths, *, failed_bore=None):
        count = header["archive_storage"]["accepted_elements_saved"]
        try:
            _mesh_gate({"stats": header["stats"]}, options, accepted_count=count)
        except ValueError as exc:
            covered = [bore for bore, path in bore_paths if path in paths]
            missing = [bore for bore in bore_ids if bore not in covered]
            if source["mode"] == 0:
                mesh_svg(directory / "input-triangles.svg", _mesh_records(paths, "triangle"),
                         screens[0], source["mode"], _mesh_records(paths, "unresolved"), partial=True)
            _dump(directory / "incomplete.json", {
                "status": "mesh_acceptance_failed", "source": source,
                "reason": str(exc), "stats": header["stats"],
                "archive_storage": header["archive_storage"], "failed_bore": failed_bore,
                "saved_bores": covered, "missing_bores": missing,
                "source_partition_complete": False, "fields_computed": False,
            })
            raise

    if mesh_path.exists():
        header = _mesh_header(mesh_path)
        if header["source"] != source:
            raise ValueError("Stage 19 mesh source differs from the run manifest")
        gate(header, [mesh_path])
        records = _records(sim, _mesh_records([mesh_path], "triangle"))
    else:
        for bore, path in bore_paths:
            identity = {"source": source, "bore": bore, "signature": mesh_signature}
            if path.exists():
                part_header = _mesh_header(path)
                if part_header["source"] != identity:
                    raise ValueError("Stage 19 bore checkpoint differs from its source/configuration/code")
                part = None
            else:
                part = build_source_mesh(sim, source["origin"], dict(mesh_options, bores=[bore]),
                                         retain_nodes=options.get("archive_nodes", "none") != "none")
                save_mesh(path, identity, part, node_policy=options.get("archive_nodes", "none"))
                part_header = _mesh_header(path)
            parts.append((path, part_header))
            # A failed bore is durable and stops work before tracing another bore.
            gate(part_header, [saved_path for saved_path, _ in parts], failed_bore=bore)
            if part is None:
                records.extend(_records(sim, _mesh_records([path], "triangle")))
            else:
                records.extend(_records(sim, part["triangles"]))
                part["triangles"], part["nodes"], part["unresolved"] = [], [], []
                del part
        header = _combine_bore_archives(mesh_path, source, parts, mesh_options)
        gate(header, [mesh_path])
    # The final MP archive is published before known intermediate files disappear.
    for bore, path in bore_paths:
        if path.exists() and _mesh_header(path)["source"] == {
                "source": source, "bore": bore, "signature": mesh_signature}:
            path.unlink()
    if source["mode"] == 0:
        mesh_svg(directory / "input-triangles.svg", _mesh_records([mesh_path], "triangle"),
                 screens[0], source["mode"], _mesh_records([mesh_path], "unresolved"))
    (directory / "incomplete.json").unlink(missing_ok=True)
    fields = []
    for screen in screens:
        for observable in options["observables"]:
            began = time.perf_counter()
            result = _evaluate_field(records, float(sim.lines[0].k), screen, observable, options["kernel"])
            data = result["field"]
            if len(data) != 16 * screen["nx"] * screen["ny"]:
                raise ValueError("Stage 19 kernel returned the wrong field buffer size")
            name = f"{screen['name']}-{observable}.bin"
            _write_binary(directory / name, data)
            fields.append({"name": f"{screen['name']}-{observable}", "file": name,
                           "sha256": sha256(data).hexdigest(),
                           "diagnostics": result.get("diagnostics", {}),
                           "seconds": time.perf_counter() - began})
    result = {"mode": source["mode"], "source": source, "stats": header["stats"],
              "archive_storage": header["archive_storage"],
              "fields": fields, "seconds": time.perf_counter() - started}
    _dump(completed, result)
    return str(completed)


def _mesh_gate(mesh, options, *, accepted_count=None):
    if accepted_count is None:
        accepted_count = len(mesh["triangles"])
    if accepted_count < 1:
        raise ValueError("Stage 19 source has no accepted field elements; mesh diagnostics were saved")
    physical_exit = options["mesh"].get("phase_coordinates", "pullback") == "physical_exit"
    if physical_exit and not mesh["stats"]["bores"]:
        raise ValueError("Stage 19 physical_exit mesh has no bore domain diagnostics")
    for bore in mesh["stats"]["bores"]:
        accepted = float(bore["accepted_fraction"])
        if not math.isfinite(accepted) or not 0 <= accepted <= 1:
            raise ValueError(f"Stage 19 bore {bore['bore']} has invalid accepted fraction {bore['accepted_fraction']}")
        if accepted < options["minimum_accepted_fraction"]:
            raise ValueError(f"Stage 19 bore {bore['bore']} accepted fraction {bore['accepted_fraction']} "
                             f"is below {options['minimum_accepted_fraction']}; mesh diagnostics were saved")
        if physical_exit:
            diagnostic = bore.get("domain_indicator")
            if not isinstance(diagnostic, dict) or diagnostic.get("passed") is not True:
                raise ValueError(f"Stage 19 bore {bore['bore']} physical_exit domain indicator is missing or failed")
            try:
                relative = Decimal(str(diagnostic["weighted_domain_relative"]))
                tolerance = Decimal(str(diagnostic["tolerance"]))
                expected_tolerance = Decimal(str(options["mesh"].get("domain_relative", ".02")))
                normalizer = Decimal(str(diagnostic["normalizer_sum_abs_A_exit_area"]))
                geometry_ok = (diagnostic["exit_vertex_disagreements"] == 0
                               and diagnostic["counts"].get("nonmanifold", 0) == 0
                               and diagnostic["p6_fan_checks"]["non_simple"] == 0
                               and diagnostic["p6_fan_checks"]["bad_fan_orientation"] == 0)
            except (KeyError, TypeError, ValueError, AttributeError, InvalidOperation) as exc:
                raise ValueError(f"Stage 19 bore {bore['bore']} has invalid domain diagnostics") from exc
            if (not all(value.is_finite() for value in (relative, tolerance, normalizer, expected_tolerance))
                    or not 0 <= relative <= tolerance or tolerance <= 0 or normalizer <= 0 or not geometry_ok):
                raise ValueError(f"Stage 19 bore {bore['bore']} has invalid or failed domain diagnostics")
            if tolerance != expected_tolerance:
                raise ValueError(f"Stage 19 bore {bore['bore']} domain tolerance differs from the configured gate")


def _write_state(path, state):
    scalars, buffers, descriptors = {}, [], []
    for key, value in state.items():
        if isinstance(value, (bytes, bytearray)):
            buffers.append(value)
            descriptors.append([key, len(value)])
        else:
            scalars[key] = value
    header = json.dumps({"format": FORMAT, "byteorder": sys.byteorder,
                         "scalars": scalars, "buffers": descriptors}, allow_nan=False).encode()
    _write_binary(path, b"F19ACC1\n" + struct.pack("<Q", len(header)) + header + b"".join(buffers))


def _read_state(path):
    with open(path, "rb") as handle:
        if handle.read(8) != b"F19ACC1\n":
            raise ValueError("Invalid Stage 19 accumulator checkpoint")
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 100000:
            raise ValueError("Invalid Stage 19 accumulator header length")
        header = json.loads(handle.read(length))
        if header["format"] != FORMAT or header["byteorder"] != sys.byteorder:
            raise ValueError("Incompatible Stage 19 accumulator checkpoint")
        state = header["scalars"]
        for name, size in header["buffers"]:
            data = handle.read(size)
            if len(data) != size:
                raise ValueError("Truncated Stage 19 accumulator checkpoint")
            state[name] = data
        if handle.read(1):
            raise ValueError("Trailing bytes in Stage 19 accumulator checkpoint")
    return state


def _code_hashes(options=None):
    from ... import _formula
    directory = Path(__file__).parent
    names = ["_s19_config.py", "_s19_mesh.py", "_s19_transport.py"]
    if options and options["mesh"].get("record_cache_directory") is not None:
        names.append("_s19_cache.py")
    if options and options["mesh"].get("phase_coordinates", "pullback") == "physical_exit":
        names.append("_s19_domain.py")
    if options and options["kernel"].get("backend", "contour") == "native_nufft":
        names += ["_s19_accelerated.py", "_s19_nufft.py"]
    paths = [Path(__file__), *(directory / name for name in names),
             Path(_native().__file__), Path(_formula.__file__)]
    if options and options["mesh"].get("transport_backend", "decimal") == "interval":
        paths += [directory / "_s19_fast_transport.py",
                  Path(importlib.import_module("formula._stage19_transport").__file__)]
    result = {}
    for path in paths:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[path.name] = digest.hexdigest()
    return result


def _checkpoint(directory, accumulators, count, signature, previous):
    names = []
    for index, accumulator in enumerate(accumulators.values()):
        name = f"acc-{count:06d}-{index}.bin"
        _write_state(directory / name, accumulator.state())
        names.append(name)
    value = {"format": FORMAT, "signature": signature, "n_modes": count,
             "maps": list(accumulators), "states": names}
    _dump(directory / "checkpoint.json", value)
    for name in previous:
        if name not in names:
            (directory / name).unlink(missing_ok=True)
    return names


def _export_map(directory, spec, observable, result, intensity_floor, *, target_modes=None):
    directory.mkdir(parents=True, exist_ok=True)
    intensity, mutual, coherence = (_as_array(result[key]) for key in ("I", "W", "mu"))
    ref_intensity = result["reference_intensity"]
    if not math.isfinite(ref_intensity) or ref_intensity <= 0:
        raise ValueError("Stage 19 reference has no finite positive ensemble intensity")
    maximum = max(intensity, default=0.0)
    threshold = intensity_floor * maximum
    nx, ny = spec["nx"], spec["ny"]
    xs, ys, width, reference = _grid(spec)
    valid = result["valid"]
    mu_grid, i_grid, over_one = [], [], 0
    rows = directory / "coherence.jsonl.gz"
    with gzip.open(rows, "wt", encoding="utf-8", newline="\n", compresslevel=3) as handle:
        for iy in range(ny):
            mu_row, i_row = [], []
            for ix in range(nx):
                index = iy * nx + ix
                real, imag = coherence[2 * index:2 * index + 2]
                good = bool(valid[index] and intensity[index] > threshold and ref_intensity > threshold
                            and math.isfinite(real) and math.isfinite(imag))
                magnitude = math.hypot(real, imag) if good else None
                over_one += bool(good and magnitude > 1 + 1e-10)
                mu_row.append(magnitude)
                i_row.append(intensity[index] if intensity[index] > 0 else None)
                handle.write(json.dumps({"pixel": index, "x": xs[ix], "y": ys[iy],
                    "I": intensity[index], "W": [mutual[2*index], mutual[2*index+1]],
                    "mu": [real, imag] if good else None, "abs_mu": magnitude,
                    "valid": good}, allow_nan=False, separators=(",", ":")) + "\n")
            mu_grid.append(mu_row)
            i_grid.append(i_row)
    cx, cy = map(float, spec["center"])
    ex, ey = float(spec["edge_x"]), float(spec["edge_y"])
    extent = tuple(v * 1e6 for v in (cx-ex/2, cx+ex/2, cy-ey/2, cy+ey/2))
    caption = f"{result['n_modes']} source realizations; {observable}; experimental"
    if target_modes is not None:
        caption = f"PRELIMINARY {result['n_modes']}/{target_modes} source realizations; {observable}"
    figure = render.heatmap(mu_grid, extent, "Stage 19 |mu|", "x [um]", "y [um]",
                            subtitle=caption, cbar_label="|mu|", vmax=1, w=nx, h=ny)
    render.save(directory / "coherence.svg", figure)
    image_rows = [[render.viridis(value) if value is not None else (210, 210, 210)
                   for value in row] for row in reversed(mu_grid)]
    (directory / "coherence.png").write_bytes(render._png_bytes(image_rows))
    render.save(directory / "intensity.svg", render.heatmap(i_grid, extent,
                "Stage 19 intensity", "x [um]", "y [um]", subtitle=caption,
                cbar_label="I", log=True, w=nx, h=ny))
    metadata = {"format": FORMAT, "screen": spec, "observable": observable,
                "n_modes": result["n_modes"], "reference_pixel": reference,
                "reference_center": [xs[reference % nx], ys[reference // nx]],
                "reference_intensity": ref_intensity, "cell_width_m": width,
                "intensity_floor": intensity_floor, "valid_pixels": sum(v is not None for row in mu_grid for v in row),
                "over_mu_pixels": over_one, "status": "experimental",
                "conventions": {"W": "mean(E(P) conjugate(E(reference)))",
                    "point": "field at the cell center", "coherent_cell": "field mean over the full square cell",
                    "source": "same equal-weight source set for every map",
                    "phasor": "exp(-i omega t), exp(+i k OPL)",
                    "propagation": "screen-global exp(i k distance) omitted; no cross-plane correlations computed"}}
    if target_modes is not None:
        metadata.update(status="preliminary", target_modes=target_modes,
                        ensemble_complete=result["n_modes"] == target_modes)
    _dump(directory / "meta.json", metadata)
    return metadata


def _coverage_summary(directory, count):
    bores = {}
    total_triangles = total_nodes = total_unresolved = 0
    for mode in range(count):
        result = json.loads((directory / "modes" / f"mode-{mode:06d}" / "result.json").read_text(encoding="utf-8"))
        stats = result["stats"]
        total_triangles += stats.get("accepted_triangles", 0)
        total_nodes += stats.get("nodes", 0)
        total_unresolved += stats.get("unresolved_triangles", 0)
        for row in stats.get("bores", []):
            value = float(row["accepted_fraction"])
            item = bores.setdefault(str(row["bore"]), {"min": value, "max": value, "sum": 0.0,
                "sources": 0, "aperture_deficit_fraction_max": 0.0})
            item["min"], item["max"] = min(item["min"], value), max(item["max"], value)
            item["sum"] += value
            item["sources"] += 1
            item["aperture_deficit_fraction_max"] = max(item["aperture_deficit_fraction_max"], float(row["aperture_deficit_fraction"]))
            diagnostic = row.get("domain_indicator")
            if diagnostic is not None:
                relative = float(diagnostic["weighted_domain_relative"])
                domain = item.setdefault("domain_indicator_relative", {
                    "min": relative, "max": relative, "sum": 0.0, "sources": 0})
                domain["min"], domain["max"] = min(domain["min"], relative), max(domain["max"], relative)
                domain["sum"] += relative
                domain["sources"] += 1
    for item in bores.values():
        item["mean"] = item.pop("sum") / item["sources"]
        if "domain_indicator_relative" in item:
            domain = item["domain_indicator_relative"]
            domain["mean"] = domain.pop("sum") / domain["sources"]
    return {"accepted_fraction_by_bore": bores, "total_accepted_triangles": total_triangles,
            "total_unresolved_triangles": total_unresolved, "total_traced_nodes": total_nodes,
            "note": "Input area coverage is not a bound on field or coherence error."}


def _export_preview(directory, accumulators, specs, count, target, signature, intensity_floor):
    """Publish a complete prefix ensemble without changing its checkpoint."""
    parent = directory / "previews"
    parent.mkdir(exist_ok=True)
    destination = parent / f"modes-{count:06d}"
    if destination.exists():
        saved = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
        if (saved.get("signature"), saved.get("n_modes"), saved.get("target_modes")) != (signature, count, target):
            raise ValueError("Stage 19 existing preview differs from its checkpoint")
        return destination
    staging = parent / f".modes-{count:06d}.tmp"
    staging.mkdir(exist_ok=True)
    maps = []
    for name, accumulator in accumulators.items():
        screen, observable = specs[name]
        maps.append(_export_map(staging / name, screen, observable, accumulator.result(),
                                intensity_floor, target_modes=target))
    _dump(staging / "meta.json", {
        "format": FORMAT, "signature": signature, "status": "preliminary",
        "n_modes": count, "target_modes": target, "ensemble_complete": count == target,
        "source_selection": f"First {count} recorded sources in canonical archive order",
        "normalization": "Equal-weight moments over the actual preview count",
        "maps": maps, "mesh_coverage": _coverage_summary(directory, count),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["Preliminary ensemble; source and mesh convergence are not established.",
                        "Input area coverage is not a field or coherence error bound."],
    })
    os.replace(staging, destination)
    return destination


def run_stage19(sim, out_dir, options=None, *, rays_paths=None, log=print):
    options = validate(sim.cfg) if options is None else options
    preflight_stage19_inputs(sim, options)
    preflight_stage19_output(out_dir, options)
    directory = Path(out_dir) / "stage19"
    directory.mkdir(parents=True, exist_ok=True)
    with _run_lock(directory):
        return _run_stage19(sim, out_dir, options, rays_paths=rays_paths, log=log)


def _run_stage19(sim, out_dir, options, *, rays_paths=None, log=print):
    native = _native()
    screens = _screen_specs(sim, options)
    sources, provenance = read_sources(sim, options, rays_paths)
    if any(count > len(sources) for count in options.get("preview_modes", [])):
        raise ValueError("stage19.preview_modes exceeds the configured source count")
    directory = Path(out_dir) / "stage19"
    (directory / "modes").mkdir(exist_ok=True)
    began = time.perf_counter()
    contract = {"format": FORMAT, "config": sim.cfg.raw, "options": options,
                "screens": screens, "sources": sources, "source_provenance": provenance,
                "code_sha256": _code_hashes(options)}
    signature = _digest(contract)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Stage 19 output belongs to a different configuration/source set")
    else:
        _dump(manifest_path, {"signature": signature, **contract})
    accumulators = {}
    specs = {}
    for screen in screens:
        for observable in options["observables"]:
            name = f"{screen['name']}-{observable}"
            _, _, _, reference = _grid(screen)
            accumulators[name] = native.CoherenceAccumulator(screen["nx"] * screen["ny"], reference)
            specs[name] = (screen, observable)
    count, old_states = 0, []
    checkpoint = directory / "checkpoint.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["signature"] != signature or saved["maps"] != list(accumulators):
            raise ValueError("Stage 19 checkpoint does not match its manifest")
        count, old_states = saved["n_modes"], saved["states"]
        expected = [f"acc-{count:06d}-{index}.bin" for index in range(len(accumulators))]
        if old_states != expected or not 0 <= count <= len(sources):
            raise ValueError("Stage 19 checkpoint has invalid source count or state names")
        for name, state in zip(saved["maps"], old_states):
            values = _read_state(directory / state)
            if values["n_modes"] != count:
                raise ValueError("Stage 19 accumulator source count differs from its checkpoint")
            accumulators[name] = native.CoherenceAccumulator.from_state(values)
    log(f"Stage 19: {count}/{len(sources)} source realizations already accumulated")

    def preview():
        if count in options.get("preview_modes", []):
            try:
                path = _export_preview(directory, accumulators, specs, count, len(sources),
                                       signature, options["intensity_floor"])
            except Exception as exc:
                raise RuntimeError(f"Stage 19 preview export failed at {count} sources; "
                                   f"the accumulated checkpoint is preserved: {exc}") from exc
            log(f"Stage 19: preliminary maps for {count}/{len(sources)} sources: {path}")

    preview()
    tasks = [(sim.cfg.raw, options, screens, source,
              str(directory / "modes" / f"mode-{source['mode']:06d}")) for source in sources[count:]]

    def consume(path):
        nonlocal count, old_states
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if result["mode"] != count:
            raise ValueError("Stage 19 sources must accumulate in canonical mode order")
        if [field["name"] for field in result["fields"]] != list(accumulators):
            raise ValueError("Stage 19 staged source does not contain the expected maps")
        for field in result["fields"]:
            if field["file"] != field["name"] + ".bin":
                raise ValueError("Stage 19 staged field name is invalid")
            data = (Path(path).parent / field["file"]).read_bytes()
            if sha256(data).hexdigest() != field["sha256"]:
                raise ValueError("Stage 19 staged field checksum mismatch")
            accumulators[field["name"]].add(data)
        count += 1
        old_states = _checkpoint(directory, accumulators, count, signature, old_states)
        for field in result["fields"]:
            (Path(path).parent / field["file"]).unlink()
        stats = result["stats"]
        accepted = min((float(v["accepted_fraction"]) for v in stats.get("bores", [])), default=0)
        log(f"Stage 19: source {count}/{len(sources)}; {result['seconds']:.1f} s; "
            f"triangles={stats.get('accepted_triangles', 0)}, nodes={stats.get('nodes', 0)}, "
            f"minimum bore coverage={accepted:.6%}")
        preview()

    if options["workers"] == 1:
        for task in tasks:
            consume(_source_task(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=options["workers"]) as pool:
            pending, iterator = deque(), iter(tasks)
            for _ in range(options["workers"]):
                task = next(iterator, None)
                if task is not None:
                    pending.append(pool.submit(_source_task, task))
            while pending:
                consume(pending.popleft().result())
                task = next(iterator, None)
                if task is not None:
                    pending.append(pool.submit(_source_task, task))
    maps = []
    for name, accumulator in accumulators.items():
        screen, observable = specs[name]
        maps.append(_export_map(directory / name, screen, observable,
                                accumulator.result(), options["intensity_floor"]))
    metadata = {"format": FORMAT, "signature": signature, "n_modes": count,
                "status": "experimental", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "invocation_seconds": time.perf_counter() - began, "maps": maps,
                "source_provenance": provenance,
                "archive_nodes": options.get("archive_nodes", "none"),
                "archive_compression": options.get("archive_compression", "gzip"),
                "phase_coordinates": options["mesh"].get("phase_coordinates", "pullback"),
                "mesh_coverage": _coverage_summary(directory, count),
                "limitations": ["Finite local ray probes do not certify wave accuracy.",
                                "The physical_exit shared-edge domain indicator is a finite-curve L1 envelope test, not a certified full-field or coherence error bound.",
                                "Unresolved input regions remain recorded in every source mesh.",
                                "Field models are retained for every source; node provenance depends on archive_nodes, and cache IDs need not point to saved nodes.",
                                "Monte Carlo convergence and kernel/mesh tolerance ladders require separate checks."]}
    _dump(directory / "meta.json", metadata)
    files = [str(path.relative_to(Path(out_dir))).replace(os.sep, "/")
             for path in directory.rglob("*") if path.is_file() and path.suffix in {".json", ".svg", ".png", ".gz", ".xz"}
             and "modes" not in path.parts]
    files.append("stage19/modes/mode-000000/input-triangles.svg")
    return {"results": metadata, "files": files,
            "report": ["## Stage 19 — 18LITE prototype", f"- {count} recorded source realizations; equal weights",
                       "- Observables: " + ", ".join(options["observables"]),
                       "- Status: experimental; inspect per-source omissions and convergence results.",
                       f"- This invocation: {metadata['invocation_seconds']:.1f} s"]}
