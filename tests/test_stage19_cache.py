"""Cold-cache identity, exact values, failure semantics and real mesh parity."""

from decimal import Decimal as D, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from formula.formula import Number
from formula.capsysred.stages import _s19_cache as C, _s19_mesh as M


def test_eviction_preserves_shared_record_mutations_and_decimal_representation(tmp_path):
    store = C.TraceStore(tmp_path, 1)
    try:
        q = (D("-0.00"), D("1.2300"))
        first = store.create(id=0, q=q, value=D("-0.000"), fresnel_each=[])
        model = {"center": first, "vertices": [first]}
        escaped_list = first["fresnel_each"]
        second = store.create(id=1, q=(D(2), D(3)), value=D("4.50"))
        first["value"] = D("123.4500")
        first["determinant"] = D("0.00100")
        second["value"]
        escaped_list.append((D("1.00"), D("-0.00")))
        assert model["center"] is model["vertices"][0] is first
        assert first["q"] is q
        assert first["value"].as_tuple() == D("123.4500").as_tuple()
        assert first["determinant"].as_tuple() == D("0.00100").as_tuple()
        assert first["fresnel_each"] is escaped_list
        assert first["fresnel_each"][0][1].as_tuple() == D("-0.00").as_tuple()
        assert list(first) == ["id", "q", "value", "fresnel_each", "determinant"]
        assert store.loads and store.evictions and store.bytes_written
        assert len(store.hot) == 1
    finally:
        store.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_cannot_silently_lose_updates(tmp_path, capacity):
    with pytest.raises(ValueError, match="positive integer"):
        C.TraceStore(tmp_path, capacity)


def test_corruption_is_storage_failure_not_rejected_physical_ray(tmp_path):
    store = C.TraceStore(tmp_path, 1)
    try:
        first = store.create(id=0, q=(D(0), D(0)), value=D(1))
        store.create(id=1, q=(D(1), D(1)), value=D(2))
        offset, length = first.location
        store.file.seek(offset)
        store.file.write(b"x" * length)
        with pytest.raises(RuntimeError, match="Cannot decode"):
            first["value"]
    finally:
        store.close()


def test_closed_write_cannot_be_misclassified_as_a_transport_value_error(tmp_path):
    store = C.TraceStore(tmp_path, 1)
    store.close()
    with pytest.raises(RuntimeError, match="Cannot write"):
        store.write({"point": (D(0), D(1), D(2))})


@pytest.mark.parametrize("precision", [32, 64])
def test_native_guard_digits_survive_cold_pending_ray(tmp_path, precision):
    store = C.TraceStore(tmp_path, 1)
    try:
        ray_record = store.create(id=0, q=(D(0), D(0)), point=(D(1), D(2), D(3)),
                                  direction=(D(0), D(0), D(1)))
        original = Number("sqrt(2)/1000", precision)
        cold = store.pending(ray_record, [((D("-0.00"), D("1.20"), D(3)), original)])
        restored = cold.materialize()
        sine = restored.reflections[0][1]
        assert sine is not original  # Backend equality, not object identity, is the codec contract.
        assert sine.precision == original.precision
        assert sine._value == original._value
        assert sine._value.str(0) == original._value.str(0)
        assert restored.reflections[0][0][0].as_tuple() == D("-0.00").as_tuple()
        assert restored.point == ray_record["point"]
        assert restored.direction == ray_record["direction"]
    finally:
        store.close()


def _sim():
    precision = 32
    cap = SimpleNamespace(z0=Number(0, precision), z1=Number(".1", precision),
                          bores=[dict(center=[0, 0], radius="2.4e-5")])
    cfg = SimpleNamespace(precision=precision, max_bounces=20, capillary=cap)
    return SimpleNamespace(cfg=cfg, lines=[1], lam=Number("1e-5", precision),
                           fresnel=lambda sine: Number(1, precision)/(1+sine))


def _without_storage_diagnostics(value):
    if isinstance(value, dict):
        return {key: _without_storage_diagnostics(item) for key, item in value.items()
                if not key.endswith("seconds") and key not in
                {"record_cache", "record_cache_directory", "record_cache_capacity"}}
    if isinstance(value, list):
        return [_without_storage_diagnostics(item) for item in value]
    return value


def test_real_mesh_all_nodes_and_counters_equal_with_forced_cold_eviction(tmp_path):
    sim = _sim()
    options = dict(initial_rings=1, aperture_inset=".001", screens=[], max_nodes=300,
                   max_triangles=200, max_depth=8, lazy_physics=True)
    baseline = M.build_source_mesh(sim, ["0", "0", "-.01"], options)
    actual = M.build_source_mesh(sim, ["0", "0", "-.01"], dict(options,
        record_cache_directory=str(tmp_path), record_cache_capacity=1))
    assert _without_storage_diagnostics(actual) == _without_storage_diagnostics(baseline)
    assert actual["stats"]["bores"][0]["record_cache"]["loads"] > 1
    assert not list(tmp_path.iterdir())


def test_explicit_determinant_refinement_preserves_existing_record_alias(tmp_path):
    sim = _sim()
    with localcontext() as context:
        context.prec = 32
        common = dict(M.DEFAULTS)
        plain = M._TraceCache(sim, (D(0), D(0), D("-.01")), sim.cfg.capillary.bores[0], common, 32)
        disk = M._TraceCache(sim, plain.origin, sim.cfg.capillary.bores[0],
            dict(common, record_cache_directory=str(tmp_path), record_cache_capacity=1), 32)
        try:
            q = (D("1e-6"), D("2e-6"))
            old = disk.get(q)
            plain.get(q)
            disk.get_raw((D("2e-6"), D("1e-6")))
            assert disk.refine_determinant(q) is old
            plain.refine_determinant(q)
            assert M._serialize(old) == M._serialize(plain.records[q])
            assert disk.determinant_retraces == plain.determinant_retraces == 1
        finally:
            disk.close()


def test_mesh_exception_closes_its_private_tempfile(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("test failure after cache creation")
    monkeypatch.setattr(M, "disk_partition", failure)
    with pytest.raises(RuntimeError, match="test failure"):
        M.build_source_mesh(_sim(), ["0", "0", "-.01"], dict(screens=[],
            record_cache_directory=str(tmp_path), record_cache_capacity=1))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [{"record_cache_directory": ""}, {"record_cache_directory": False},
    {"record_cache_capacity": 0}, {"record_cache_capacity": True}, {"record_cache_capacity": 1.5}])
def test_driver_rejects_invalid_cache_configuration(options):
    from formula.capsysred.stages._s19_config import validate
    with pytest.raises(ValueError, match="record_cache"):
        validate(SimpleNamespace(raw={"stage19": {"mesh": options}}))


def test_enabled_storage_helper_is_in_the_checkpoint_code_contract(tmp_path):
    from formula.capsysred.stages.stage19 import _code_hashes
    hashes = _code_hashes({"mesh": {"record_cache_directory": str(tmp_path)}, "kernel": {}})
    assert "_s19_cache.py" in hashes and len(hashes["_s19_cache.py"]) == 64
