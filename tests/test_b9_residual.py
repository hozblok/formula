"""Stratified residual quadrature of unresolved curved faces and its variance accounting."""
import numpy as np
import pytest

from formula.capsysred.config import Config
from formula.capsysred.stages._b9_curved import curved_residual_fields
from formula.capsysred.stages.stage18 import _residual_correction
from tests.test_b9_curved_mesh import build
from tests.test_stage18_curved_integration import curved_scene


def direct(points, coefficients, x, y, width, k, distance, order=4):
    """Reference Fresnel sum of weighted points, coherently averaged over each square cell."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    kappa = k/distance
    field = np.zeros((len(y), len(x)), complex)
    for a, wa in zip(nodes, weights/2):
        for b, wb in zip(nodes, weights/2):
            px, py = np.meshgrid(x+a*width/2, y+b*width/2)
            delta2 = (px[..., None]-points[:, 0])**2+(py[..., None]-points[:, 1])**2
            field += wa*wb*np.sum(coefficients*np.exp(.5j*kappa*delta2), axis=-1)
    return kappa/(2j*np.pi)*field


def synthetic_residual(batches=3, faces=7, seed=5):
    rng = np.random.default_rng(seed)
    area = rng.uniform(1e-13, 5e-13, faces)
    points = rng.uniform(-5e-6, 5e-6, (batches, faces, 2))
    phase = rng.uniform(-np.pi, np.pi, (batches, faces))
    density = rng.normal(size=(batches, faces))+1j*rng.normal(size=(batches, faces))
    valid = rng.random((batches, faces)) > .2
    density[~valid] = 0
    return dict(residual_batches=batches, residual_area=area, residual_points=points,
                residual_phase=phase, residual_density=density, residual_valid=valid)


def test_residual_fields_match_direct_fresnel_sum():
    pytest.importorskip("finufft")
    mesh = synthetic_residual()
    x, y = np.linspace(-3e-6, 3e-6, 5), np.linspace(-2e-6, 2e-6, 4)
    args = dict(k=4e10, distance=.35, x=x, y=y, cell_width=.3e-6, pixel_order=4, eps=1e-12)
    fields, stats = curved_residual_fields(mesh, return_stats=True, **args)
    assert fields.shape == (3, 4, 5) and stats["faces"] == 7 and stats["zero_weight_samples"] == int(np.sum(~mesh["residual_valid"]))
    for b in range(3):
        coefficients = mesh["residual_area"]*mesh["residual_density"][b]*np.exp(1j*mesh["residual_phase"][b])
        expected = direct(mesh["residual_points"][b], coefficients, x, y, .3e-6, 4e10, .35)
        np.testing.assert_allclose(fields[b], expected, rtol=1e-8, atol=1e-8*abs(expected).max())


def test_residual_fields_without_batches_or_faces_are_zero():
    pytest.importorskip("finufft")
    args = dict(k=4e10, distance=.35, x=np.linspace(-1e-6, 1e-6, 3), y=np.linspace(-1e-6, 1e-6, 3), cell_width=.3e-6)
    assert curved_residual_fields({}, **args).shape == (0, 3, 3)
    empty = synthetic_residual(batches=2, faces=0)
    np.testing.assert_array_equal(curved_residual_fields(empty, **args), 0)
    bad = synthetic_residual()
    bad["residual_points"] = bad["residual_points"][:, :3]
    with pytest.raises(ValueError, match="residual arrays"):
        curved_residual_fields(bad, **args)


def test_variance_accounting_removes_the_sampling_bias():
    rng = np.random.default_rng(11)
    base = rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4))
    delta = .5*(rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4)))
    reference, sigma, batches, trials = (1, 2), .8, 4, 4000
    raw_i = raw_w = fixed_i = fixed_w = 0
    for _ in range(trials):
        noise = sigma*(rng.normal(size=(batches, 3, 4))+1j*rng.normal(size=(batches, 3, 4)))/np.sqrt(2)
        correction = _residual_correction(delta+noise, reference)
        field = base+correction["mean"]
        raw_i = raw_i+abs(field)**2/trials
        raw_w = raw_w+field*field[reference].conjugate()/trials
        fixed_i = fixed_i+(abs(field)**2-correction["variance"])/trials
        fixed_w = fixed_w+(field*field[reference].conjugate()-correction["reference_covariance"])/trials
    truth = base+delta
    tolerance = 6*sigma**2/np.sqrt(batches*trials)+1e-12
    assert np.all(abs(raw_i-abs(truth)**2-sigma**2/batches) < tolerance)          # the raw estimate is biased by var/B
    assert np.all(abs(fixed_i-abs(truth)**2) < tolerance)
    assert np.all(abs(fixed_w-truth*truth[reference].conjugate()) < tolerance)
    assert abs(raw_w[reference]-abs(truth[reference])**2-sigma**2/batches) < tolerance


def test_mesh_residual_samples_cover_the_unresolved_faces():
    mesh, d = build(residual_batches=2, residual_seed=3)
    unresolved = ~mesh["partition_accepted"]
    n = int(unresolved.sum())
    assert n > 0 and mesh["residual_batches"] == 2
    assert mesh["residual_points"].shape == (2, n, 2) and mesh["residual_density"].shape == (2, n)
    np.testing.assert_array_equal(mesh["residual_entrance_triangles"], mesh["partition_entrance_triangles"][unresolved])
    np.testing.assert_allclose(mesh["residual_area"].sum(), d["coverage"]["unresolved_area_m2"], rtol=1e-12)
    assert np.all(mesh["residual_density"][~mesh["residual_valid"]] == 0)
    assert np.any(mesh["residual_valid"]) and np.isfinite(mesh["residual_density"]).all()
    residual = d["residual"]
    assert residual["batches"] == 2 and residual["faces"] == n and residual["emitted"] == 2*n
    assert sum(residual["fate_counts"].values()) == 2*n
    # samples lie inside their faces
    tri = mesh["residual_entrance_triangles"]
    for b in range(2):
        q = np.einsum("ni,nij->nj", np.array([[1/3, 1/3, 1/3]]*n), tri)        # centroid as an inside check anchor
        assert np.all(np.linalg.norm(mesh["residual_points"][b]-q, axis=1) < 1e-3)   # exit positions differ from entrance by < 1 mm
    plain, _ = build()
    assert plain.get("residual_batches", 0) == 0 and "residual_points" not in plain
    with pytest.raises(ValueError, match="residual_batches"):
        build(residual_batches=1)


@pytest.mark.parametrize("bad", [{"residual_batches": 1}, {"residual_batches": -1}, {"residual_batches": 2.5},
                                 {"residual_seed": -3}, {"residual_batches": True}])
def test_residual_option_validation(bad):
    raw = curved_scene()
    raw["b9_estimator"]["curved_retrace"].update(bad)
    with pytest.raises(ValueError, match="b9_estimator.curved_retrace"):
        Config(raw).validate_b9_estimator()
    raw = curved_scene()
    raw["b9_estimator"]["curved_retrace"].update(residual_batches=2, residual_seed=7)
    assert Config(raw).validate_b9_estimator()["curved_retrace"]["residual_batches"] == 2
