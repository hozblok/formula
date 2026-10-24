"""Independent physical/analytic controls for the NumPy-free Stage 19 kernel."""
import array
import cmath
import math

import pytest

native = pytest.importorskip("formula._stage19")


def record(vertices, *, amplitude=1+0j, anchor=(0., 0.), gradient=(0., 0.), phase=0.):
    return [*(v for q in vertices for v in q), *anchor, *gradient,
            amplitude.real, amplitude.imag, phase]


def complex_values(raw):
    values = array.array("d")
    values.frombytes(raw)
    return [complex(values[i], values[i+1]) for i in range(0, len(values), 2)]


def real_values(raw):
    values = array.array("d")
    values.frombytes(raw)
    return list(values)


def field_bytes(values):
    return array.array("d", [part for z in values for part in (z.real, z.imag)]).tobytes()


def gauss(order):
    """Independent 1-D Gauss rule for a 2-D surface reference (no library)."""
    result = []
    for i in range(order):
        x = math.cos(math.pi*(i+.75)/(order+.5))
        for _ in range(40):
            a, b = 1., x
            for j in range(2, order+1):
                a, b = b, ((2*j-1)*x*b-(j-1)*a)/j
            derivative = order*(x*b-a)/(x*x-1)
            delta = b/derivative
            x -= delta
            if abs(delta) < 2e-16:
                break
        result.append(((x+1)/2, 1/((1-x*x)*derivative*derivative)))
    return result


def surface_reference(row, point, k, distance, order=96):
    """Direct Duffy surface integration, independent of the contour identity."""
    v0, v1, v2 = row[0:2], row[2:4], row[4:6]
    a, b = [v1[i]-v0[i] for i in range(2)], [v2[i]-v0[i] for i in range(2)]
    determinant = abs(a[0]*b[1]-a[1]*b[0])
    rule = gauss(order)
    values = []
    for u, wu in rule:
        for v, wv in rule:
            q = [v0[j]+u*a[j]+(1-u)*v*b[j] for j in range(2)]
            phi = row[12]+sum(row[8+j]*(q[j]-row[6+j]) for j in range(2))
            phi += k/(2*distance)*sum((q[j]-point[j])**2 for j in range(2))
            values.append(wu*wv*(1-u)*cmath.exp(1j*phi))
    integral = complex(math.fsum(z.real for z in values), math.fsum(z.imag for z in values))
    return complex(row[10], row[11])*determinant*k/(2j*math.pi*distance)*integral


@pytest.mark.parametrize("point,k,gradient", [
    ((.3, .2), 3., (0., 0.)),       # stationary point inside
    ((0., 0.), 3., (0., 0.)),       # stationary point at a vertex
    ((.5, 0.), 3., (0., 0.)),       # stationary point on an edge
    ((18., -7.), 3., (1., -.5)),    # stationary point outside
    ((.1, .2), 180., (4., -2.)),    # many phase turns
    ((.3, .2), 1e-6, (3., -2.)),   # near-linear phase; distant stationary point
])
def test_exact_quadratic_phase_contour_vs_independent_surface(point, k, gradient):
    row = record([(0., 0.), (1., 0.), (.1, .8)], amplitude=.7+.4j,
                 gradient=gradient, phase=.3)
    actual = native.eval_contour([row], k, 1., [point[0]], [point[1]], order=16)
    value = complex_values(actual["field"])[0]
    expected = surface_reference(row, point, k, 1.)
    assert value == pytest.approx(expected, rel=4e-11, abs=2e-13)
    assert actual["diagnostics"]["common_longitudinal_phase_omitted"] is True


def square_records(amplitude=1+0j, gradient=(0., 0.), phase=0.):
    opts = dict(amplitude=amplitude, gradient=gradient, phase=phase)
    return [record([(0., 0.), (1., 0.), (0., 1.)], **opts),
            record([(1., 1.), (0., 1.), (1., 0.)], **opts)]


@pytest.mark.parametrize("gradient", [(0., 0.), (1e-8, -2e-8), (.2, -.3), (25., -13.)])
def test_exit_square_mean_is_analytic_sinc_and_full_area(gradient):
    gx, gy = gradient
    amplitude, phase = .7+.4j, .3
    result = native.eval_exit(square_records(amplitude, gradient, phase), [.5], [.5], 1.)
    sx = math.sin(gx/2)/(gx/2) if gx else 1.
    sy = math.sin(gy/2)/(gy/2) if gy else 1.
    expected = amplitude*cmath.exp(1j*(phase+gx/2+gy/2))*sx*sy
    assert complex_values(result["field"])[0] == pytest.approx(expected, abs=5e-14)
    # Half a square lies outside the field; normalization remains the full pixel.
    partial = native.eval_exit(square_records(), [0.], [.5], 1.)
    assert complex_values(partial["field"])[0] == pytest.approx(.5+0j, abs=2e-15)


def test_exit_shared_edge_has_one_owner_and_physical_branches_add():
    rows = square_records()
    value = complex_values(native.eval_exit(rows, [.5], [.5])["field"])[0]
    assert value == 1+0j
    # Distinct overlapping branches carry unequal complex coefficients.
    rows += square_records(.25+.5j, (1.2, -.7), .4)
    expected = 1+(.25+.5j)*cmath.exp(1j*(.4+.6-.35))
    assert complex_values(native.eval_exit(rows, [.5], [.5])["field"])[0] == pytest.approx(expected)


def test_orientation_is_geometric_not_an_extra_phase():
    row = record([(0., 0.), (1., 0.), (.1, .8)], gradient=(2., -1.))
    reversed_row = row[:2]+row[4:6]+row[2:4]+row[6:]
    for method in (lambda r: native.eval_contour([r], 4., 1., [.2], [.1]),
                   lambda r: native.eval_exit([r], [.2], [.1], .3)):
        assert complex_values(method(row)["field"]) == pytest.approx(complex_values(method(reversed_row)["field"]))


def test_cell_propagation_matches_independent_tensor_target_quadrature():
    rows = square_records(.3+.7j, (2., -3.), .2)
    x, y, width = .2, .3, .15
    mean = complex_values(native.eval_contour(rows, 5., 1., [x], [y], width,
                                             order=16, pixel_order=8)["field"])[0]
    expected = 0j
    for u, wu in gauss(10):
        for v, wv in gauss(10):
            p = (x+width*(u-.5), y+width*(v-.5))
            expected += wu*wv*sum(surface_reference(row, p, 5., 1., order=20) for row in rows)
    assert mean == pytest.approx(expected, abs=2e-13)


def test_threads_produce_identical_results_and_invalid_input_rejected():
    args = (square_records(), 4., 1., [0., .1, .2], [0., .1])
    assert native.eval_contour(*args, nthreads=1)["field"] == native.eval_contour(*args, nthreads=3)["field"]
    with pytest.raises(ValueError):
        native.eval_contour(square_records(), 4., 0., [0.], [0.])
    with pytest.raises(ValueError):
        native.eval_exit([record([(0., 0.), (1., 0.), (2., 0.)])], [0.], [0.])
    with pytest.raises(RuntimeError, match="max_subdivisions"):
        native.eval_contour(square_records(), 500., 1., [0.], [0.], max_subdivisions=1)


def test_coherence_complex_moments_and_resume():
    fields = [[1+0j, 2j, 0j], [2j, .25-.5j, 0j], [.2+.4j, -1+.7j, 0j]]
    accumulator = native.CoherenceAccumulator(3, 0)
    accumulator.add(field_bytes(fields[0]))
    accumulator = native.CoherenceAccumulator.from_state(accumulator.state())
    for field in fields[1:]:
        accumulator.add(field_bytes(field))
    result = accumulator.result()
    intensity = [sum(abs(field[j])**2 for field in fields)/3 for j in range(3)]
    w = [sum(field[j]*field[0].conjugate() for field in fields)/3 for j in range(3)]
    assert real_values(result["I"]) == pytest.approx(intensity)
    assert complex_values(result["W"]) == pytest.approx(w)
    assert result["n_modes"] == 3
    assert result["reference_intensity"] == pytest.approx(intensity[0])
    assert list(result["valid"]) == [1, 1, 0]
    mu = complex_values(result["mu"])
    assert mu[0] == pytest.approx(1)
    assert mu[1] == pytest.approx(w[1]/math.sqrt(intensity[1]*intensity[0]))
    assert math.isnan(mu[2].real)
    with pytest.raises(ValueError):
        accumulator.add(field_bytes([complex(float("nan")), 0j, 0j]))
    assert accumulator.result()["n_modes"] == 3


def test_fast_source_quadrature_matches_contour_before_any_nufft():
    """Separate source quadrature error from the fast transform implementation."""
    rows = [record([(0., 0.), (1e-6, 0.), (0., 1e-6)],
                   amplitude=.7+.4j, gradient=(3e6, -1e6), phase=.3),
            record([(10e-6, 20e-6), (10e-6, 21e-6), (11e-6, 20e-6)],
                   amplitude=-.2+.8j, anchor=(10e-6, 20e-6),
                   gradient=(-2e6, 4e6), phase=-.7)]
    k, distance = 40785096803.62667, .35
    kappa = k/distance
    source = native.quadrature_sources(rows, kappa, -160e-6, 160e-6,
                                      -160e-6, 160e-6, minimum=8, safety=1.)
    xs, ys = real_values(source["x"]), real_values(source["y"])
    cs = complex_values(source["coefficients"])
    for x, y in [(0., 0.), (40e-6, -20e-6), (-160e-6, 160e-6), (160e-6, -160e-6)]:
        terms = [c*cmath.exp(-1j*kappa*(x*qx+y*qy)) for qx, qy, c in zip(xs, ys, cs)]
        integral = complex(math.fsum(z.real for z in terms), math.fsum(z.imag for z in terms))
        actual = kappa/(2j*math.pi)*cmath.exp(.5j*kappa*(x*x+y*y))*integral
        expected = complex_values(native.eval_contour(rows, k, distance, [x], [y], order=16)["field"])[0]
        assert actual == pytest.approx(expected, rel=2e-10, abs=1e-13)
