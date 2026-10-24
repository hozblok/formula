"""Native accelerated integral against the independent contour implementation."""

from array import array
import math
import pytest

native = pytest.importorskip("formula._stage19")
from formula.capsysred.stages._s19_accelerated import evaluate_field
from formula.capsysred.stages.stage19 import _grid


def values(raw):
    parts = array("d"); parts.frombytes(raw)
    return [complex(parts[i], parts[i+1]) for i in range(0,len(parts),2)]


@pytest.mark.parametrize("nx,ny,observable", [
    (7,5,"point"), (6,4,"point"), (5,5,"coherent_cell"),
])
def test_native_acceleration_preserves_contour_integral(nx,ny,observable):
    records = [
        [0.,0., 1.,0., .1,.8, .2,.3, 4.,-2., .7,.4, .3],
        [.1,.1, .1,.8, .9,.3, .2,.3, -3.,7., -.4,.2, 1.7],
    ]
    spec = dict(nx=nx,ny=ny,center=[".15","-.1"],edge_x=str(nx*.03),
                edge_y=str(ny*.03),reference=[".15","-.1"],distance="1")
    opts = dict(source_order=10,source_safety=1.,pixel_order=6,nthreads=1,
                contour_check=True,integration_tolerance=2e-8,nufft_eps=1e-12)
    result = evaluate_field(records,180.,spec,observable,opts)
    xs,ys,width,_ = _grid(spec)
    reference = native.eval_contour(records,180.,1.,xs,ys,
        pixel_width=width if observable=="coherent_cell" else 0.,order=16,pixel_order=8)
    a,b=values(result["field"]),values(reference["field"])
    error=math.sqrt(sum(abs(x-y)**2 for x,y in zip(a,b))/sum(abs(y)**2 for y in b))
    assert error < 2e-8


def test_longitudinal_phase_preserves_moments_for_unequal_source_fields():
    fields = [[1+.7j,.4-.6j], [.3-.4j,-1.2+.8j], [2+.1j,.7+.2j]]
    a=native.CoherenceAccumulator(2,0)
    b=native.CoherenceAccumulator(2,0)
    for i,row in enumerate(fields):
        z=complex(math.cos(.4+i),math.sin(.4+i))
        for acc,multiplier in ((a,1.),(b,z)):
            acc.add(array("d",[v for f in row for v in ((f*multiplier).real,(f*multiplier).imag)]))
    ra,rb=a.result(),b.result()
    for key in ("I","W","mu"):
        left=array("d");left.frombytes(ra[key])
        right=array("d");right.frombytes(rb[key])
        assert list(left) == pytest.approx(list(right),abs=1e-14)
