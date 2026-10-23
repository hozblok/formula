"""Stage 16 dependencies stay optional for the tracer and other stages.

Use fresh interpreters: the numerical wave tests import NumPy/SciPy during
collection, which would otherwise conceal an eager dependency import.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
_WAVE = "formula.capsysred.stages.wave"
_PRELUDE = """
import copy
from pathlib import Path
import sys

attempts = []
class ImportBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.split('.')[0] in blocked:
            attempts.append(fullname)
            raise ModuleNotFoundError('blocked dependency: ' + fullname,
                                      name=fullname)
sys.meta_path.insert(0, ImportBlocker())

raw = {
    'precision': 32, 'seed': 271828, 'energy_kev': 8.048,
    'capillary': {
        'z0': 0.0, 'z1': 0.05,
        'bores': [{'center': [0.0, 0.0], 'radius': 6e-6}],
        'source': {'shape': 'disk', 'size': 3e-7,
                   'position': [0.0, 0.0, -0.01],
                   'n_modes': 3, 'n_rays': 24},
        'screen': {'z': 0.06, 'edge_x': 24e-6, 'edge_y': 24e-6,
                   'nx': 3, 'ny': 3, 'reference': [0.0, 0.0]},
    },
}
"""


def _fresh_python(tmp_path, body, blocked=()):
    script = (f"blocked = {set(blocked)!r}\n" + textwrap.dedent(_PRELUDE)
              + textwrap.dedent(body))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1",
               CAPSYSRED_STAGE14_JOBS="1", CAPSYSRED_STAGE11_JOBS="1")
    env.pop("CAPSYSRED_PYTHON_TRACE", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_trace_and_stage14_do_not_import_wave_dependencies(tmp_path):
    _fresh_python(tmp_path, """
        import yaml
        from formula.capsysred import Simulation, rays_v3
        from formula.capsysred.trace_v3 import trace

        variants = {
            'absent': None,
            'valid': {'provider': 'uisk', 'grid_dx': 1e-7},
            'invalid': {'provider': 'unused-invalid-provider', 'grid_dx': -1},
        }
        outputs = []
        for name, section in variants.items():
            config = copy.deepcopy(raw)
            if section is not None:
                config['wave_estimator'] = section
            config_path = Path(name + '.yaml')
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            out = Path(name)
            archive = out / 'rays-modes'
            trace(str(config_path), str(archive), jobs=1, level=1,
                  log=lambda _: None, scenes=('capillary',))
            index = rays_v3.load_index(str(archive))
            trace_rows = b''.join(rays_v3.scene_lines(str(archive), index,
                                                    'capillary'))
            assert len(trace_rows.splitlines()) == 72
            sim = Simulation.from_yaml(str(config_path))
            sim.run(str(out), stages=[14])
            result = out / 'stage14' / 'mu-jack.jsonl'
            assert result.is_file() and result.stat().st_size > 0
            outputs.append((trace_rows, result.read_bytes()))
            if section is None:
                assert 'wave_estimator' not in sim.cfg.raw
            else:
                assert sim.cfg.raw['wave_estimator'] == section
        assert outputs[0] == outputs[1] == outputs[2]
        assert attempts == [], attempts
        assert 'formula.capsysred.stages.wave' not in sys.modules
        assert not any(n.split('.')[0] in {'numpy', 'scipy'} for n in sys.modules)
    """, blocked=(_WAVE, "numpy", "scipy"))


@pytest.mark.parametrize("missing", ["numpy", "scipy"])
def test_selected_wave_fails_before_reading_or_creating_output(tmp_path, missing):
    if missing == "scipy" and importlib.util.find_spec("numpy") is None:
        pytest.skip("the SciPy-only failure case needs NumPy to be installed")
    _fresh_python(tmp_path, f"""
        import formula.capsysred.simulation as simulation
        from formula.capsysred import Simulation

        def unexpected(*args, **kwargs):
            raise AssertionError('another stage or archive reader ran')
        simulation.run_stage14 = unexpected
        simulation.RaysReader = unexpected
        simulation.MultiRaysReader = unexpected
        simulation.Simulation._stage11 = unexpected
        # Fail on a missing backend dependency, before geometry validation too.
        # The cylinder is intentionally unsupported by Stage 16.
        for method, stages in [('run', [16]), ('run', [14, 16]),
                               ('replay', [16]), ('replay', [11, 14, 16])]:
            sim = Simulation.from_dict(copy.deepcopy(raw))
            output = Path(method + '-' + '-'.join(map(str, stages)))
            try:
                if method == 'run':
                    sim.run(str(output), stages=stages)
                else:
                    sim.replay('must-not-be-read', str(output), stages=stages)
            except ValueError as exc:
                assert 'Stage 16' in str(exc), str(exc)
                assert 'NumPy and SciPy' in str(exc), str(exc)
                assert 'missing dependency: {missing}' in str(exc), str(exc)
                assert isinstance(exc.__cause__, ModuleNotFoundError)
                assert exc.__cause__.name == '{missing}'
            else:
                raise AssertionError('Stage 16 accepted a missing dependency')
            assert not output.exists()
        assert attempts and all(n == '{missing}' for n in attempts), attempts
    """, blocked=(missing,))


def test_wave_import_does_not_mask_unrelated_backend_errors(tmp_path):
    _fresh_python(tmp_path, """
        import formula.capsysred.simulation as simulation
        from formula.capsysred import Simulation

        class BrokenWave:
            error = None
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'formula.capsysred.stages.wave':
                    raise self.error
        broken = BrokenWave()
        sys.meta_path.insert(0, broken)
        errors = [ModuleNotFoundError('unrelated internal import',
                                      name='unrelated_internal_module'),
                  RuntimeError('backend initialization defect')]
        for i, error in enumerate(errors):
            broken.error = error
            output = Path('broken-' + str(i))
            try:
                Simulation.from_dict(copy.deepcopy(raw)).run(str(output), stages=[16])
            except Exception as exc:
                assert exc is error, (type(exc), str(exc))
            else:
                raise AssertionError('backend initialization error was ignored')
            assert not output.exists()
    """)
