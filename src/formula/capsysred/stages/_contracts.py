"""Dependency-free screen and optical-physics contracts."""


def _screen_spec(scr) -> dict:
    reference = getattr(scr, "reference", None)
    return {"z": float(scr.z), "nx": int(scr.nx), "ny": int(scr.ny),
            "center": [float(scr.center[0]), float(scr.center[1])],
            "edge_x": float(scr.edge_x), "edge_y": float(scr.edge_y),
            "reference": None if not reference else
            [float(reference[0]), float(reference[1])]}


def _physics_contract(sim) -> dict:
    return {
        "energy_kev": str(sim.cfg.energy_kev),
        "lines": [{"energy_kev": str(line.e_kev), "k": float(line.k),
                   "weight": float(line.weight)} for line in sim.lines],
        "material": sim.cfg.material.name,
        # exact optical-constant strings entering the amplitudes (E0 and per line)
        "two_delta_beta_e0": [str(sim.fresnel.d2), str(sim.fresnel.b2)],
        "two_delta_beta_lines": [[dd, bb] for dd, bb in sim.line_amps.two_delta_beta],
        "per_line_fresnel": bool(sim.per_line),
        "amplitude_min": float(sim.cfg.amplitude_min),
        "precision": int(sim.cfg.precision),
        "deposit": "float-line-amplitudes-phase-exp-v1",
    }
