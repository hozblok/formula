"""Shared archive fixtures."""

import pytest


@pytest.fixture
def write_v3_archive():
    from formula.capsysred import rays_v3
    from formula.capsysred.config import load
    from formula.capsysred.rays import RNG_SCHEME, geometry_metadata

    def write(archive, cfg_raw, rows_by_scene_mode, lean=False):
        """Write prepared rows keyed by scene then mode; return the v3 index."""
        meta = {
            "format": rays_v3.FORMAT,
            "geometry": geometry_metadata(load(cfg_raw)),
            "rng": {"scheme": RNG_SCHEME},
        }
        if lean:
            meta["lean"] = True
        rays_v3.write_fingerprint(archive, meta)
        entries = []
        for scene, modes in rows_by_scene_mode.items():
            for mode, rows in modes.items():
                writer = rays_v3.SectionWriter(archive, scene, mode, 0, len(rows))
                try:
                    for row in rows:
                        writer.write_row(row)
                    entries.append(writer.close())
                except BaseException:
                    writer.abort()
                    raise
        return rays_v3.write_index(archive, entries)

    return write
