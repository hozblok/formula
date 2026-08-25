"""rays v3: one gzip section file per (scene, mode, ray range) plus an index.

    ARCHIVE/
      rays-fingerprint.yaml   format 3: geometry (with seed), lean, rng scheme
      rays-index.jsonl        one entry per section file
      modes/<scene>-m<mode:06>-r<r0:08>-<r1:08>.jsonl.gz

A section file holds a header line ``{"header": 3, scene, mode, r0, r1,
origin}``, the v2 rows of rays r0..r1-1 with absolute mode/ray ids, and a
trailer ``{"scene_end", "rows", "mode", "r0", "r1"}``.  The index is the
authority for budgets: modes 0..M-1 all present, sections contiguous from
ray 0, one common ray count per scene.  Readers hash every section while
streaming and compare with the index; a section is only ever published by
tmp -> rename and the index by atomic replace.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import time
import zlib
from contextlib import suppress
from typing import NamedTuple

import yaml

from .shared.types import RayRecord

FORMAT = 3
METADATA_NAME = "rays-fingerprint.yaml"
INDEX_NAME = "rays-index.jsonl"
MODES_DIR = "modes"
DEFAULT_LEVEL = 6
SHA256_HEX = 64
READ_ATTEMPTS = 5               # transient device errors: reopen and resume
READ_BACKOFF_S = 10.0


class Section(NamedTuple):
    scene: str
    mode: int
    r0: int
    r1: int
    rows: int
    bytes: int
    sha256: str
    file: str

    def as_dict(self) -> dict:
        return dict(self._asdict())


def is_v3(path) -> bool:
    return os.path.isdir(os.fspath(path))


def fingerprint_path(archive) -> str:
    return os.path.join(os.fspath(archive), METADATA_NAME)


def index_path(archive) -> str:
    return os.path.join(os.fspath(archive), INDEX_NAME)


def section_name(scene: str, mode: int, r0: int, r1: int) -> str:
    return f"{scene}-m{mode:06d}-r{r0:08d}-{r1:08d}.jsonl.gz"


def section_path(archive, entry: Section) -> str:
    return os.path.join(os.fspath(archive), MODES_DIR, entry.file)


# ----------------------------------------------------------------- metadata

def read_fingerprint(archive) -> dict:
    path = fingerprint_path(archive)
    try:
        with open(path, encoding="utf-8") as fh:
            meta = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: unreadable v3 rays fingerprint") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ValueError(f"{path}: rays fingerprint format must be {FORMAT}")
    if not isinstance(meta.get("geometry"), dict):
        raise ValueError(f"{path}: rays geometry metadata must be a mapping")
    if "lean" in meta and meta["lean"] is not True:
        raise ValueError(f"{path}: lean must be true when present")
    if "budgets" in meta:
        raise ValueError(f"{path}: v3 budgets live in the index, not the fingerprint")
    return meta


def write_fingerprint(archive, meta: dict) -> str:
    """No-clobber: an identical fingerprint is accepted, a different one refused."""
    if meta.get("format") != FORMAT:
        raise ValueError("v3 fingerprint must carry format 3")
    path = fingerprint_path(archive)
    canonical = json.dumps(meta, sort_keys=True, default=str)
    if os.path.lexists(path):
        existing = read_fingerprint(archive)
        if json.dumps(existing, sort_keys=True, default=str) == canonical:
            return path
        raise ValueError(f"{path}: fingerprint exists and differs; remove it manually")
    os.makedirs(os.fspath(archive), exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False, allow_unicode=True)
        fh.flush()
        os.fsync(fh.fileno())
    return path


# -------------------------------------------------------------------- index

class Index:
    """Validated section index: budgets and per-mode section lists."""

    def __init__(self, entries: list[Section]):
        self.entries = sorted(entries, key=lambda e: (e.scene, e.mode, e.r0))
        seen = set()
        for entry in self.entries:
            key = (entry.scene, entry.mode, entry.r0)
            if key in seen:
                raise ValueError(f"rays index repeats section {entry.file}")
            seen.add(key)
            if entry.file != section_name(entry.scene, entry.mode, entry.r0, entry.r1):
                raise ValueError(f"rays index entry has a non-canonical file name {entry.file}")
        self._modes: dict[str, list[list[Section]]] = {}
        for entry in self.entries:
            per_scene = self._modes.setdefault(entry.scene, [])
            if entry.mode == len(per_scene):
                per_scene.append([entry])
            elif entry.mode == len(per_scene) - 1:
                per_scene[-1].append(entry)
            else:
                raise ValueError(
                    f"rays index scene {entry.scene!r}: modes are not contiguous at {entry.mode}")
        self.budgets: dict[str, list[int]] = {}
        for scene, modes in self._modes.items():
            n = None
            for sections in modes:
                expected = 0
                for entry in sections:
                    if entry.r0 != expected or entry.r1 <= entry.r0 or entry.rows != entry.r1 - entry.r0:
                        raise ValueError(
                            f"rays index scene {scene!r} mode {entry.mode}: sections are not "
                            f"contiguous from ray 0 ({entry.file})")
                    expected = entry.r1
                if n is None:
                    n = expected
                elif n != expected:
                    raise ValueError(
                        f"rays index scene {scene!r}: modes have unequal ray counts "
                        f"({n} vs {expected} at mode {sections[0].mode})")
            self.budgets[scene] = [len(modes), n]

    def scenes(self):
        return list(self._modes)

    def modes(self, scene: str) -> list[list[Section]]:
        return self._modes.get(scene, [])

    def sections(self, scene: str, mode: int) -> list[Section]:
        return self._modes[scene][mode]

    def total_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries)


def _parse_entry(row: dict, path: str) -> Section:
    try:
        entry = Section(str(row["scene"]), int(row["mode"]), int(row["r0"]),
                        int(row["r1"]), int(row["rows"]), int(row["bytes"]),
                        str(row["sha256"]), str(row["file"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid rays index entry {row!r}") from exc
    if entry.mode < 0 or entry.r0 < 0 or len(entry.sha256) != SHA256_HEX:
        raise ValueError(f"{path}: invalid rays index entry {row!r}")
    return entry


def load_index(archive) -> Index:
    path = index_path(archive)
    entries = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                entries.append(_parse_entry(row, path))
    except OSError as exc:
        raise ValueError(f"{path}: rays index is missing or unreadable") from exc
    if not entries:
        raise ValueError(f"{path}: rays index is empty")
    return Index(entries)


def write_index(archive, entries) -> Index:
    """Atomically replace the index with the validated, sorted entries."""
    index = Index(list(entries))
    path = index_path(archive)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for entry in index.entries:
            fh.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return index


def index_digest(archive) -> str:
    with open(index_path(archive), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def metadata(archive) -> dict:
    """Fingerprint plus index-derived budgets: the v2-shaped meta consumers expect."""
    meta = dict(read_fingerprint(archive))
    index = load_index(archive)
    meta["budgets"] = {scene: list(budget) for scene, budget in index.budgets.items()}
    return meta


# ------------------------------------------------------------------ writing

class SectionWriter:
    """Streams one section into modes/<name>.tmp, publishes it by rename."""

    def __init__(self, archive, scene: str, mode: int, r0: int, r1: int,
                 origin=None, header_extra: dict | None = None,
                 level: int = DEFAULT_LEVEL):
        if r1 <= r0 or r0 < 0 or mode < 0:
            raise ValueError("section ray range must be non-empty and non-negative")
        self.archive = os.fspath(archive)
        self.entry_args = (scene, mode, r0, r1)
        self.name = section_name(scene, mode, r0, r1)
        self.final = os.path.join(self.archive, MODES_DIR, self.name)
        self.tmp = self.final + ".tmp"
        os.makedirs(os.path.dirname(self.final), exist_ok=True)
        if os.path.lexists(self.final):
            raise ValueError(f"{self.final}: section already exists")
        header = {"header": FORMAT, "scene": scene, "mode": mode,
                  "r0": r0, "r1": r1, "origin": origin}
        if header_extra:
            header.update(header_extra)
        self._raw = open(self.tmp, "wb")
        try:
            self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw,
                                     compresslevel=level, mtime=0)
            self._gz.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
        except BaseException:
            self._raw.close()
            with suppress(FileNotFoundError):
                os.remove(self.tmp)
            raise
        self.rows = 0
        self._closed = False

    def write_line(self, line: bytes) -> None:
        """One complete v2 row line (ends with a newline)."""
        self._gz.write(line)
        self.rows += 1

    def write_lines(self, lines: bytes, rows: int) -> None:
        self._gz.write(lines)
        self.rows += rows

    def write_row(self, row: dict) -> None:
        self.write_line(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            with suppress(OSError, ValueError):
                self._gz.close()
            with suppress(OSError, ValueError):
                self._raw.close()
            with suppress(FileNotFoundError):
                os.remove(self.tmp)

    def close(self) -> Section:
        if self._closed:
            raise ValueError("section writer already closed")
        scene, mode, r0, r1 = self.entry_args
        if self.rows != r1 - r0:
            self.abort()
            raise ValueError(
                f"{self.name}: wrote {self.rows} rows, section promises {r1 - r0}")
        trailer = {"scene_end": scene, "rows": self.rows, "mode": mode, "r0": r0, "r1": r1}
        self._gz.write(json.dumps(trailer).encode("utf-8") + b"\n")
        self._gz.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        size, digest = _hash_file(self.tmp)
        os.replace(self.tmp, self.final)
        self._closed = True
        return Section(scene, mode, r0, r1, self.rows, size, digest, self.name)


def _hash_file(path: str) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


# ------------------------------------------------------------------ reading

class _HashingRaw(io.RawIOBase):
    """Hashes what it reads; an OSError reopens the file and resumes at the
    same offset, so a flaky device never corrupts the stream."""

    def __init__(self, path):
        self.path = path
        self.fh = None
        self.h = hashlib.sha256()
        self.n = 0

    def readable(self):
        return True

    def readinto(self, b):
        n = 0
        for attempt in range(1, READ_ATTEMPTS + 1):
            try:
                if self.fh is None:
                    self.fh = open(self.path, "rb")
                    self.fh.seek(self.n)
                n = self.fh.readinto(b)
                break
            except FileNotFoundError:
                raise
            except OSError:
                self.close_file()
                if attempt == READ_ATTEMPTS:
                    raise
                time.sleep(READ_BACKOFF_S * attempt)
        if n:
            self.h.update(memoryview(b)[:n])
            self.n += n
        return n

    def close_file(self):
        if self.fh is not None:
            with suppress(OSError):
                self.fh.close()
            self.fh = None

    def close(self):
        self.close_file()
        super().close()


def _check_header(row, entry: Section, path: str) -> dict:
    if (not isinstance(row, dict) or row.get("header") != FORMAT
            or row.get("scene") != entry.scene or row.get("mode") != entry.mode
            or row.get("r0") != entry.r0 or row.get("r1") != entry.r1):
        raise ValueError(f"{path}: section header contradicts the index")
    return row


def section_header(archive, entry: Section) -> dict:
    """Header only (origin etc.); does not verify the section body."""
    path = section_path(archive, entry)
    with gzip.open(path, "rb") as fh:
        line = fh.readline()
    try:
        row = json.loads(line)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid section header") from exc
    return _check_header(row, entry, path)


def iter_section_lines(archive, entry: Section, sink: dict | None = None):
    """Yield the raw row lines of one section, verifying header, trailer,
    row count and the file's sha256 against the index. ``sink['header']``
    receives the parsed header when a dict is given."""
    path = section_path(archive, entry)
    try:
        yield from _section_lines(path, entry, sink)
    except (EOFError, zlib.error, OSError) as exc:
        raise ValueError(f"{path}: section is truncated or corrupt; remove it "
                         "manually (verify the archive)") from exc


def _section_lines(path, entry: Section, sink):
    with _HashingRaw(path) as hashing:
        buffered = io.BufferedReader(hashing, 1 << 20)
        with gzip.GzipFile(fileobj=buffered, mode="rb") as gz:
            first = gz.readline()
            try:
                header = json.loads(first)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid section header") from exc
            _check_header(header, entry, path)
            if sink is not None:
                sink["header"] = header
            rows = 0
            trailer = None
            for line in gz:
                if not line.endswith(b"\n"):
                    raise ValueError(f"{path}: unterminated line")
                if line.startswith(b'{"scene_end"'):
                    trailer = json.loads(line)
                    if gz.read(1):
                        raise ValueError(f"{path}: data after the section trailer")
                    break
                rows += 1
                yield line
            if trailer is None:
                raise ValueError(f"{path}: missing section trailer")
            if (trailer.get("scene_end") != entry.scene or trailer.get("rows") != rows
                    or trailer.get("mode") != entry.mode or rows != entry.rows):
                raise ValueError(f"{path}: section trailer contradicts the index")
        while buffered.read(1 << 20):
            pass
    if hashing.n != entry.bytes or hashing.h.hexdigest() != entry.sha256:
        raise ValueError(f"{path}: section bytes/sha256 differ from the index")


def _mode_of(line: bytes) -> int:
    i = line.find(b'"mode": ') + 8
    return int(line[i:line.index(b",", i)])


def _ray_of(line: bytes) -> int:
    i = line.find(b'"ray": ') + 7
    return int(line[i:line.index(b",", i)])


def verify_section(archive, entry: Section) -> dict:
    """Full read-back of one section: hashes, counts, canonical mode/ray ids."""
    sink = {}
    ray = entry.r0
    prefix = ('{"stage": ' + json.dumps(entry.scene) + ", ").encode("utf-8")
    for line in iter_section_lines(archive, entry, sink):
        if not line.startswith(prefix) or _mode_of(line) != entry.mode or _ray_of(line) != ray:
            raise ValueError(
                f"{entry.file}: row {ray - entry.r0} is not mode {entry.mode} ray {ray}")
        ray += 1
    return sink["header"]


def scene_lines(archive, index: Index, scene: str):
    """Rows of a scene, mode-major, sections in ray order (raw bytes)."""
    for sections in index.modes(scene):
        for entry in sections:
            yield from iter_section_lines(archive, entry)


def _record(row: dict) -> RayRecord:
    point = direction = None
    if "x" in row:
        point = (row["x"], row["y"], float("nan"))
        dx, dy = row["dx"], row["dy"]
        direction = (dx, dy, math.sqrt(max(1.0 - dx * dx - dy * dy, 0.0)))
    return RayRecord(row["mode"], row["ray"], row["fate"], row["pixel"],
                     point, direction, row["opl"], tuple(row["sins"]),
                     tuple(tuple(p) for p in row.get("refl", ())))


def scene_records(archive, index: Index, scene: str):
    for line in scene_lines(archive, index, scene):
        yield _record(json.loads(line))


def origins(archive, index: Index, scene: str) -> list:
    """Per-mode origin from the first section header (None when unrecorded)."""
    return [section_header(archive, sections[0]).get("origin")
            for sections in index.modes(scene)]


__all__ = [
    "FORMAT", "METADATA_NAME", "INDEX_NAME", "MODES_DIR", "Section", "Index",
    "SectionWriter", "is_v3", "fingerprint_path", "index_path", "section_name",
    "section_path", "read_fingerprint", "write_fingerprint", "load_index",
    "write_index", "index_digest", "metadata", "iter_section_lines",
    "section_header", "verify_section", "scene_lines", "scene_records", "origins",
]
