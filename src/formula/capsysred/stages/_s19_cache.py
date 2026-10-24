"""Lossless, per-bore cold storage for the Stage 19 trace cache.

Only locally generated payloads are unpickled. Stable record wrappers preserve
aliases held by accepted models; input-coordinate keys and cumulative IDs stay
in memory. This bounds decoded trace payloads, not the whole adaptive mesh.
"""

from collections import OrderedDict
from collections.abc import MutableMapping
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryFile
from types import SimpleNamespace
import pickle
import weakref
import zlib

from ...backend import mp_class
from ...formula import Number


def _immutable(value):
    if value is None or isinstance(value, (str, int, float, bool, bytes, Decimal)):
        return True
    return isinstance(value, tuple) and all(_immutable(item) for item in value)


def _encode_number(value):
    """Preserve backend guard digits, which Number.parts deliberately omits."""
    if not isinstance(value, Number) or value.is_complex:
        raise RuntimeError("Cold incidence cache requires a real Formula Number")
    text = value._value.str(0)
    try:
        if not Decimal(text).is_finite():
            raise RuntimeError("Cold incidence cache cannot encode nonfinite values")
        restored = Number(mp_class(value.precision)(text))
    except (ArithmeticError, ValueError) as exc:
        raise RuntimeError("Cannot encode the native incidence value exactly") from exc
    # Number.__eq__ compares displayed parts; compare the native backend itself.
    if (restored.precision != value.precision or restored.is_complex
            or restored._value != value._value or restored._value.str(0) != text):
        raise RuntimeError("Native incidence codec lost backend digits")
    return value.precision, text


def _decode_number(encoded):
    precision, text = encoded
    result = Number(mp_class(precision)(text))
    if result.precision != precision or result.is_complex or result._value.str(0) != text:
        raise RuntimeError("Native incidence cache failed its roundtrip check")
    return result


class TraceStore:
    """Append-only temporary file and a bounded decoded-record LRU."""

    def __init__(self, directory, capacity=4096):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("record_cache_capacity must be a positive integer")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.file = TemporaryFile(mode="w+b", prefix="stage19-records-", suffix=".bin", dir=directory)
        self.capacity, self.hot, self.orders = capacity, OrderedDict(), {}
        self.created = self.loads = self.writes = self.hits = self.evictions = 0
        self.bytes_written = self.peak_hot = self.pending_created = 0

    def close(self):
        self.hot.clear()
        self.file.close()

    def order(self, keys):
        value = tuple(keys)
        return self.orders.setdefault(value, value)

    def write(self, values):
        try:
            payload = zlib.compress(pickle.dumps(values, protocol=5), 1)
            self.file.seek(0, 2)
            offset = self.file.tell()
            written = self.file.write(payload)
            if written != len(payload):
                raise OSError("Short write in Stage 19 cold trace cache")
        except OSError:
            raise
        except Exception as exc:
            raise RuntimeError("Cannot write Stage 19 cold trace cache") from exc
        self.writes += 1
        self.bytes_written += written
        return offset, written

    def read(self, location):
        try:
            offset, length = location
            self.file.seek(offset)
            payload = self.file.read(length)
            if len(payload) != length:
                raise RuntimeError("Truncated Stage 19 cold trace cache")
            return pickle.loads(zlib.decompress(payload))
        except OSError:
            raise
        except Exception as exc:
            # _TraceCache catches ValueError as a physical transport failure.
            # Storage failures must stop the run instead of changing its mesh.
            raise RuntimeError("Cannot decode Stage 19 cold trace cache") from exc

    def _trim(self):
        self.peak_hot = max(self.peak_hot, len(self.hot))
        while len(self.hot) > self.capacity:
            _, (reference, values, dirty) = self.hot.popitem(last=False)
            record = reference()
            if record is not None and dirty:
                record.location = self.write(values)
            self.evictions += 1

    def values(self, record, *, dirty=False):
        if record.ident in self.hot:
            self.hits += 1
            entry = self.hot[record.ident]
            self.hot.move_to_end(record.ident)
            entry[2] |= dirty
            return entry[1]
        self.loads += 1
        values = self.read(record.location)
        self.hot[record.ident] = [weakref.ref(record), values, dirty]
        self._trim()
        return values

    def create(self, **values):
        record = TraceRecord(self, values)
        self.created += 1
        return record

    def pending(self, record, reflections):
        encoded = [(tuple(point), _encode_number(sine)) for point, sine in reflections]
        self.pending_created += 1
        return PendingRay(self, record, self.write(encoded))

    def statistics(self):
        return dict(backend="temporary_lossless_disk", capacity=self.capacity,
                    created=self.created, loads=self.loads, writes=self.writes,
                    hits=self.hits, evictions=self.evictions, hot=len(self.hot),
                    peak_hot=self.peak_hot, bytes_written=self.bytes_written,
                    pending_created=self.pending_created,
                    native_codec="cpp_dec_float.str(0), backend-equality checked on write")


class TraceRecord(MutableMapping):
    """An identity-stable mutable record; cold scalar values remain exact."""

    __slots__ = ("store", "ident", "q", "location", "pinned", "keys_order", "__weakref__")

    def __init__(self, store, values):
        self.store, self.ident, self.q = store, values["id"], values["q"]
        self.location, self.pinned = None, None
        self.keys_order = store.order(values)
        data = {}
        for key, value in values.items():
            if key not in ("id", "q"):
                if _immutable(value):
                    data[key] = value
                else:
                    self._pin(key, value)
        store.hot[self.ident] = [weakref.ref(self), data, True]
        store._trim()

    def _pin(self, key, value):
        if not isinstance(value, (list, dict, set)):
            raise RuntimeError("Unsupported native or mutable cache record value")
        if self.pinned is None:
            self.pinned = {}
        self.pinned[key] = value

    def __getitem__(self, key):
        if key == "id":
            return self.ident
        if key == "q":
            return self.q
        if self.pinned is not None and key in self.pinned:
            return self.pinned[key]
        return self.store.values(self)[key]

    def __setitem__(self, key, value):
        if key in ("id", "q"):
            if value != self[key]:
                raise ValueError("Record identity cannot change")
            return
        data = self.store.values(self, dirty=True)
        if _immutable(value):
            data[key] = value
            if self.pinned is not None:
                self.pinned.pop(key, None)
        else:
            self._pin(key, value)
            data.pop(key, None)
        if key not in self.keys_order:
            self.keys_order = self.store.order((*self.keys_order, key))

    def __delitem__(self, key):
        if key in ("id", "q"):
            raise ValueError("Record identity cannot change")
        if self.pinned is not None and key in self.pinned:
            del self.pinned[key]
        else:
            del self.store.values(self, dirty=True)[key]
        self.keys_order = self.store.order(k for k in self.keys_order if k != key)

    def __iter__(self):
        return iter(self.keys_order)

    def __len__(self):
        return len(self.keys_order)


class PendingRay:
    """Materialize one exact ray when its delayed transport is requested."""

    __slots__ = ("store", "record", "location")

    def __init__(self, store, record, location):
        self.store, self.record, self.location = store, record, location

    def materialize(self):
        try:
            reflections = [(point, _decode_number(encoded)) for point, encoded in self.store.read(self.location)]
        except OSError:
            raise
        except Exception as exc:
            raise RuntimeError("Cannot restore the pending native incidence ray") from exc
        return SimpleNamespace(point=self.record["point"], direction=self.record["direction"],
                               reflections=reflections)
