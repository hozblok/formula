"""Pure-stdlib SVG rendering: line charts, PNG-backed heatmaps, setup schemes.

No numpy/matplotlib: PNG via zlib+struct, everything else is SVG text. A figure
is {"w", "h", "body"}; hstack composes figures, save() writes the file.
"""

import base64
import math
import struct
import zlib
from xml.sax.saxutils import escape

FONT = 'font-family="DejaVu Sans, Arial, sans-serif"'
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

# Normative Stage-14 Okabe-Ito taxonomy palette.  ``None`` is deliberately
# absent: an inapplicable classifier is rendered as a grey checkerboard, not
# as a tenth scientific class.
FLAG_COLORS = {
    "trusted": "#d9d9d9",
    "solo-rays-only": "#009E73",
    "negative-Ic": "#0072B2",
    "null-Ic": "#56B4E9",
    "noisy-Ic": "#E69F00",
    "no-ref-realizations": "#F0E442",
    "noisy-mu": "#CC79A7",
    "over-mu": "#000000",
    "no-rays": "#ffffff",
}

_VIRIDIS = [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
            (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
            (253, 231, 37)]


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e5:
        s = f"{v:.6g}"
    else:
        s = f"{v:.3g}"
    return s


def viridis(v: float):
    v = min(1.0, max(0.0, v))
    x = v * (len(_VIRIDIS) - 1)
    i = min(int(x), len(_VIRIDIS) - 2)
    f = x - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    return tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3))


def _png_bytes(rows) -> bytes:
    """Truecolor PNG from rows of (r, g, b) tuples."""
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _png_uri(rows) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(rows)).decode()


def nice_ticks(lo: float, hi: float, n: int = 5):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + step * 1e-9:
        out.append(0.0 if abs(t) < step * 1e-9 else t)
        t += step
    return out


def _text(x, y, s, size=13, anchor="start", color="#222", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}" {extra}>{escape(str(s))}</text>')


def _line(x0, y0, x1, y1, color="#222", width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')


def _rect(x, y, w, h, fill, stroke="none", extra=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" {extra}/>')


class _Axes:
    """Data->pixel mapping plus frame, ticks and labels."""

    def __init__(self, w, h, x_range, y_range, left=70, right=16, top=46, bottom=56):
        self.px0, self.px1 = left, w - right
        self.py0, self.py1 = h - bottom, top
        (self.xa, self.xb), (self.ya, self.yb) = x_range, y_range

    def x(self, v):
        return self.px0 + (v - self.xa) / (self.xb - self.xa) * (self.px1 - self.px0)

    def y(self, v):
        return self.py0 + (v - self.ya) / (self.yb - self.ya) * (self.py1 - self.py0)

    def frame(self, xlabel, ylabel, title, subtitle=""):
        e = [_rect(self.px0, self.py1, self.px1 - self.px0, self.py0 - self.py1,
                   "white", "#888")]
        for t in nice_ticks(self.xa, self.xb):
            px = self.x(t)
            e.append(_line(px, self.py0, px, self.py1, "#eee"))
            e.append(_line(px, self.py0, px, self.py0 + 4, "#555"))
            e.append(_text(px, self.py0 + 17, _fmt(t), 11, "middle", "#444"))
        for t in nice_ticks(self.ya, self.yb):
            py = self.y(t)
            e.append(_line(self.px0, py, self.px1, py, "#eee"))
            e.append(_line(self.px0 - 4, py, self.px0, py, "#555"))
            e.append(_text(self.px0 - 7, py + 4, _fmt(t), 11, "end", "#444"))
        cx = (self.px0 + self.px1) / 2
        e.append(_text(cx, self.py0 + 36, xlabel, 13, "middle"))
        ly = (self.py0 + self.py1) / 2
        e.append(_text(0, 0, ylabel, 13, "middle",
                       extra=f'transform="translate(16,{ly:.0f}) rotate(-90)"'))
        e.append(_text(self.px0, 20, title, 15, "start", "#111",
                       'font-weight="bold"'))
        if subtitle:
            e.append(_text(self.px0, 37, subtitle, 11.5, "start", "#666"))
        return e


def _ranges(series, y_zero: bool):
    xs = [v for s in series for v in s["xs"]]
    ys = [v for s in series for v in s["ys"]]
    ys += [v for s in series for key in ("lo", "hi") for v in (s.get(key) or ())]
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    if xb <= xa:
        pad_x = abs(xa) * 0.05 or 1.0
        xa, xb = xa - pad_x, xb + pad_x
    if y_zero:
        ya = min(0.0, ya)
    pad = (yb - ya) * 0.06 or 1.0
    return (xa, xb), (ya, yb + pad)


def line_chart(series, title, xlabel, ylabel, subtitle="", vlines=(),
               y_zero=True, w=560, h=400):
    """series: [{xs, ys, label, color?, dash?, width?, lo?, hi?, dots?}];
    lo/hi: shaded band; dots: circle markers instead of a line."""
    ax = _Axes(w, h, *_ranges(series, y_zero))
    e = ax.frame(xlabel, ylabel, title, subtitle)
    for x, label in vlines:
        px = ax.x(x)
        e.append(_line(px, ax.py0, px, ax.py1, "#999", 1.0, "5,4"))
        if label:
            e.append(_text(px + 3, ax.py1 + 12, label, 10.5, "start", "#777"))
    for i, s in enumerate(series):
        color = s.get("color") or PALETTE[i % len(PALETTE)]
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        if s.get("lo") and s.get("hi"):
            band = " ".join(f"{ax.x(x):.1f},{ax.y(y):.1f}" for x, y in
                            list(zip(s["xs"], s["hi"]))
                            + list(zip(reversed(s["xs"]), reversed(s["lo"]))))
            e.append(f'<polygon points="{band}" fill="{color}" '
                     f'fill-opacity="0.16" stroke="none"/>')
        if s.get("dots"):
            e.extend(f'<circle cx="{ax.x(x):.1f}" cy="{ax.y(y):.1f}" r="3" '
                     f'fill="{color}"/>' for x, y in zip(s["xs"], s["ys"]))
            continue
        pts = " ".join(f"{ax.x(x):.1f},{ax.y(y):.1f}"
                       for x, y in zip(s["xs"], s["ys"]))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="{s.get("width", 1.8)}"{dash}/>')
    ly = ax.py1 + 14
    lx = ax.px1 - 10
    for i, s in enumerate(reversed(series)):
        if not s.get("label"):
            continue
        color = s.get("color") or PALETTE[(len(series) - 1 - i) % len(PALETTE)]
        e.append(_text(lx, ly, s["label"], 11.5, "end", "#333"))
        e.append(_line(lx - len(s["label"]) * 6.6 - 26, ly - 4, lx - len(s["label"]) * 6.6 - 6,
                       ly - 4, color, 2.2, s.get("dash")))
        ly += 15
    return {"w": w, "h": h, "body": "".join(e)}


CBAR_GUTTER = 74      # heatmap colorbar margin
LEGEND_GUTTER = 190   # category_map legend margin


def _mark_pixel(ax, mark, nx, ny):
    """Reference pixel as one filled data cell."""
    w = (ax.px1 - ax.px0) / nx
    h = (ax.py0 - ax.py1) / ny
    return _rect(ax.x(mark[0]) - w / 2, ax.y(mark[1]) - h / 2, w, h, "#d62728")


def heatmap(grid, extent, title, xlabel, ylabel, subtitle="", cbar_label="",
            w=560, h=460, vmax=None, mark=None, equal=False):
    """grid: row-major [iy][ix], iy=0 at the bottom edge; extent=(x0,x1,y0,y1).
    equal=True: pick h so one data unit spans equal px on both axes."""
    ny, nx = len(grid), len(grid[0])
    finite = [float(v) for row in grid for v in row
              if v is not None and math.isfinite(float(v))]
    vmax = vmax or max(finite, default=1.0) or 1.0
    scale = max(1, int(360 / max(nx, ny)))
    rows = []
    for iy in range(ny - 1, -1, -1):
        row = []
        for ix in range(nx):
            value = grid[iy][ix]
            color = (((224, 224, 224) if (ix + iy) % 2 else (184, 184, 184))
                     if value is None else viridis(float(value) / vmax))
            row.extend([color] * scale)
        rows.extend([row] * scale)
    ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=CBAR_GUTTER)
    if equal:
        # size h so one data unit spans equal px on both axes
        span = (ax.px1 - ax.px0) * (extent[3] - extent[2]) / (extent[1] - extent[0])
        h = round(h - (ax.py0 - ax.py1) + span)
        ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=CBAR_GUTTER)
    e = ax.frame(xlabel, ylabel, title, subtitle)
    e.append(f'<image x="{ax.px0:.1f}" y="{ax.py1:.1f}" '
             f'width="{ax.px1 - ax.px0:.1f}" height="{ax.py0 - ax.py1:.1f}" '
             f'preserveAspectRatio="none" image-rendering="pixelated" '
             f'href="{_png_uri(rows)}"/>')
    if mark:
        e.append(_mark_pixel(ax, mark, nx, ny))
    cb = [[viridis(1.0 - j / 255.0)] * 12 for j in range(256)]
    cx = ax.px1 + 14
    e.append(f'<image x="{cx}" y="{ax.py1:.1f}" width="14" '
             f'height="{ax.py0 - ax.py1:.1f}" preserveAspectRatio="none" '
             f'href="{_png_uri(cb)}"/>')
    e.append(_rect(cx, ax.py1, 14, ax.py0 - ax.py1, "none", "#888"))
    for frac in (0.0, 0.5, 1.0):
        y = ax.py0 - frac * (ax.py0 - ax.py1)
        e.append(_text(cx + 18, y + 4, _fmt(vmax * frac), 10.5, "start", "#444"))
    if cbar_label:
        e.append(_text(cx + 7, ax.py0 + 36, cbar_label, 11.5, "middle", "#333"))
    return {"w": w, "h": h, "body": "".join(e)}


def ray_scatter(counts, extent, title, xlabel, ylabel, subtitle="", w=640):
    """Ray-location image: white background, blue density; counts row-major
    [iy][ix], iy=0 at the bottom edge; always equal aspect."""
    vmax = max((v for row in counts for v in row), default=1) or 1
    blue = (34, 34, 187)
    rows = []
    for iy in range(len(counts) - 1, -1, -1):
        row = []
        for v in counts[iy]:
            t = (v / vmax) ** 0.4 if v else 0.0
            row.append(tuple(round(255 + (c - 255) * t) for c in blue))
        rows.append(row)
    ax = _Axes(w, 460, (extent[0], extent[1]), (extent[2], extent[3]))
    span = (ax.px1 - ax.px0) * (extent[3] - extent[2]) / (extent[1] - extent[0])
    h = round(460 - (ax.py0 - ax.py1) + span)
    ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]))
    e = ax.frame(xlabel, ylabel, title, subtitle)
    e.append(f'<image x="{ax.px0:.1f}" y="{ax.py1:.1f}" '
             f'width="{ax.px1 - ax.px0:.1f}" height="{ax.py0 - ax.py1:.1f}" '
             f'preserveAspectRatio="none" href="{_png_uri(rows)}"/>')
    return {"w": w, "h": h, "body": "".join(e)}


def vstack(figs, gap=8):
    w = max(f["w"] for f in figs)
    body, y = [], 0
    for f in figs:
        body.append(f'<g transform="translate(0,{y})">{f["body"]}</g>')
        y += f["h"] + gap
    return {"w": w, "h": y - gap, "body": "".join(body)}


def hstack(figs, gap=12):
    w = sum(f["w"] for f in figs) + gap * (len(figs) - 1)
    h = max(f["h"] for f in figs)
    body, x = [], 0
    for f in figs:
        body.append(f'<g transform="translate({x},0)">{f["body"]}</g>')
        x += f["w"] + gap
    return {"w": w, "h": h, "body": "".join(body)}


def save(path, fig):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{fig["w"]}" height="{fig["h"]}" '
           f'viewBox="0 0 {fig["w"]} {fig["h"]}">'
           f'<rect width="100%" height="100%" fill="white"/>'
           f'{fig["body"]}</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)


# ---------------------------------------------------------------- schemes

def _arrow_h(e, x0, x1, y, label, size=11.5):
    e.append(_line(x0, y, x1, y, "#333"))
    for x, s in ((x0, 1), (x1, -1)):
        e.append(f'<path d="M {x:.1f} {y:.1f} l {6 * s} -3.4 l 0 6.8 z" fill="#333"/>')
    e.append(_text((x0 + x1) / 2, y - 6, label, size, "middle", "#222"))


def _arrow_v(e, x, y0, y1, label, size=11.5, side=1):
    e.append(_line(x, y0, x, y1, "#333"))
    for y, s in ((y0, 1), (y1, -1)):
        e.append(f'<path d="M {x:.1f} {y:.1f} l -3.4 {6 * s} l 6.8 0 z" fill="#333"/>')
    e.append(_text(x + 8 * side, (y0 + y1) / 2 + 4, label, size,
                   "start" if side > 0 else "end", "#222"))


def _description(e, x, y, lines):
    for i, line in enumerate(lines):
        e.append(_text(x, y + i * 18, line, 12.5, "start", "#333"))


def scheme_setup(info):
    """Not-to-scale project scheme: source -> capillary bundle -> screen."""
    w, h = 980, 400 + 18 * len(info["description"])
    ay = 210
    xs, xc0, xc1, xscr = 150, 380, 660, 840
    e = [_text(90, 30, info["title"], 17, "start", "#111", 'font-weight="bold"'),
         _text(880, 30, "not to scale", 11, "end", "#999"),
         _line(90, ay, 910, ay, "#bbb", 1, "7,5")]
    # source with a small ray fan
    for dy in (-26, 0, 26):
        e.append(_line(xs, ay, xc0 - 6, ay + dy, "#f2b100", 1))
    e.append(f'<circle cx="{xs}" cy="{ay}" r="7" fill="#e67e22"/>')
    _description(e, xs - 55, ay - 56, info["source_label"])
    # capillary walls (bore exaggerated)
    bore = 20
    for y0 in (ay - bore - 14, ay + bore):
        e.append(_rect(xc0, y0, xc1 - xc0, 14, "#b6c8d8", "#5a7a94"))
    if info.get("n_bores", 1) > 1:
        for y0 in (ay - 2 * bore - 34, ay + 2 * bore + 6):
            e.append(_rect(xc0, y0, xc1 - xc0, 10, "#dbe6ee", "#9db4c6"))
    _arrow_v(e, xc0 - 18, ay - bore, ay + bore, info["bore_label"], side=-1)
    _description(e, xc0 + 10, ay - bore - 30, [info["capillary_title"]])
    # screen
    e.append(_rect(xscr, ay - 110, 5, 220, "#444"))
    _description(e, xscr - 40, ay - 130, info["screen_label"])
    _arrow_v(e, xscr + 26, ay - 60, ay + 60, info["window_label"])
    # distances
    _arrow_h(e, xs, xc0, ay + 84, info["d0_label"])
    _arrow_h(e, xc0, xc1, ay + 84, info["len_label"])
    _arrow_h(e, xc1, xscr, ay + 84, info["d2_label"])
    _description(e, 90, ay + 140, info["description"])
    return {"w": w, "h": h, "body": "".join(e)}


def _hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def category_map(grid, extent, title, xlabel, ylabel, subtitle="", mark=None,
                 counts=None, lit_counts=None, lit_total=None,
                 w=620, h=460, equal=False):
    """Categorical Stage-14 flag map.

    ``grid`` contains normative flag strings or ``None``.  Null is rendered
    with a checkerboard and listed separately; legend counts are normally
    supplied by the classifier so its denominator can remain ``n_rays > 0``.
    """
    ny, nx = len(grid), len(grid[0])
    scale = max(1, int(360 / max(nx, ny)))
    # w means the same plot width as heatmap's w; the wider legend adds on top
    w += LEGEND_GUTTER - CBAR_GUTTER
    rows = []
    null_a, null_b = (238, 238, 238), (184, 184, 184)
    for iy in range(ny - 1, -1, -1):
        row = []
        for ix in range(nx):
            flag = grid[iy][ix]
            color = (_hex_rgb(FLAG_COLORS[flag]) if flag is not None
                     else (null_a if (ix + iy) % 2 == 0 else null_b))
            row.extend([color] * scale)
        rows.extend([row] * scale)
    ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=LEGEND_GUTTER)
    if equal:
        span = (ax.px1 - ax.px0) * (extent[3] - extent[2]) / (extent[1] - extent[0])
        h = round(h - (ax.py0 - ax.py1) + span)
        ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=LEGEND_GUTTER)
    e = ax.frame(xlabel, ylabel, title, subtitle)
    e.append(f'<image x="{ax.px0:.1f}" y="{ax.py1:.1f}" '
             f'width="{ax.px1 - ax.px0:.1f}" height="{ax.py0 - ax.py1:.1f}" '
             f'preserveAspectRatio="none" image-rendering="pixelated" '
             f'href="{_png_uri(rows)}"/>')
    if mark:
        e.append(_mark_pixel(ax, mark, nx, ny))
    observed = counts or {}
    lit_observed = observed if lit_counts is None else lit_counts
    total = (lit_total if lit_total is not None else
             sum(observed.get(name, 0) for name in FLAG_COLORS
                 if name != "no-rays") + observed.get(None, 0))
    lx, ly = ax.px1 + 16, ax.py1 + 4
    order = tuple(FLAG_COLORS) + (None,)
    for name in order:
        n = observed.get(name, 0)
        fill = FLAG_COLORS.get(name, "#b8b8b8")
        label = "unclassified" if name is None else name
        pct_n = lit_observed.get(name, 0)
        pct = 100.0 * pct_n / total if total and name != "no-rays" else 0.0
        e.append(_rect(lx, ly, 12, 12, fill, "#777"))
        suffix = f" {n:,}" + (f" ({pct:.1f}%)" if pct else "")
        e.append(_text(lx + 18, ly + 11, label + suffix, 10.5, "start", "#333"))
        ly += 18
    return {"w": w, "h": h, "body": "".join(e)}


def overlay_map(mu_grid, flag_grid, extent, title, xlabel, ylabel,
                subtitle="", mark=None, w=620, h=460, equal=False):
    """Viridis raw-mu display with every classified non-trusted target pink."""
    ny, nx = len(mu_grid), len(mu_grid[0])
    scale = max(1, int(360 / max(nx, ny)))
    rows, bad = [], 0
    for iy in range(ny - 1, -1, -1):
        row = []
        for ix in range(nx):
            flag, value = flag_grid[iy][ix], mu_grid[iy][ix]
            if flag not in (None, "trusted", "no-rays"):
                color, bad = _hex_rgb("#CC79A7"), bad + 1
            elif flag == "no-rays":
                color = (255, 255, 255)
            elif value is None:
                color = (224, 224, 224) if (ix + iy) % 2 else (184, 184, 184)
            else:
                color = viridis(min(float(value), 1.0))
            row.extend([color] * scale)
        rows.extend([row] * scale)
    ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=26)
    if equal:
        span = (ax.px1 - ax.px0) * (extent[3] - extent[2]) / (extent[1] - extent[0])
        h = round(h - (ax.py0 - ax.py1) + span)
        ax = _Axes(w, h, (extent[0], extent[1]), (extent[2], extent[3]), right=26)
    e = ax.frame(xlabel, ylabel, title, subtitle)
    e.append(f'<image x="{ax.px0:.1f}" y="{ax.py1:.1f}" '
             f'width="{ax.px1 - ax.px0:.1f}" height="{ax.py0 - ax.py1:.1f}" '
             f'preserveAspectRatio="none" image-rendering="pixelated" '
             f'href="{_png_uri(rows)}"/>')
    if mark:
        e.append(f'<circle cx="{ax.x(mark[0]):.1f}" cy="{ax.y(mark[1]):.1f}" r="5" '
                 f'fill="none" stroke="#d62728" stroke-width="1.8"/>')
    e.append(_rect(ax.px0 + 8, ax.py0 - 25, 12, 12, "#CC79A7", "#555"))
    e.append(_text(ax.px0 + 26, ax.py0 - 14,
                   f"classified but not trusted ({bad:,})", 10.5))
    return {"w": w, "h": h, "body": "".join(e)}


GAUGE_OK, GAUGE_WARN = "#009E73", "#E69F00"


def gauge_table(checks, title, header="", footer="", w=770):
    """Check rows: name + detail line, big value, gauge with a threshold.

    checks: [{name, detail, big, value, lo, hi, threshold, threshold_label}];
    value None = unavailable (no marker, warn tone)."""
    top, row_h, bottom = 96, 92, 34
    h = top + row_h * len(checks) + bottom
    x_check, x_big, x_gauge, gauge_w = 14, 400, 545, 205
    e = [_text(x_check, 24, title, 15, "start", "#111", 'font-weight="bold"')]
    if header:
        e.append(_text(x_check, 46, header, 11, "start", "#666"))
    for x, label in ((x_check, "check"), (x_big, "value"),
                     (x_gauge, "gauge · threshold")):
        e.append(_text(x, 72, label, 11, "start", "#666", 'font-weight="bold"'))
    e.append(_line(0, 80, w, 80, "#bbb", 0.8))
    for i, c in enumerate(checks):
        y = top + row_h * i + row_h / 2
        value = c["value"]
        good = value is not None and value >= c["threshold"]
        tone = GAUGE_OK if good else GAUGE_WARN
        e.append(_text(x_check, y - 6, c["name"], 13.5))
        e.append(_text(x_check, y + 14, c["detail"], 11.5, color="#444"))
        e.append(_text(x_big, y + 9, c["big"], 26, "start", tone, 'font-weight="bold"'))
        gh, lo, hi = 20, c["lo"], c["hi"]
        ft = (c["threshold"] - lo) / (hi - lo)
        e.append(_rect(x_gauge, y - gh / 2, gauge_w * ft, gh, GAUGE_WARN,
                       extra='fill-opacity="0.13"'))
        e.append(_rect(x_gauge + gauge_w * ft, y - gh / 2, gauge_w * (1 - ft), gh,
                       GAUGE_OK, extra='fill-opacity="0.13"'))
        e.append(_rect(x_gauge, y - gh / 2, gauge_w, gh, "none", "#999"))
        xt = x_gauge + gauge_w * ft
        e.append(_line(xt, y - gh / 2 - 4, xt, y + gh / 2 + 4, "#333", 1.2, "4,3"))
        e.append(_text(x_gauge, y + gh / 2 + 12, _fmt(lo), 9, "middle", "#888"))
        e.append(_text(x_gauge + gauge_w, y + gh / 2 + 12, _fmt(hi), 9, "middle", "#888"))
        e.append(_text(xt, y - gh / 2 - 7, c["threshold_label"], 9.5, "middle", "#333"))
        if value is not None:
            xv = x_gauge + gauge_w * (min(max(value, lo), hi) - lo) / (hi - lo)
            e.append(f'<circle cx="{xv:.1f}" cy="{y:.1f}" r="7" fill="{tone}" '
                     f'stroke="#222" stroke-width="1"/>')
    if footer:
        e.append(_text(x_check, h - 12, footer, 10, "start", "#666"))
    return {"w": w, "h": h, "body": "".join(e)}
