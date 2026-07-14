"""To-scale schematic (SVG) of a configured scene with real traced rays.

Side view in the meridional plane (x–z): source, optic, screen and a handful
of rays (config `schematic: n_rays`) traced through the actual geometry. Transverse features (µm) and axial
distances (cm) differ by ~10^3-10^4, so x and z carry independent linear
scales — both get a scale bar and every length is dimensioned. Multi-bore /
faceted / bent optics also get a transverse (x–y) inset; a single in-plane
bent bore gets a bore-relative (unrolled) panel.

Used by stage 1 (01a-scheme-traced.svg) and the exp/schematic.py CLI.
"""

import math
import random
from xml.sax.saxutils import escape

from .nums import lift, vunit
from .surfaces import CapillaryBundle, Mirror, entrance_disk
from .trace import trace_ray

UM, MM = 1e6, 1e3
N_RAYS = 10
GREEN, BLUE, WALL, AXIS = "#2ca02c", "#3060c0", "#5a7a94", "#bbbbbb"
INK, DIM = "#222", "#333"


# ---------------------------------------------------------------- SVG helpers
def esc(s):
    return escape(str(s))


def T(x, y, s, size=12, anchor="start", color=INK, weight=None, rot=None):
    w = f' font-weight="{weight}"' if weight else ""
    tr = f' transform="rotate({rot[0]} {rot[1]:.1f} {rot[2]:.1f})"' if rot else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, Arial, '
            f'sans-serif" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{color}"{w}{tr}>{esc(s)}</text>')


def L(x0, y0, x1, y1, color=INK, w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="{color}" stroke-width="{w}"{d}/>')


def RECT(x, y, w, h, fill="none", stroke="none", sw=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def POLY(pts, color, w=1.0, fill="none", dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    s = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return (f'<polyline points="{s}" fill="{fill}" stroke="{color}" '
            f'stroke-width="{w}"{d} opacity="{opacity}"/>')


def CIRC(x, y, r, fill="none", stroke="none", sw=1.0):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def hsv(i, n):
    h = 0.82 * i / max(1, n - 1)
    r, g, b = [max(0.0, min(1.0, abs((h * 6 + k) % 6 - 3) - 1)) for k in (0, 4, 2)]
    return f"#{int(r*205):02x}{int(g*205):02x}{int(b*205):02x}"


def eng(v, unit="auto"):
    """Length with a readable unit."""
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e-3 * (1 - 1e-9):                      # epsilon: dodge float-subtraction noise
        return f"{v*MM:.4g} mm"
    if a >= 1e-7 * (1 - 1e-9):
        return f"{v*UM:.4g} µm"
    return f"{v*1e9:.4g} nm"


# ---------------------------------------------------------------- dimensions
def arrow_h(x0, x1, y, label, size=11):
    e = [L(x0, y, x1, y, DIM)]
    for x, s in ((x0, 1), (x1, -1)):
        e.append(f'<path d="M {x:.1f} {y:.1f} l {6*s} -3.4 l 0 6.8 z" fill="{DIM}"/>')
    e.append(T((x0 + x1) / 2, y - 5, label, size, "middle", DIM))
    return e


def arrow_v(x, y0, y1, label, size=11, side=1):
    e = [L(x, y0, x, y1, DIM)]
    for y, s in ((y0, 1), (y1, -1)):
        e.append(f'<path d="M {x:.1f} {y:.1f} l -3.4 {6*s} l 6.8 0 z" fill="{DIM}"/>')
    e.append(T(x + 7 * side, (y0 + y1) / 2 + 4, label, size,
              "start" if side > 0 else "end", DIM))
    return e


# ---------------------------------------------------------------- geometry
def r_of_z(bore, zf):
    """Wall radius r(z) of an axisymmetric bore (metres)."""
    kind = bore.get("kind", "cylinder")
    if kind == "revolution":
        c0, c1, c2 = (float(v) for v in bore["r2_poly"])
        return math.sqrt(max(c0 + zf * (c1 + zf * c2), 0.0))
    if kind == "polygon":
        return float(bore["radius"])            # apothem: y=0 faces sit here
    return float(bore["radius"])


def torus_axis(bore, z0f, zf):
    """In-plane bent-axis x(z) for a torus bore (metres); assumes toward in x."""
    R = float(bore["bend"]["radius"])
    ux = float(bore["bend"]["toward"][0])
    cx = float(bore["center"][0])
    phi = math.asin(max(-1.0, min(1.0, (zf - z0f) / R)))
    return cx + R * (1.0 - math.cos(phi)) * ux


def build_geometry(cfg, mode: str, bores=None):
    """G dict for compose(): the scene of `mode` with real rays traced through.

    bores overrides cfg.capillary.bores (typed twin of an implicit config)."""
    p = cfg.precision
    G = {"mode": mode, "cfg": cfg}
    if mode == "lloyd":
        o = cfg.lloyd
        src, scr = o.source, o.screen
        G.update(z0=float(o.z0), z1=float(o.z1), height=float(o.height),
                 optic=Mirror(o.z0, o.z1), bores=None)
    elif mode == "free":
        src, scr = cfg.free_source, cfg.free_screen
        G.update(z0=None, z1=None, optic=None, bores=None)
    else:
        o = cfg.capillary
        src, scr = o.source, o.screen
        bores = bores or o.bores
        G.update(z0=float(o.z0), z1=float(o.z1),
                 optic=CapillaryBundle(bores, o.z0, o.z1, cfg.engine_method),
                 bores=bores)
    G["src"] = {"z": float(src.position[2]), "x": float(src.position[0]),
                "size": float(src.size), "shape": src.shape}
    G["scr"] = {"z": float(scr.z), "cx": float(scr.center[0]),
                "hx": float(scr.edge_x) / 2, "hy": float(scr.edge_y) / 2,
                "nx": scr.nx, "ny": scr.ny,
                "ref": scr.reference[0] if scr.reference else None}
    G["cfg_src"], G["cfg_scr"] = src, scr
    G["rays"] = trace_rays(G, p, n=cfg.schematic_rays)
    return G


def trace_rays(G, p, n=N_RAYS, seed=7):
    rng = random.Random(seed)
    mode, optic = G["mode"], G["optic"]
    src, scr = G["src"], G["scr"]
    sz = src["z"]
    if mode == "capillary":
        z0f = G["z0"]
        disks = [entrance_disk(b, z0f) for b in G["bores"]
                 if abs(float(b["center"][1])) < 1e-9] or \
                [entrance_disk(b, z0f) for b in G["bores"]]
        mb = int(G["cfg"].max_bounces)
    else:
        mb = int(G["cfg"].max_bounces) if mode == "lloyd" else 4
    screen_z = lift(scr["z"], p)
    rays = []
    for i in range(n):
        if src["shape"] == "point" or src["size"] <= 0:
            xo = src["x"]
        elif src["shape"] == "gaussian":
            xo = src["x"] + rng.gauss(0, src["size"])
        else:
            xo = src["x"] + src["size"] * (2 * rng.random() - 1)
        origin = (lift(xo, p), lift(0.0, p), lift(sz, p))
        if mode == "capillary":
            cx, cy, a = disks[rng.randrange(len(disks))]
            tx = cx + a * (2 * rng.random() - 1)
            d = vunit((lift(tx - xo, p), lift(0.0, p), lift(G["z0"] - sz, p)))
        elif mode == "lloyd":
            # alternate: direct rays into the window / reflected rays into the
            # window via the virtual source, mirror hit kept within [z0, z1]
            D = scr["z"] - sz
            x_lo, x_hi = scr["cx"] - scr["hx"], scr["cx"] + scr["hx"]
            if i % 2 == 0:
                tx = x_lo + (x_hi - x_lo) * rng.random()
                m = (tx - xo) / D
            else:
                lo = max((x_lo + xo) / D, xo / (G["z1"] - sz))
                hi = min((x_hi + xo) / D, xo / (G["z0"] - sz)) if G["z0"] > sz \
                    else (x_hi + xo) / D
                if lo >= hi:
                    lo, hi = (x_lo + xo) / D, (x_hi + xo) / D
                m = -(lo + (hi - lo) * rng.random())
            d = vunit((lift(m, p), lift(0.0, p), lift(1.0, p)))
        else:
            tx = scr["cx"] + scr["hx"] * (2 * rng.random() - 1)
            d = vunit((lift(tx - xo, p), lift(0.0, p), lift(scr["z"] - sz, p)))
        tr = trace_ray(origin, d, optic, screen_z, mb)
        pts = [origin] + [q for q, _ in tr.reflections] + [tr.point]
        rays.append((tr.fate, [(float(q[2]), float(q[0])) for q in pts]))
    return rays


# ---------------------------------------------------------------- rendering
class View:
    """Independent linear scales for z (horizontal) and x (vertical, up)."""

    def __init__(self, zr, xr, box):
        self.za, self.zb = zr
        self.xa, self.xb = xr
        self.x0, self.y0, self.x1, self.y1 = box     # pixel frame (y down)

    def px(self, z):
        return self.x0 + (z - self.za) / (self.zb - self.za) * (self.x1 - self.x0)

    def py(self, x):
        return self.y1 - (x - self.xa) / (self.xb - self.xa) * (self.y1 - self.y0)

    def pt(self, z, x):
        return (self.px(z), self.py(x))


def side_view(G):
    mode = G["mode"]
    src, scr = G["src"], G["scr"]
    za, zb = src["z"], scr["z"]
    zpad = 0.03 * (zb - za)
    zr = (za - zpad, zb + zpad)
    # x extent from the features only (source, optic, window); stray rays clip
    xs = [src["x"] - src["size"], src["x"] + src["size"],
          scr["cx"] - scr["hx"], scr["cx"] + scr["hx"]]
    if mode == "lloyd":
        xs += [0.0, G["height"], -G["height"]]
    if mode == "capillary":
        for b in G["bores"]:
            cx, cy = float(b["center"][0]), float(b["center"][1])
            if abs(cy) < 1e-9:
                for k in range(13):
                    zf = G["z0"] + (G["z1"] - G["z0"]) * k / 12
                    ax = torus_axis(b, G["z0"], zf) if b.get("kind") == "torus" else cx
                    xs += [ax + r_of_z(b, zf), ax - r_of_z(b, zf)]
    xlo, xhi = min(xs), max(xs)
    xm = 0.12 * (xhi - xlo or 1e-6)
    xr = (xlo - xm, xhi + xm)

    W = 1080
    top, bot = 62, 352
    box = (96, top, W - 210, bot)
    H = bot + (104 if mode == "lloyd" else 74)
    v = View(zr, xr, box)
    e = [RECT(box[0], box[1], box[2] - box[0], box[3] - box[1], "white", "#ccc")]
    ax_y = v.py(0.0)
    e.append(L(box[0], ax_y, box[2], ax_y, AXIS, 1.0, "6,5"))
    mag = ((v.y1 - v.y0) / (v.xb - v.xa)) / ((v.x1 - v.x0) / (v.zb - v.za))
    e.append(T(box[0], top - 8, f"side view (x–z)  ·  transverse ×{mag:.0f}",
              11, "start", "#888"))

    e.append(f'<clipPath id="rayclip"><rect x="{box[0]}" y="{box[1]}" '
             f'width="{box[2] - box[0]}" height="{box[3] - box[1]}"/></clipPath>')
    e += draw_optic(G, v, box)
    e += draw_source(G, v)
    e += draw_screen(G, v, box)
    e.append('<g clip-path="url(#rayclip)">')
    for i, (fate, pts) in enumerate(G["rays"]):
        e.append(POLY([v.pt(z, x) for z, x in pts], hsv(i, len(G["rays"])), 1.1,
                      opacity=0.85))
    e.append('</g>')
    e += dimensions(G, v, box)
    e += scale_bars(G, v, box)
    e += legend(G, box)
    return {"w": W, "h": H, "body": "".join(e)}


def draw_optic(G, v, box):
    mode = G["mode"]
    e = []
    if mode == "free":
        return e
    if mode == "lloyd":
        z0, z1 = G["z0"], G["z1"]
        y0 = v.py(0.0)
        e.append(L(v.px(z0), y0, v.px(z1), y0, "#37474f", 2.4))
        for xx in range(int(v.px(z0)), int(v.px(z1)), 13):     # glass hatching below
            e.append(L(xx, y0, xx + 9, min(box[3], y0 + 11), "#90a4ae", 0.7))
        e.append(T((v.px(z0) + v.px(z1)) / 2, y0 + 26, "mirror (glass x<0)",
                  11, "middle", "#37474f"))
        return e
    # capillary walls
    zs = [G["z0"] + (G["z1"] - G["z0"]) * k / 60 for k in range(61)]
    for b in G["bores"]:
        cy = float(b["center"][1])
        if abs(cy) > 1e-9:
            continue
        kind = b.get("kind", "cylinder")
        top, bot = [], []
        for zf in zs:
            ax = torus_axis(b, G["z0"], zf) if kind == "torus" else float(b["center"][0])
            r = r_of_z(b, zf)
            top.append(v.pt(zf, ax + r))
            bot.append(v.pt(zf, ax - r))
        poly = top + bot[::-1]
        fill = "#eaf2f8"
        e.append(f'<polygon points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in poly)}" '
                 f'fill="{fill}" stroke="none" opacity="0.6"/>')
        e.append(POLY(top, WALL, 1.6))
        e.append(POLY(bot, WALL, 1.6))
        e.append(L(*v.pt(G["z0"], float(b["center"][0]) + r_of_z(b, G["z0"])),
                   *v.pt(G["z0"], float(b["center"][0]) - r_of_z(b, G["z0"])),
                   WALL, 1.2))
    for b in G["bores"]:                           # bend geometry, once
        if b.get("kind") == "torus" and abs(float(b["center"][1])) < 1e-9:
            R = float(b["bend"]["radius"])
            th = math.asin((G["z1"] - G["z0"]) / R)
            sag = R * (1.0 - math.cos(th))
            zm = (G["z0"] + G["z1"]) / 2
            apex = torus_axis(b, G["z0"], zm) + r_of_z(b, zm)
            e.append(T(v.px(zm), v.py(apex) - 8,
                      f"bend R = {R:g} m,  θ = {th*1e3:.2f} mrad", 10.5,
                      "end", "#37474f"))
            cx = float(b["center"][0])
            e += arrow_v(v.px(G["z1"]) + 14, v.py(cx),
                         v.py(torus_axis(b, G["z0"], G["z1"])),
                         "sag " + eng(sag), 10)
            break
    return e


def draw_source(G, v):
    src = G["src"]
    e = []
    z, x, sz = src["z"], src["x"], src["size"]
    point = src["shape"] == "point" or sz <= 0
    if point:
        e.append(CIRC(*v.pt(z, x), 4.5, GREEN, "#186a18", 1))
    else:
        p0, p1 = v.pt(z, x - sz), v.pt(z, x + sz)
        e.append(L(p0[0], p0[1], p1[0], p1[1], GREEN, 3.4))
        e.append(CIRC(*v.pt(z, x), 2.0, GREEN))
    px, py = v.pt(z, x)
    e.append(T(px + 9, py - 5, "source", 11.5, "start", "#186a18"))
    if not point:
        tag = "σ = " if src["shape"] == "gaussian" else "r = "
        e.append(T(px + 9, py + 8, tag + eng(sz), 10.5, "start", "#186a18"))
    if G["mode"] == "lloyd":                       # mirror image behind x=0
        vx, vy = v.pt(z, -G["height"])
        e.append(CIRC(vx, vy, 4.5, "none", "#186a18", 1.2))
        e.append(f'<circle cx="{vx:.1f}" cy="{vy:.1f}" r="4.5" fill="none" '
                 f'stroke="#186a18" stroke-dasharray="3,2"/>')
        e.append(T(vx + 9, vy + 4, "virtual source", 10.5, "start", "#7a9a7a"))
    return e


def draw_screen(G, v, box):
    scr = G["scr"]
    z = scr["z"]
    y_lo, y_hi = v.py(scr["cx"] - scr["hx"]), v.py(scr["cx"] + scr["hx"])
    px = v.px(z)
    e = [L(px, max(box[1], y_hi), px, min(box[3], y_lo), BLUE, 3.0)]
    e.append(T(px - 4, box[1] - 6, "screen", 11.5, "end", BLUE))
    e.append(T(px - 4, box[1] + 9, "window " + eng(2 * scr["hx"]), 10, "end", BLUE))
    if scr["ref"] is not None:
        ry = v.py(scr["ref"])
        e.append(CIRC(px, ry, 3.5, "none", "#d62728", 1.4))
        e.append(T(px - 8, ry + 4, "P_ref", 10, "end", "#d62728"))
    return e


def dimensions(G, v, box):
    mode = G["mode"]
    src, scr = G["src"], G["scr"]
    e = []
    yb = box[3] + 34

    def zspan(z0, z1, label):
        if abs(v.px(z1) - v.px(z0)) < 18:          # too narrow for an arrow: label only
            e.append(T((v.px(z0) + v.px(z1)) / 2, yb - 5, label, 11, "middle", DIM))
        else:
            e.extend(arrow_h(v.px(z0), v.px(z1), yb, label))

    if mode in ("capillary", "lloyd"):
        zspan(src["z"], G["z0"], "d₀ = " + eng(G["z0"] - src["z"]))
        zspan(G["z0"], G["z1"], "L = " + eng(G["z1"] - G["z0"]))
        zspan(G["z1"], scr["z"], "d₂ = " + eng(scr["z"] - G["z1"]))
    if mode == "lloyd":
        e += arrow_h(v.px(src["z"]), v.px(scr["z"]), yb + 30, "D = " + eng(scr["z"] - src["z"]))
    if mode == "free":
        e += arrow_h(v.px(src["z"]), v.px(scr["z"]), yb, "D = " + eng(scr["z"] - src["z"]))

    if mode == "lloyd":                            # source height above the mirror plane
        e += arrow_v(v.px(src["z"]) + 12, v.py(0.0), v.py(G["height"]), "r₀ = " + eng(G["height"]))
    if mode == "capillary":                        # bore diameter, right margin
        b0 = G["bores"][0]
        r0, r1 = r_of_z(b0, G["z0"]), r_of_z(b0, G["z1"])
        cx0 = float(b0["center"][0])
        note = "⌀" + eng(2 * r0)
        if b0.get("kind") == "revolution" and abs(r1 - r0) > 0.01 * r0:
            note += " → ⌀" + eng(2 * r1)
        e += arrow_v(box[2] + 40, v.py(cx0 - r0), v.py(cx0 + r0), note)
    return e


def _nice(target):
    """Largest nice number (1/2/5·10ⁿ) not exceeding target."""
    mag = 10 ** math.floor(math.log10(target))
    return next(m * mag for m in (10, 5, 2, 1, 0.5, 0.2, 0.1) if m * mag <= target)


def scale_bars(G, v, box):
    """L-shaped z/x scale bars: equal-looking pixel lengths expose the anamorphism."""
    e = []
    sz = _nice((v.zb - v.za) / 6)
    sx = _nice((v.xb - v.xa) / 6)
    x0, y0 = box[0] + 12, box[3] - 14
    lz = v.px(v.za + sz) - v.px(v.za)
    lx = v.py(v.xa) - v.py(v.xa + sx)
    e.append(L(x0, y0, x0 + lz, y0, INK, 2.2))
    e.append(T(x0, y0 + 13, eng(sz), 10, "start", INK))
    e.append(L(x0, y0, x0, y0 - lx, INK, 2.2))
    e.append(T(x0 + 5, y0 - lx + 3, eng(sx), 10, "start", INK))
    return e


def legend(G, box):
    items = [("source", GREEN), ("screen", BLUE)]
    if G["optic"] is not None:
        items.append(("mirror" if G["mode"] == "lloyd" else "wall", WALL))
    items.append(("rays", hsv(3, 10)))
    y = box[3] + (92 if G["mode"] == "lloyd" else 60)      # below the z-dimensions
    x = box[0]
    e = []
    for lab, col in items:
        e.append(L(x, y - 4, x + 18, y - 4, col, 2.6))
        e.append(T(x + 23, y, lab, 10.5, "start", "#444"))
        x += 45 + len(lab) * 6.6
    return e


# ---------------------------------------------------------------- transverse
def need_inset(G):
    """Transverse view only when it adds something the side view can't show."""
    if G["mode"] != "capillary":
        return False
    if G["scr"]["ny"] > 1 or len(G["bores"]) > 1:
        return True
    return any(b.get("kind") == "polygon" for b in G["bores"])


def inset(G):
    bores, scr, src = G["bores"], G["scr"], G["src"]
    pts = []
    for b in bores:
        cx, cy = float(b["center"][0]), float(b["center"][1])
        r = r_of_z(b, G["z0"])
        if b.get("kind") == "polygon":
            r = r / math.cos(math.pi / int(b["sides"]))
        pts += [(cx + r, cy + r), (cx - r, cy - r)]
    if scr["ny"] > 1:                              # thin strip (ny=1) is not a transverse feature
        pts += [(scr["cx"] + scr["hx"], scr["hy"]), (scr["cx"] - scr["hx"], -scr["hy"])]
    xs = [q[0] for q in pts] + [src["x"] + src["size"], src["x"] - src["size"]]
    ys = [q[1] for q in pts] + [src["size"], -src["size"]]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = 0.12 * (hi - lo or 1e-6)
    lo, hi = lo - pad, hi + pad
    S = 330
    box = (60, 50, 60 + S, 50 + S)

    def m(x, y):
        px = box[0] + (x - lo) / (hi - lo) * S
        py = box[3] - (y - lo) / (hi - lo) * S
        return px, py

    e = [RECT(box[0], box[1], S, S, "white", "#ccc")]
    e.append(L(*m(lo, 0), *m(hi, 0), AXIS, 0.8, "5,4"))
    e.append(L(*m(0, lo), *m(0, hi), AXIS, 0.8, "5,4"))
    # screen window
    if scr["ny"] > 1:
        a, d = m(scr["cx"] - scr["hx"], scr["hy"])
        c, bb = m(scr["cx"] + scr["hx"], -scr["hy"])
        e.append(RECT(min(a, c), min(bb, d), abs(c - a), abs(bb - d), "none", BLUE, 1.2, "4,3"))
    # bores
    for b in bores:
        cx, cy = float(b["center"][0]), float(b["center"][1])
        r = r_of_z(b, G["z0"])
        if b.get("kind") == "polygon":
            ns = int(b["sides"])
            rot = float(b["rotation"])
            rc = r / math.cos(math.pi / ns)
            poly = [m(cx + rc * math.cos(rot + 2 * math.pi * k / ns + math.pi / ns),
                      cy + rc * math.sin(rot + 2 * math.pi * k / ns + math.pi / ns))
                    for k in range(ns)]
            e.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in poly)}" '
                     f'fill="#eaf2f8" stroke="{WALL}" stroke-width="1.4"/>')
        else:
            cxp, cyp = m(cx, cy)
            rp = abs(m(cx + r, cy)[0] - cxp)
            e.append(CIRC(cxp, cyp, rp, "#eaf2f8", WALL, 1.4))
        if b.get("bend"):                      # convergence/deflection arrow
            ux, uy = (float(c) for c in b["bend"]["toward"])
            x0, y0 = m(cx, cy)
            e.append(f'<path d="M {x0:.1f} {y0:.1f} l {ux*13:.1f} {-uy*13:.1f}" '
                     f'stroke="#d62728" stroke-width="1.4" marker-end="url(#ah)"/>')
    # source spot
    sx, sy = m(src["x"], 0.0)
    if src["shape"] == "point" or src["size"] <= 0:
        e.append(CIRC(sx, sy, 3, GREEN, "#186a18", 1))
    else:
        rp = abs(m(src["x"] + src["size"], 0)[0] - sx)
        e.append(CIRC(sx, sy, max(2, rp), GREEN, "#186a18", 1))
    e.append(T(box[0] + S / 2, box[1] - 8, "entrance section (x–y)", 11.5, "middle", "#444"))
    # scale bar
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / 3))
    step = next(mm * step for mm in (1, 2, 5, 10) if mm * step >= span / 4)
    bx, by = box[0] + 8, box[3] - 8
    e.append(L(bx, by, bx + step / (hi - lo) * S, by, INK, 2.2))
    e.append(T(bx, by - 4, eng(step, "auto"), 10, "start", INK))
    b0 = bores[0]
    two = 2 * r_of_z(b0, G["z0"])
    lab = ("across-flats " + eng(two, "auto")) if b0.get("kind") == "polygon" else ("⌀" + eng(two, "auto"))
    e.append(T(box[0] + S / 2, box[3] + 18, lab, 11, "middle", "#444"))
    if len(bores) > 1:
        c0 = bores[0]["center"]
        c1 = bores[1]["center"]
        pitch = math.hypot(float(c1[0]) - float(c0[0]), float(c1[1]) - float(c0[1]))
        e.append(T(box[0] + S / 2, box[3] + 34, "pitch " + eng(pitch, "auto"), 11, "middle", "#444"))
    defs = ('<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="5" refY="3" '
            'orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#d62728"/></marker></defs>')
    return {"w": box[0] + S + 60, "h": box[1] + S + 50, "body": defs + "".join(e)}


def need_unrolled(G):
    """Bore-relative panel: one in-plane bent bore whose sag dwarfs its radius."""
    return (G["mode"] == "capillary" and len(G["bores"]) == 1
            and G["bores"][0].get("kind") == "torus"
            and abs(float(G["bores"][0]["center"][1])) < 1e-9)


def unrolled(G):
    """x − x_axis(z): the whispering-gallery pattern the to-scale view hides."""
    b = G["bores"][0]
    z0, z1 = G["z0"], G["z1"]
    a = r_of_z(b, z0)
    W, H = 1080, 250
    box = (96, 44, W - 210, H - 40)
    v = View((z0, z1), (-1.45 * a, 1.45 * a), box)
    e = [RECT(box[0], box[1], box[2] - box[0], box[3] - box[1], "white", "#ccc"),
         f'<clipPath id="unroll"><rect x="{box[0]}" y="{box[1]}" '
         f'width="{box[2] - box[0]}" height="{box[3] - box[1]}"/></clipPath>',
         L(box[0], v.py(0), box[2], v.py(0), AXIS, 0.8, "6,5"),
         L(box[0], v.py(a), box[2], v.py(a), WALL, 1.6),
         L(box[0], v.py(-a), box[2], v.py(-a), WALL, 1.6),
         T(box[0], box[1] - 8, "bore-relative view:  x − x_axis(z)   "
           "(walls straightened, same rays)", 11, "start", "#888"),
         T(box[2] + 6, v.py(a) + 4, "+a", 10, "start", WALL),
         T(box[2] + 6, v.py(-a) + 4, "−a", 10, "start", WALL)]
    e.append('<g clip-path="url(#unroll)">')
    for i, (fate, pts) in enumerate(G["rays"]):
        loc = []
        for (zA, xA), (zB, xB) in zip(pts, pts[1:]):
            for k in range(25):
                zk = zA + (zB - zA) * k / 24
                if z0 <= zk <= z1 and abs(zB - zA) > 0:
                    xk = xA + (xB - xA) * (zk - zA) / (zB - zA)
                    loc.append(v.pt(zk, xk - torus_axis(b, z0, zk)))
        if loc:
            e.append(POLY(loc, hsv(i, len(G["rays"])), 1.1, opacity=0.85))
    e.append('</g>')
    e += arrow_v(box[2] + 40, v.py(-a), v.py(a), "⌀" + eng(2 * a))
    return {"w": W, "h": H, "body": "".join(e)}


def compose(G):
    figs = [side_view(G)]
    if need_inset(G):
        figs.append(inset(G))
    if need_unrolled(G):
        figs.append(unrolled(G))
    if len(figs) == 1:
        return figs[0]
    w = max(f["w"] for f in figs)
    h = sum(f["h"] for f in figs)
    body, y = [], 0
    for f in figs:
        body.append(f'<g transform="translate({(w - f["w"]) / 2:.0f},{y})">'
                    f'{f["body"]}</g>')
        y += f["h"]
    return {"w": w, "h": h, "body": "".join(body)}
