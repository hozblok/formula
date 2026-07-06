// C++ twin of the CAPSYSred tracer (capsysred/trace.py, surfaces.py,
// wall_*.py): the same mp operations in the same order, so every result is
// bit-identical to the Python reference (tests/test_native_trace.py).
//
// Every binary op goes through the lvalue helpers below. Python reaches
// boost only through the pybind-bound lvalue operators (t = a; t op= b);
// bare C++ expressions would pick boost's rvalue overloads, which compute
// in place with swapped operands and leave different low-order limbs in
// cpp_dec_float results. Operand order is Python's, verbatim.
//
// Number bridges are exact by construction: float(Number) formats at P
// digits and parses with CPython's own float(str); lift(float) parses
// CPython's own repr(float); comparisons re-parse display strings.
#ifndef CS_TRACE_HPP
#define CS_TRACE_HPP

#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <cseval/cseval.hpp>

#include "strip_neg_zero.hpp"

namespace cstrace {

// Forward-step floor for the ray parameter t (m); capsysred.types._EPS_T.
constexpr double kEpsT = 1e-12;

// float(str): CPython's parser itself (correctly rounded, overflow -> inf).
inline double parse_double(const std::string &s) {
  double v = PyOS_string_to_double(s.c_str(), nullptr, nullptr);
  if (v == -1.0 && PyErr_Occurred()) {
    throw pybind11::error_already_set();
  }
  return v;
}

// repr(float): CPython's shortest round-trip formatter itself.
inline std::string shortest_repr(double v) {
  char *s = PyOS_double_to_string(v, 'r', 0, Py_DTSF_ADD_DOT_0, nullptr);
  if (!s) {
    throw std::bad_alloc();
  }
  std::string out(s);
  PyMem_Free(s);
  return out;
}

// CPython complex arithmetic (Objects/complexobject.c) for the float
// Durand-Kerner stage of wall_torus._dk_roots.
struct PyC {
  double re, im;
};

inline PyC c_prod(PyC a, PyC b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

inline PyC c_sub(PyC a, PyC b) { return {a.re - b.re, a.im - b.im}; }

inline bool c_truthy(PyC a) { return a.re != 0.0 || a.im != 0.0; }

inline double c_abs(PyC z) {  // _Py_c_abs
  if (!std::isfinite(z.re) || !std::isfinite(z.im)) {
    if (std::isinf(z.re)) {
      return std::fabs(z.re);
    }
    if (std::isinf(z.im)) {
      return std::fabs(z.im);
    }
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::hypot(z.re, z.im);
}

inline PyC c_quot(PyC a, PyC b) {  // _Py_c_quot (Smith's rule)
  PyC r;
  const double abs_breal = b.re < 0 ? -b.re : b.re;
  const double abs_bimag = b.im < 0 ? -b.im : b.im;
  if (abs_breal >= abs_bimag) {
    if (abs_breal == 0.0) {
      r.re = r.im = 0.0;
    } else {
      const double ratio = b.im / b.re;
      const double denom = b.re + b.im * ratio;
      r.re = (a.re + a.im * ratio) / denom;
      r.im = (a.im - a.re * ratio) / denom;
    }
  } else if (abs_bimag >= abs_breal) {
    const double ratio = b.re / b.im;
    const double denom = b.re * ratio + b.im;
    r.re = (a.re * ratio + a.im) / denom;
    r.im = (a.im * ratio - a.re) / denom;
  } else {
    r.re = r.im = std::numeric_limits<double>::quiet_NaN();
  }
  return r;
}

inline PyC c_powu(PyC x, long n) {  // c_powu: complex ** small positive int
  PyC r{1.0, 0.0}, p = x;
  long mask = 1;
  while (mask > 0 && n >= mask) {
    if (n & mask) {
      r = c_prod(r, p);
    }
    mask <<= 1;
    p = c_prod(p, p);
  }
  return r;
}

// wall_torus._dk_roots: all roots of a float polynomial, Durand-Kerner.
inline std::vector<PyC> dk_roots(const std::vector<double> &cf) {
  double scale = 1.0;
  bool have = false;
  for (size_t i = 0; i + 1 < cf.size(); ++i) {
    double v = cf[i + 1];
    if (v == 0.0) {  // Python truthiness filter `if v`
      continue;
    }
    double item = std::pow(std::fabs(v), 1.0 / static_cast<double>(i + 1));
    if (!have || item > scale) {
      scale = item;
      have = true;
    }
  }
  if (scale == 0.0) {  // `or 1.0`
    scale = 1.0;
  }
  const size_t n = cf.size() - 1;
  std::vector<PyC> roots(n);
  for (size_t k = 1; k <= n; ++k) {
    roots[k - 1] = c_prod(c_powu({0.4, 0.9}, static_cast<long>(k)),
                          {scale, 0.0});
  }
  for (int it = 0; it < 80; ++it) {
    double moved = 0.0;
    for (size_t i = 0; i < n; ++i) {
      PyC r = roots[i];
      PyC num{cf[0], 0.0};
      for (size_t j = 1; j < cf.size(); ++j) {
        PyC t = c_prod(num, r);
        num = {t.re + cf[j], t.im + 0.0};
      }
      PyC den{cf[0], 0.0};
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          den = c_prod(den, c_sub(r, roots[j]));
        }
      }
      PyC step = c_truthy(den) ? c_quot(num, den) : PyC{0.0, 0.0};
      roots[i] = c_sub(r, step);
      double a = c_abs(step);
      if (a > moved) {
        moved = a;
      }
    }
    if (moved < 3e-15 * scale) {
      break;
    }
  }
  return roots;
}

template <unsigned P>
struct Tracer {
  using R = mp_real<P>;
  struct V3 {
    R x, y, z;
  };

  // The pybind-bound operator shapes: parameters are lvalues, so boost's
  // const& overloads run — never the in-place rvalue ones.
  static R add(const R &a, const R &b) { return a + b; }
  static R sub(const R &a, const R &b) { return a - b; }
  static R mul(const R &a, const R &b) { return a * b; }
  static R div(const R &a, const R &b) { return a / b; }
  static R neg(const R &a) { return -a; }
  static R abs_(const R &a) { return abs(a); }
  static R sqrt_(const R &a) { return sqrt(a); }

  // Interned wall constants: same parse path as Number("<s>", P).
  static const R &zero() { static const R v("0"); return v; }
  static const R &half() { static const R v("0.5"); return v; }
  static const R &one() { static const R v("1"); return v; }
  static const R &two() { static const R v("2"); return v; }
  static const R &three() { static const R v("3"); return v; }
  static const R &four() { static const R v("4"); return v; }

  // float(Number): display string at P digits -> CPython float(str).
  static double py_float(const R &v) {
    return parse_double(strip_neg_zero(v.str(P, std::ios_base::fmtflags(0))));
  }

  // Number._cmp key: the value re-parsed from its display string.
  static R cmp_key(const R &v) {
    return R(strip_neg_zero(v.str(P, std::ios_base::fmtflags(0))));
  }

  // nums.lift(float): mp value of repr(float).
  static R lift(double v) { return R(shortest_repr(v)); }

  // nums.py 3-vector kit.
  static R vdot(const V3 &a, const V3 &b) {
    return add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z));
  }
  static V3 vadd(const V3 &a, const V3 &b) {
    return {add(a.x, b.x), add(a.y, b.y), add(a.z, b.z)};
  }
  static V3 vsub(const V3 &a, const V3 &b) {
    return {sub(a.x, b.x), sub(a.y, b.y), sub(a.z, b.z)};
  }
  static V3 vscale(const V3 &a, const R &s) {
    return {mul(a.x, s), mul(a.y, s), mul(a.z, s)};
  }
  static R vnorm(const V3 &a) { return sqrt_(vdot(a, a)); }
  static V3 vunit(const V3 &a) {
    R n = vnorm(a);
    return {div(a.x, n), div(a.y, n), div(a.z, n)};
  }

  template <size_t N>
  static R horner(const std::array<R, N> &cs, const R &t) {
    R v = cs[0];
    for (size_t i = 1; i < N; ++i) {
      v = add(mul(v, t), cs[i]);
    }
    return v;
  }

  struct Hit {
    R t;
    V3 point, normal;
  };
  using OptHit = std::optional<Hit>;

  // Walls carry the exact mp/float state of their Python twins (native.py
  // passes the already-derived attributes; nothing is re-derived here).
  // CylinderWall arrives as Revolution with c1 = c2 = 0: the extra mp terms
  // are exact no-ops (x±0 and x*0 are limb-exact in cpp_dec_float), so the
  // results stay bit-equal to the Python CylinderWall.
  struct Revolution {
    R cx, cy, c0, c1, c2;
    double eps, cxf, cyf, c0f, c1f, c2f;
  };
  struct Polygon {
    std::vector<std::pair<R, R>> faces;
    std::vector<std::pair<double, double>> facesf;
    R apothem, cx, cy;
    double af, cxf, cyf;
  };
  struct Torus {
    V3 C, nhat;
    R bigR, K, fourR2;
    double cf[3], nf[2], rf, in2;
  };
  using Wall = std::variant<Revolution, Polygon, Torus>;

  // min((t for t in ts if float(t) > _EPS_T), key=float): ties keep the first.
  static std::optional<R> first_forward(const std::array<R, 2> &ts,
                                        size_t count) {
    const R *best = nullptr;
    double bestf = 0.0;
    for (size_t i = 0; i < count; ++i) {
      double f = py_float(ts[i]);
      if (f > kEpsT && (!best || f < bestf)) {
        best = &ts[i];
        bestf = f;
      }
    }
    if (!best) {
      return std::nullopt;
    }
    return *best;
  }

  // wall_revolution.RevolutionWall
  static bool inside(const Revolution &w, double xf, double yf, double zf) {
    double dx = xf - w.cxf, dy = yf - w.cyf;
    double r2 = w.c0f + zf * (w.c1f + zf * w.c2f);
    return dx * dx + dy * dy < r2 * (1.0 + 1e-9);
  }

  static OptHit hit(const Revolution &w, const V3 &O, const V3 &d, const R &) {
    R rx = sub(O.x, w.cx), ry = sub(O.y, w.cy);
    R A = sub(add(mul(d.x, d.x), mul(d.y, d.y)), mul(mul(w.c2, d.z), d.z));
    R B = sub(mul(two(), add(mul(rx, d.x), mul(ry, d.y))),
              mul(d.z, add(w.c1, mul(mul(two(), w.c2), O.z))));
    R C = sub(add(mul(rx, rx), mul(ry, ry)),
              add(w.c0, mul(O.z, add(w.c1, mul(w.c2, O.z)))));
    std::array<R, 2> ts;
    size_t count = 0;
    if (std::fabs(py_float(A)) < w.eps) {
      if (std::fabs(py_float(B)) < w.eps) {
        return std::nullopt;
      }
      ts[0] = div(sub(zero(), C), B);
      count = 1;
    } else {
      R disc = sub(mul(B, B), mul(mul(four(), A), C));
      if (py_float(disc) < 0.0) {
        return std::nullopt;
      }
      R root = sqrt_(disc);
      ts[0] = div(sub(sub(zero(), B), root), mul(two(), A));
      ts[1] = div(add(sub(zero(), B), root), mul(two(), A));
      count = 2;
    }
    std::optional<R> t = first_forward(ts, count);
    if (!t) {
      return std::nullopt;
    }
    V3 point = vadd(O, vscale(d, *t));
    V3 n = {sub(point.x, w.cx), sub(point.y, w.cy),
            sub(zero(), add(mul(w.c1, half()), mul(w.c2, point.z)))};
    return Hit{*t, point, vunit(n)};
  }

  // wall_polygon.PolygonWall
  static bool inside(const Polygon &w, double xf, double yf, double) {
    double dx = xf - w.cxf, dy = yf - w.cyf;
    double lim = w.af * (1.0 + 1e-9);
    for (const auto &f : w.facesf) {
      if (!(f.first * dx + f.second * dy < lim)) {
        return false;
      }
    }
    return true;
  }

  static OptHit hit(const Polygon &w, const V3 &O, const V3 &d, const R &) {
    double rxf = py_float(O.x) - w.cxf, ryf = py_float(O.y) - w.cyf;
    double dxf = py_float(d.x), dyf = py_float(d.y);
    long best = -1;
    double bestt = 0.0;
    for (size_t i = 0; i < w.facesf.size(); ++i) {
      double mxf = w.facesf[i].first, myf = w.facesf[i].second;
      double md = mxf * dxf + myf * dyf;
      if (md <= 1e-30) {
        continue;
      }
      double tf = (w.af - (mxf * rxf + myf * ryf)) / md;
      if (tf > kEpsT && (best < 0 || tf < bestt)) {
        best = static_cast<long>(i);
        bestt = tf;
      }
    }
    if (best < 0) {
      return std::nullopt;
    }
    const R &mx = w.faces[static_cast<size_t>(best)].first;
    const R &my = w.faces[static_cast<size_t>(best)].second;
    R rx = sub(O.x, w.cx), ry = sub(O.y, w.cy);
    R t = div(sub(w.apothem, add(mul(mx, rx), mul(my, ry))),
              add(mul(mx, d.x), mul(my, d.y)));
    return Hit{t, vadd(O, vscale(d, t)), {mx, my, zero()}};
  }

  // wall_torus._quartic_first: smallest real root in (_EPS_T, t_capf].
  static std::optional<R> quartic_first(const std::array<R, 5> &c,
                                        double t_capf) {
    std::array<R, 4> dc = {mul(c[0], four()), mul(c[1], three()),
                           mul(c[2], two()), c[3]};
    std::vector<double> cf(5);
    for (size_t i = 0; i < 5; ++i) {
      cf[i] = py_float(c[i]);
    }
    std::vector<double> cand;
    for (const PyC &r : dk_roots(cf)) {
      double m = std::fabs(r.re) > 1.0 ? std::fabs(r.re) : 1.0;
      if (std::fabs(r.im) <= 1e-6 * m && kEpsT / 2 < r.re && r.re <= t_capf) {
        cand.push_back(r.re);
      }
    }
    std::sort(cand.begin(), cand.end());
    std::optional<R> best;
    double bestf = 0.0;
    for (double seed : cand) {
      R t = lift(seed);
      for (int i = 0; i < 12; ++i) {
        R gp = horner(dc, t);
        if (py_float(abs_(gp)) == 0.0) {
          break;
        }
        R step = div(horner(c, t), gp);
        t = sub(t, step);
        double at = std::fabs(py_float(t));
        double floor_t = 1e-9 > at ? 1e-9 : at;  // max(abs(float(t)), 1e-9)
        if (py_float(abs_(step)) <= 1e-24 * floor_t) {
          double tf = py_float(t);
          if (tf > kEpsT && (!best || tf < bestf)) {
            best = t;
            bestf = tf;
          }
          break;
        }
      }
    }
    if (best) {
      return best;
    }
    // sign-change scan (geometric grid) + bisection on the exact quartic
    R lo = lift(kEpsT);
    double flo = py_float(horner(c, lo));
    for (int k = 1; k < 49; ++k) {
      R hi = lift(kEpsT * std::pow(t_capf / kEpsT, k / 48.0));
      double fhi = py_float(horner(c, hi));
      if ((flo < 0.0) != (fhi < 0.0)) {
        for (int i = 0; i < 90; ++i) {
          R mid = mul(add(lo, hi), half());
          double fmid = py_float(horner(c, mid));
          if ((flo < 0.0) != (fmid < 0.0)) {
            hi = mid;
          } else {
            lo = mid;
            flo = fmid;
          }
        }
        return mul(add(lo, hi), half());
      }
      lo = hi;
      flo = fhi;
    }
    return std::nullopt;
  }

  // wall_torus.TorusWall
  static bool inside(const Torus &w, double xf, double yf, double zf) {
    double wx = xf - w.cf[0], wy = yf - w.cf[1], wz = zf - w.cf[2];
    double s = wx * w.nf[0] + wy * w.nf[1];
    double m = wx * wx + wy * wy + wz * wz - s * s;
    double rho = std::sqrt(0.0 > m ? 0.0 : m);  // max(m, 0.0)
    double dr = rho - w.rf;
    return dr * dr + s * s < w.in2;
  }

  static OptHit hit(const Torus &w, const V3 &O, const V3 &d,
                    const R &t_exit) {
    V3 wv = vsub(O, w.C);
    R ws = vdot(wv, wv), wd = vdot(wv, d);
    R s0 = vdot(wv, w.nhat), sd = vdot(d, w.nhat);
    R u1 = mul(two(), wd), u0 = add(ws, w.K);
    std::array<R, 5> c = {
        one(),
        mul(two(), u1),
        sub(add(mul(u1, u1), mul(two(), u0)),
            mul(w.fourR2, sub(one(), mul(sd, sd)))),
        sub(mul(mul(two(), u1), u0),
            mul(mul(w.fourR2, two()), sub(wd, mul(s0, sd)))),
        sub(mul(u0, u0), mul(w.fourR2, sub(ws, mul(s0, s0))))};
    std::optional<R> t =
        quartic_first(c, py_float(t_exit) * (1.0 + 1e-9) + kEpsT);
    if (!t) {
      return std::nullopt;
    }
    V3 point = vadd(O, vscale(d, *t));
    V3 w2 = vsub(point, w.C);
    R s = vdot(w2, w.nhat);
    V3 q = vsub(w2, vscale(w.nhat, s));
    R rho = vnorm(q);
    V3 n = vadd(vscale(q, div(sub(rho, w.bigR), rho)), vscale(w.nhat, s));
    return Hit{*t, point, vunit(n)};
  }

  // surfaces.py event protocol:
  // ("reflect", t, point, normal) | ("pass", t) | ("absorb", t) | ("exit",)
  enum EventKind { kExit = 0, kPass = 1, kAbsorb = 2, kReflect = 3 };
  struct Event {
    EventKind kind;
    R t;
    V3 point, normal;
  };

  struct MirrorO {
    R z0, z1;
    double z0f, z1f;
  };
  struct Bundle {
    R z0, z1;
    double z0f, z1f;
    std::vector<Wall> walls;
  };
  using Optic = std::variant<MirrorO, Bundle>;

  // surfaces.Mirror.next_event
  static Event next_event(const MirrorO &m, const V3 &O, const V3 &d) {
    if (py_float(O.x) <= 0.0 || py_float(d.x) >= 0.0) {
      return {kExit, R(), {}, {}};
    }
    R t = div(neg(O.x), d.x);
    double zf = py_float(O.z) + py_float(t) * py_float(d.z);
    if (zf > m.z1f) {
      return {kExit, R(), {}, {}};
    }
    if (zf < m.z0f) {
      return {kAbsorb, div(sub(m.z0, O.z), d.z), {}, {}};
    }
    return {kReflect, t, vadd(O, vscale(d, t)), {one(), zero(), zero()}};
  }

  static OptHit wall_hit(const Wall &w, const V3 &O, const V3 &d,
                         const R &t_exit) {
    return std::visit([&](const auto &ww) { return hit(ww, O, d, t_exit); },
                      w);
  }

  // surfaces.CapillaryBundle._locate
  static const Wall *locate(const Bundle &b, const V3 &O) {
    double xf = py_float(O.x), yf = py_float(O.y), zf = py_float(O.z);
    for (const Wall &w : b.walls) {
      bool in = std::visit(
          [&](const auto &ww) { return inside(ww, xf, yf, zf); }, w);
      if (in) {
        return &w;
      }
    }
    return nullptr;
  }

  // surfaces.CapillaryBundle.next_event
  static Event next_event(const Bundle &b, const V3 &O, const V3 &d) {
    if (py_float(d.z) <= 0.0) {
      return {kAbsorb, zero(), {}, {}};
    }
    double zf = py_float(O.z);
    if (zf < b.z0f - kEpsT) {
      return {kPass, div(sub(b.z0, O.z), d.z), {}, {}};
    }
    if (zf >= b.z1f - kEpsT) {
      return {kExit, R(), {}, {}};
    }
    const Wall *wall = locate(b, O);
    if (!wall) {
      return {kAbsorb, zero(), {}, {}};
    }
    R t_exit = div(sub(b.z1, O.z), d.z);
    OptHit h = wall_hit(*wall, O, d, t_exit);
    if (!h || cmp_key(t_exit) <= cmp_key(h->t)) {
      return {kPass, t_exit, {}, {}};
    }
    return {kReflect, h->t, h->point, h->normal};
  }

  static Event next_event(const Optic &o, const V3 &O, const V3 &d) {
    return std::visit([&](const auto &v) { return next_event(v, O, d); }, o);
  }

  // trace.trace_ray
  enum Fate { fScreen = 0, fAbsorbed = 1, fLost = 2 };
  struct Refl {
    V3 point;
    R sin_g;
  };
  struct Result {
    Fate fate;
    V3 point;
    R opl;
    std::vector<Refl> reflections;
  };

  static Result trace(const Optic *optic, V3 O, V3 d, const R &screen_z,
                      long max_bounces) {
    R opl = zero();
    std::vector<Refl> refl;
    if (optic) {
      bool exited = false;
      for (long i = 0; i < max_bounces; ++i) {
        Event ev = next_event(*optic, O, d);
        if (ev.kind == kExit) {
          exited = true;
          break;
        }
        if (ev.kind == kPass) {
          O = vadd(O, vscale(d, ev.t));
          opl = add(opl, ev.t);
          continue;
        }
        if (ev.kind == kAbsorb) {
          return {fAbsorbed, vadd(O, vscale(d, ev.t)), add(opl, ev.t),
                  std::move(refl)};
        }
        opl = add(opl, ev.t);
        R dot = vdot(d, ev.normal);
        d = vsub(d, vscale(ev.normal, mul(two(), dot)));
        refl.push_back({ev.point, abs_(dot)});
        O = ev.point;
      }
      if (!exited) {  // bounce budget spent: for-else in trace_ray
        return {fLost, O, opl, std::move(refl)};
      }
    }
    if (py_float(d.z) <= 0.0) {
      return {fLost, O, opl, std::move(refl)};
    }
    R t = div(sub(screen_z, O.z), d.z);
    if (py_float(t) < 0.0) {
      return {fLost, O, opl, std::move(refl)};
    }
    V3 point = vadd(O, vscale(d, t));
    return {fScreen, point, add(opl, t), std::move(refl)};
  }
};

}  // namespace cstrace

#endif  // CS_TRACE_HPP
