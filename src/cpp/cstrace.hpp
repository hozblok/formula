// C++ fast path of the CAPSYSred tracer (capsysred/trace.py, surfaces.py,
// wall_*.py): the same algorithms, branch structure and mp type as the Python
// reference, written in plain boost arithmetic. Results agree with Python to
// working precision, not bit-for-bit (tests/test_native_trace.py asserts
// fates, event kinds and branch choices exactly, values within tolerances).
#ifndef CS_TRACE_HPP
#define CS_TRACE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <cseval/cseval.hpp>

namespace cstrace {

// Forward-step floor for the ray parameter t (m); capsysred.types._EPS_T.
constexpr double kEpsT = 1e-12;

// wall_torus._dk_roots: all roots of a float polynomial (Durand-Kerner);
// only seed material — every root is re-polished in mp or bisected exactly.
inline std::vector<std::complex<double>> dk_roots(
    const std::vector<double> &cf) {
  if (cf.size() < 2) {
    return {};
  }
  double scale = 0.0;
  for (size_t i = 1; i < cf.size(); ++i) {
    if (cf[i] != 0.0) {
      scale = std::max(scale,
                       std::pow(std::fabs(cf[i]), 1.0 / static_cast<double>(i)));
    }
  }
  if (scale == 0.0) {
    scale = 1.0;
  }
  const size_t n = cf.size() - 1;
  std::vector<std::complex<double>> roots(n);
  std::complex<double> seed{1.0, 0.0};
  for (size_t k = 0; k < n; ++k) {
    seed *= std::complex<double>{0.4, 0.9};
    roots[k] = seed * scale;
  }
  for (int it = 0; it < 80; ++it) {
    double moved = 0.0;
    for (size_t i = 0; i < n; ++i) {
      std::complex<double> num{cf[0], 0.0};
      for (size_t j = 1; j < cf.size(); ++j) {
        num = num * roots[i] + cf[j];
      }
      std::complex<double> den{cf[0], 0.0};
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          den *= roots[i] - roots[j];
        }
      }
      std::complex<double> step =
          den != 0.0 ? num / den : std::complex<double>{};
      roots[i] -= step;
      moved = std::max(moved, std::abs(step));
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

  static double to_double(const R &v) {
    return v.template convert_to<double>();
  }

  // types._EPS_T as an interned mp constant.
  static const R &eps_t() {
    static const R v(kEpsT);
    return v;
  }

  // nums.py 3-vector kit.
  static R vdot(const V3 &a, const V3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }
  static V3 vadd(const V3 &a, const V3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
  }
  static V3 vsub(const V3 &a, const V3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
  }
  static V3 vscale(const V3 &a, const R &s) {
    return {a.x * s, a.y * s, a.z * s};
  }
  static R vnorm(const V3 &a) { return sqrt(vdot(a, a)); }
  static V3 vunit(const V3 &a) {
    R n = vnorm(a);
    return {a.x / n, a.y / n, a.z / n};
  }

  template <size_t N>
  static R horner(const std::array<R, N> &cs, const R &t) {
    R v = cs[0];
    for (size_t i = 1; i < N; ++i) {
      v *= t;
      v += cs[i];
    }
    return v;
  }

  struct Hit {
    R t;
    V3 point, normal;
  };
  using OptHit = std::optional<Hit>;

  // Walls carry the mp/float state of their Python twins (native.py passes
  // the already-derived attributes; nothing is re-derived here). CylinderWall
  // arrives as Revolution with c1 = c2 = 0: `straight` skips the known-zero
  // terms while keeping the same branch structure.
  struct Revolution {
    R cx, cy, c0, c1, c2;
    double eps, cxf, cyf, c0f, c1f, c2f;
    bool straight() const { return c1f == 0.0 && c2f == 0.0; }
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

  // min((t for t in ts if float(t) > _EPS_T), key=float): ties keep the
  // first; exact mp order replaces the float key.
  static std::optional<R> first_forward(const std::array<R, 2> &ts,
                                        size_t count) {
    const R *best = nullptr;
    for (size_t i = 0; i < count; ++i) {
      if (ts[i] > eps_t() && (!best || ts[i] < *best)) {
        best = &ts[i];
      }
    }
    if (!best) {
      return std::nullopt;
    }
    return *best;
  }

  // wall_revolution.RevolutionWall (straight: wall_cylinder.CylinderWall)
  static bool inside(const Revolution &w, double xf, double yf, double zf) {
    double dx = xf - w.cxf, dy = yf - w.cyf;
    double r2 = w.c0f + zf * (w.c1f + zf * w.c2f);
    return dx * dx + dy * dy < r2 * (1.0 + 1e-9);
  }

  static OptHit hit(const Revolution &w, const V3 &O, const V3 &d, const R &) {
    R rx = O.x - w.cx, ry = O.y - w.cy;
    R A = d.x * d.x + d.y * d.y;
    R B = 2 * (rx * d.x + ry * d.y);
    R C = rx * rx + ry * ry - w.c0;
    if (!w.straight()) {
      A -= w.c2 * d.z * d.z;
      B -= d.z * (w.c1 + 2 * w.c2 * O.z);
      C -= O.z * (w.c1 + w.c2 * O.z);
    }
    std::array<R, 2> ts;
    size_t count = 0;
    if (std::fabs(to_double(A)) < w.eps) {
      if (std::fabs(to_double(B)) < w.eps) {
        return std::nullopt;
      }
      ts[0] = -C / B;
      count = 1;
    } else {
      R disc = B * B - 4 * A * C;
      if (disc < 0) {
        return std::nullopt;
      }
      R root = sqrt(disc);
      R twoA = 2 * A;
      ts[0] = (-B - root) / twoA;
      ts[1] = (-B + root) / twoA;
      count = 2;
    }
    std::optional<R> t = first_forward(ts, count);
    if (!t) {
      return std::nullopt;
    }
    V3 point = vadd(O, vscale(d, *t));
    V3 n{point.x - w.cx, point.y - w.cy, R()};
    if (!w.straight()) {
      n.z = -(w.c1 / 2 + w.c2 * point.z);
    }
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
    double rxf = to_double(O.x) - w.cxf, ryf = to_double(O.y) - w.cyf;
    double dxf = to_double(d.x), dyf = to_double(d.y);
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
    R t = (w.apothem - (mx * (O.x - w.cx) + my * (O.y - w.cy))) /
          (mx * d.x + my * d.y);
    return Hit{t, vadd(O, vscale(d, t)), {mx, my, R()}};
  }

  // wall_torus._quartic_first: smallest real root in (_EPS_T, t_capf].
  static std::optional<R> quartic_first(const std::array<R, 5> &c,
                                        double t_capf) {
    std::array<R, 4> dc = {c[0] * 4, c[1] * 3, c[2] * 2, c[3]};
    std::vector<double> cf(5);
    for (size_t i = 0; i < 5; ++i) {
      cf[i] = to_double(c[i]);
    }
    std::vector<double> cand;
    for (const std::complex<double> &r : dk_roots(cf)) {
      if (std::fabs(r.imag()) <= 1e-6 * std::max(1.0, std::fabs(r.real())) &&
          kEpsT / 2 < r.real() && r.real() <= t_capf) {
        cand.push_back(r.real());
      }
    }
    std::sort(cand.begin(), cand.end());
    std::optional<R> best;
    double bestf = 0.0;
    for (double seed : cand) {
      R t(seed);
      for (int i = 0; i < 12; ++i) {
        R gp = horner(dc, t);
        if (gp == 0) {
          break;
        }
        R step = horner(c, t) / gp;
        t -= step;
        double stepf = std::fabs(to_double(step));
        if (stepf <= 1e-24 * std::max(std::fabs(to_double(t)), 1e-9)) {
          double tf = to_double(t);
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
    R lo = eps_t();
    bool lneg = horner(c, lo) < 0;
    for (int k = 1; k < 49; ++k) {
      R hi(kEpsT * std::pow(t_capf / kEpsT, k / 48.0));
      bool hneg = horner(c, hi) < 0;
      if (lneg != hneg) {
        for (int i = 0; i < 90; ++i) {
          R mid = (lo + hi) / 2;
          if (lneg != (horner(c, mid) < 0)) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        return (lo + hi) / 2;
      }
      lo = hi;
      lneg = hneg;
    }
    return std::nullopt;
  }

  // wall_torus.TorusWall
  static bool inside(const Torus &w, double xf, double yf, double zf) {
    double wx = xf - w.cf[0], wy = yf - w.cf[1], wz = zf - w.cf[2];
    double s = wx * w.nf[0] + wy * w.nf[1];
    double m = wx * wx + wy * wy + wz * wz - s * s;
    double rho = std::sqrt(std::max(m, 0.0));
    double dr = rho - w.rf;
    return dr * dr + s * s < w.in2;
  }

  static OptHit hit(const Torus &w, const V3 &O, const V3 &d,
                    const R &t_exit) {
    V3 wv = vsub(O, w.C);
    R ws = vdot(wv, wv), wd = vdot(wv, d);
    R s0 = vdot(wv, w.nhat), sd = vdot(d, w.nhat);
    R u1 = 2 * wd, u0 = ws + w.K;
    std::array<R, 5> c = {R(1),
                          2 * u1,
                          u1 * u1 + 2 * u0 - w.fourR2 * (1 - sd * sd),
                          2 * u1 * u0 - 2 * w.fourR2 * (wd - s0 * sd),
                          u0 * u0 - w.fourR2 * (ws - s0 * s0)};
    std::optional<R> t =
        quartic_first(c, to_double(t_exit) * (1.0 + 1e-9) + kEpsT);
    if (!t) {
      return std::nullopt;
    }
    V3 point = vadd(O, vscale(d, *t));
    V3 w2 = vsub(point, w.C);
    R s = vdot(w2, w.nhat);
    V3 q = vsub(w2, vscale(w.nhat, s));
    R rho = vnorm(q);
    V3 n = vadd(vscale(q, (rho - w.bigR) / rho), vscale(w.nhat, s));
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

  struct Mirror {
    R z0, z1;
    double z0f, z1f;
  };
  struct Bundle {
    R z0, z1;
    double z0f, z1f;
    std::vector<Wall> walls;
  };
  using Optic = std::variant<Mirror, Bundle>;

  // surfaces.Mirror.next_event
  static Event next_event(const Mirror &m, const V3 &O, const V3 &d) {
    if (O.x <= 0 || d.x >= 0) {
      return {kExit, R(), {}, {}};
    }
    R t = -O.x / d.x;
    double zf = to_double(O.z + t * d.z);
    if (zf > m.z1f) {
      return {kExit, R(), {}, {}};
    }
    if (zf < m.z0f) {
      return {kAbsorb, (m.z0 - O.z) / d.z, {}, {}};
    }
    return {kReflect, t, vadd(O, vscale(d, t)), {R(1), R(), R()}};
  }

  static OptHit wall_hit(const Wall &w, const V3 &O, const V3 &d,
                         const R &t_exit) {
    return std::visit([&](const auto &ww) { return hit(ww, O, d, t_exit); },
                      w);
  }

  static bool wall_inside(const Wall &w, double xf, double yf, double zf) {
    return std::visit([&](const auto &ww) { return inside(ww, xf, yf, zf); },
                      w);
  }

  // surfaces.CapillaryBundle._locate
  static const Wall *locate(const Bundle &b, double xf, double yf,
                            double zf) {
    for (const Wall &w : b.walls) {
      if (wall_inside(w, xf, yf, zf)) {
        return &w;
      }
    }
    return nullptr;
  }

  // surfaces.CapillaryBundle.next_event
  static Event next_event(const Bundle &b, const V3 &O, const V3 &d) {
    if (d.z <= 0) {
      return {kAbsorb, R(), {}, {}};
    }
    double zf = to_double(O.z);
    if (zf < b.z0f - kEpsT) {
      return {kPass, (b.z0 - O.z) / d.z, {}, {}};
    }
    if (zf >= b.z1f - kEpsT) {
      return {kExit, R(), {}, {}};
    }
    const Wall *wall = locate(b, to_double(O.x), to_double(O.y), zf);
    if (!wall) {
      return {kAbsorb, R(), {}, {}};
    }
    R t_exit = (b.z1 - O.z) / d.z;
    OptHit h = wall_hit(*wall, O, d, t_exit);
    if (!h || t_exit <= h->t) {
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
    V3 direction;
  };

  static Result trace(const Optic *optic, V3 O, V3 d, const R &screen_z,
                      long max_bounces) {
    R opl;
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
          opl += ev.t;
          continue;
        }
        if (ev.kind == kAbsorb) {
          return {fAbsorbed, vadd(O, vscale(d, ev.t)), opl + ev.t,
                  std::move(refl), d};
        }
        opl += ev.t;
        R dot = vdot(d, ev.normal);
        d = vsub(d, vscale(ev.normal, 2 * dot));
        refl.push_back({ev.point, abs(dot)});
        O = ev.point;
      }
      if (!exited) {  // bounce budget spent: for-else in trace_ray
        return {fLost, O, opl, std::move(refl), d};
      }
    }
    if (d.z <= 0) {
      return {fLost, O, opl, std::move(refl), d};
    }
    R t = (screen_z - O.z) / d.z;
    if (t < 0) {
      return {fLost, O, opl, std::move(refl), d};
    }
    V3 point = vadd(O, vscale(d, t));
    return {fScreen, point, opl + t, std::move(refl), d};
  }
};

}  // namespace cstrace

#endif  // CS_TRACE_HPP
