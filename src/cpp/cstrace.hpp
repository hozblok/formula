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
#include <charconv>
#include <clocale>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <cseval/cseval.hpp>

namespace cstrace {

// strtod of a '.'-text under any LC_NUMERIC: when the locale stops the parse
// at the point, the point is replaced by the locale's decimal point.
inline double c_strtod(std::string s) {
  char *end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (*end != '.') {
    return v;
  }
  s.replace(static_cast<size_t>(end - s.c_str()), 1,
            std::localeconv()->decimal_point);
  return std::strtod(s.c_str(), nullptr);
}

// Forward-step floor for the ray parameter t (m); capsysred.types._EPS_T.
constexpr double kEpsT = 1e-12;
constexpr const char *kEpsTStr = "1e-12";

// Bore-membership nudge along the ray (m); capsysred.types._EPS_LOC.
constexpr const char *kEpsLocStr = "1e-7";

// On-wall inside() slack; capsysred.types._INSIDE_TOL. The probed point is a
// reflection point lying exactly ON the wall: after mp -> double conversion
// dx^2+dy^2 lands within a few ulp of a^2 and a strict < would randomly
// classify it outside (ray absorbed). 1e-9 dwarfs double roundoff yet is ~fm
// in radius for a um-scale bore, far below the kEpsLocStr nudge.
constexpr double kInsideTol = 1e-9;

// Widening of the root-search cap past t_exit; capsysred.types._TCAP_TOL.
// A hit within rounding of the exit plane must stay in the window; the
// caller's full-precision t_exit <= t comparison then makes the actual
// pass/reflect call, so the slack can only recover hits.
constexpr const char *kTcapTolStr = "1e-9";

// Newton polish stop, in digits: 10^-max(24, P/2) of the root;
// wall_torus._NEWTON_MIN_DIGITS. The final accepted step squares the error
// past 10^-P (quadratic convergence); targeting P/2 keeps the threshold ~P/2
// digits above the Horner roundoff floor ~10^-P, which a straight 10^-P
// target would chase forever (never firing — every ray would drop to the
// slow bisection fallback). The floor keeps the P=32 behavior.
constexpr unsigned kNewtonMinDigits = 24;

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

// wall_torus._float_seeds: plausible real roots from float Durand-Kerner,
// ascending. |Im| <= 1e-6*max(1, |Re|) keeps roots whose imaginary part is
// DK noise (a grazing ray's near-double pair) and drops genuinely complex
// pairs; (kEpsT/2, t_capf] keeps only the search window, the half-eps lower
// edge admitting a seed that float noise pushed slightly below the floor so
// Newton can polish it back above it.
inline std::vector<double> float_seeds(const std::vector<double> &cf,
                                       double t_capf) {
  std::vector<double> cand;
  for (const std::complex<double> &r : dk_roots(cf)) {
    if (std::fabs(r.imag()) <= 1e-6 * std::max(1.0, std::fabs(r.real())) &&
        kEpsT / 2 < r.real() && r.real() <= t_capf) {
      cand.push_back(r.real());
    }
  }
  std::sort(cand.begin(), cand.end());
  return cand;
}

template <unsigned P>
struct Tracer {
  using R = mp_real<P>;
  struct V3 {
    R x, y, z;
  };

  // float(x) of the bindings: the P-digit text, correctly rounded.
  static double to_double(const R &v) { return c_strtod(v.str(P)); }

  // nums.lift(value, p): the double's shortest round-trip decimal text, so both
  // twins place derived constants (search nodes) on the same mp values.
  static R from_double(double v) {
    char buf[32];
    auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return R(std::string(buf, res.ptr));
  }

  // types._EPS_T as an interned mp constant, from its decimal text.
  static const R &eps_t() {
    static const R v(kEpsTStr);
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
    std::vector<R> apothems;
    std::vector<double> af;
    R cx, cy;
    double cxf, cyf;
  };
  struct Torus {
    V3 C, nhat;
    R bigR, K, fourR2;
    double cf[3], nf[2], rf, in2;
  };
  struct Funnel {
    R cx, cy, r0, r02, ag, bg, af, bf, z0;
    double cxf, cyf, r0f, agf, bgf, aff, bff, z0f, in_mult;
  };
  using Wall = std::variant<Revolution, Polygon, Torus, Funnel>;

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
    return dx * dx + dy * dy < r2 * (1.0 + kInsideTol);
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
    for (size_t i = 0; i < w.facesf.size(); ++i) {
      double lim = w.af[i] * (1.0 + kInsideTol);
      if (!(w.facesf[i].first * dx + w.facesf[i].second * dy < lim)) {
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
      double tf = (w.af[i] - (mxf * rxf + myf * ryf)) / md;
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
    R t = (w.apothems[static_cast<size_t>(best)] -
           (mx * (O.x - w.cx) + my * (O.y - w.cy))) /
          (mx * d.x + my * d.y);
    return Hit{t, vadd(O, vscale(d, t)), {mx, my, R()}};
  }

  // 10^-max(kNewtonMinDigits, P/2) as an interned mp constant; the step is
  // compared against it in mp — at P ~ 1024 the step underflows double.
  static const R &newton_rtol() {
    static const R v =
        pow(R(10), -static_cast<int>(std::max(kNewtonMinDigits, P / 2)));
    return v;
  }

  // Scale floor (m) in that stop; wall_torus._NEWTON_TFLOOR: for roots below
  // ~1 nm a purely relative test keeps tightening toward zero and may never
  // fire; under the floor the threshold goes absolute.
  static const R &newton_tfloor() {
    static const R v("1e-9");
    return v;
  }

  // wall_torus._newton_polish: Newton on the exact quartic from a float
  // seed. Digits double per step, so 12 iterations lift a ~13-digit double
  // seed to ~13*2^12 digits — headroom for any supported P; the rtol stop
  // fires long before (e.g. P=64: |step| ~1e-13 -> 1e-26 -> 1e-52 fires
  // 1e-32, leaving error ~step^2 ~ 1e-104). nullopt means the seed stalled:
  // at a (near-)double root f' -> 0, steps stop shrinking and the budget
  // runs out — the caller falls back to exact bisection.
  static std::optional<R> newton_polish(const std::array<R, 5> &c,
                                        const std::array<R, 4> &dc,
                                        double seed) {
    R t(seed);
    for (int i = 0; i < 12; ++i) {
      R gp = horner(dc, t);
      if (gp == 0) {
        break;
      }
      R step = horner(c, t) / gp;
      t -= step;
      if (abs(step) <= newton_rtol() * std::max(abs(t), newton_tfloor())) {
        return t;
      }
    }
    return std::nullopt;
  }

  // wall_torus._bisect: the one sign change of the quartic in [lo, hi], lneg
  // telling p(lo) < 0: halves at decided signs, at most 8 + P*log2(10) times
  // (below 10^-P of the span), and stops at a midpoint of undecided sign.
  static R bisect(const std::array<R, 5> &c, R lo, bool lneg, R hi) {
    static const R half("0.5");
    static const int halvings = 8 + static_cast<int>(P * std::log2(10.0));
    for (int i = 0; i < halvings; ++i) {
      R mid = (lo + hi) * half;
      int s = decided_sign(c, mid);
      if (!s) {
        return mid;
      }
      if (lneg != (s < 0)) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    return (lo + hi) * half;
  }

  // wall_torus._taylor_shift: coefficients of p(x + s), leading first.
  static std::array<R, 5> taylor_shift(std::array<R, 5> d, const R &s) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 1; j < 5 - i; ++j) {
        d[j] = d[j] + s * d[j - 1];
      }
    }
    return d;
  }

  // Smallest supported precision >= 2P, where undecided signs and Descartes
  // counts are redone; 0 at the top.
  static constexpr unsigned kTwice = round_up_precision(2 * P);

  // wall_torus._bound: factor * u, u = 10^-(8*ceil(P/8)+8), the relative
  // error bound of one mp product, and of one sum against its larger operand.
  static R bound(int factor) {
    return R(std::to_string(factor) + "e-" +
             std::to_string(8 * ((P + 7) / 8) + 8));
  }

  // Horner's rule on |c_k| at |x|.
  static R magnitude(const std::array<R, 5> &c, const R &x) {
    R ax = abs(x);
    R v = abs(c[0]);
    for (size_t i = 1; i < 5; ++i) {
      v *= ax;
      v += abs(c[i]);
    }
    return v;
  }

  // wall_torus._sign: +1 or -1; 0 while |p(x)| <= 1024 u P(|x|).
  static int sign(const std::array<R, 5> &c, const R &x) {
    static const R slack = bound(1024);
    R v = horner(c, x);
    R e = slack * magnitude(c, x);
    return v > e ? 1 : (v < -e ? -1 : 0);
  }

  // wall_torus._descartes: sign variations of (1+y)^4 p(a + (b-a)/(1+y)),
  // a bound of the right parity on the open-interval roots in (a, b): 0 none,
  // 1 exactly one. nullopt while a coefficient is within
  // 512 u P(|a| + |b-a|/2); an exact zero sits inside that band — rounded
  // arithmetic cannot certify it, so it is undecided, never skipped.
  static std::optional<int> descartes(const std::array<R, 5> &c, const R &a,
                                      const R &b) {
    static const R slack = bound(512);
    static const R half("0.5");
    std::array<R, 5> d = taylor_shift(c, a);
    R w = b - a;
    R power = w;
    for (int j = 3; j >= 0; --j) {
      d[j] = d[j] * power;
      if (j) {
        power = power * w;
      }
    }
    std::array<R, 5> h = {d[4], d[3], d[2], d[1], d[0]};
    for (int i = 0; i < 4; ++i) {
      for (int j = 1; j < 5 - i; ++j) {
        h[j] = h[j] + h[j - 1];
      }
    }
    R e = slack * magnitude(c, abs(a) + abs(w) * half);
    for (const R &x : h) {
      if (abs(x) <= e) {
        return std::nullopt;
      }
    }
    int variations = 0;
    for (int i = 1; i < 5; ++i) {
      if ((h[i] < 0) != (h[i - 1] < 0)) {
        ++variations;
      }
    }
    return variations;
  }

  // wall_torus._decided: the sign and the Descartes count, redone on the same
  // values at precision kTwice while undecided at P.
  template <unsigned Q = kTwice>
  static std::array<mp_real<Q>, 5> lifted(const std::array<R, 5> &c) {
    return {mp_real<Q>(c[0]), mp_real<Q>(c[1]), mp_real<Q>(c[2]),
            mp_real<Q>(c[3]), mp_real<Q>(c[4])};
  }
  static int decided_sign(const std::array<R, 5> &c, const R &x) {
    int s = sign(c, x);
    if constexpr (kTwice != 0) {
      if (!s) {
        s = Tracer<kTwice>::sign(lifted(c), mp_real<kTwice>(x));
      }
    }
    return s;
  }
  static std::optional<int> decided_count(const std::array<R, 5> &c,
                                          const R &a, const R &b) {
    std::optional<int> k = descartes(c, a, b);
    if constexpr (kTwice != 0) {
      if (!k) {
        k = Tracer<kTwice>::descartes(lifted(c), mp_real<kTwice>(a),
                                      mp_real<kTwice>(b));
      }
    }
    return k;
  }

  // wall_torus._window_cap: t_exit (1 + _TCAP_TOL) + _EPS_T.
  static R window_cap(const R &t_exit) {
    static const R tol(kTcapTolStr);
    return t_exit + t_exit * tol + eps_t();
  }

  // wall_torus._cell_crossing: first sign change in (a, b] from the decided
  // signs at a and b (0 at an undecided window end). An interval Descartes
  // does not settle halves at its midpoint, or else a quarter point, of
  // decided sign. Where all three are undecided the span is a root cluster
  // below resolution and throws unresolved_root_error — also on opposite end
  // signs, which prove a crossing exists but not where within the span (the
  // strict refusal of 2026-11-05).
  static std::optional<R> cell_crossing(const std::array<R, 5> &c, const R &a,
                                        int sa, const R &b, int sb) {
    static const R half("0.5");
    struct Span {
      R a;
      int sa;
      R b;
      int sb;
    };
    std::vector<Span> stack{{a, sa, b, sb}};
    while (!stack.empty()) {
      Span s = std::move(stack.back());
      stack.pop_back();
      std::optional<int> k = decided_count(c, s.a, s.b);
      if (k == 0) {
        continue;
      }
      if (k == 1) {
        bool lneg = s.sa ? s.sa < 0 : (s.sb ? s.sb > 0 : horner(c, s.a) < 0);
        return bisect(c, s.a, lneg, s.b);
      }
      R mid = (s.a + s.b) * half;
      std::array<R, 3> xs = {mid, (s.a + mid) * half, (mid + s.b) * half};
      size_t pick = 3;
      int sx = 0;
      for (size_t n = 0; n < 3 && pick == 3; ++n) {
        if (s.a < xs[n] && xs[n] < s.b && (sx = decided_sign(c, xs[n])) != 0) {
          pick = n;
        }
      }
      if (pick == 3) {
        throw unresolved_root_error(
            "quartic root cluster in [" +
            s.a.str(20, std::ios_base::scientific) + ", " +
            s.b.str(20, std::ios_base::scientific) +
            "] cannot be resolved at precision " + std::to_string(P));
      }
      stack.push_back({xs[pick], sx, s.b, s.sb});
      stack.push_back({s.a, s.sa, xs[pick], sx});
    }
    return std::nullopt;
  }

  // wall_torus._first_crossing: first sign change in (kEpsT, cap], or in
  // (kEpsT, *limit]; t_capf is double(cap). A run of the 48 geometric cells
  // (the last node is cap) Descartes does not clear halves at a node of
  // decided sign, leftmost first; a cell, or a run without such a node, goes
  // to cell_crossing.
  static std::optional<R> first_crossing(const std::array<R, 5> &c,
                                         const R &cap, double t_capf,
                                         const R *limit) {
    std::vector<R> nodes{eps_t()};
    for (int k = 1; k < 49; ++k) {
      R node = k == 48 ? cap : from_double(kEpsT * std::pow(t_capf / kEpsT, k / 48.0));
      if (limit && node >= *limit) {
        nodes.push_back(*limit);
        break;
      }
      nodes.push_back(std::move(node));
    }
    std::vector<int> signs(nodes.size(), 2);
    auto sign_of = [&](size_t i) {
      if (signs[i] == 2) {
        signs[i] = decided_sign(c, nodes[i]);
      }
      return signs[i];
    };
    std::vector<std::pair<size_t, size_t>> stack{{0, nodes.size() - 1}};
    while (!stack.empty()) {
      auto [i, j] = stack.back();
      stack.pop_back();
      if (j > i + 1 && decided_count(c, nodes[i], nodes[j]) == 0) {
        continue;
      }
      size_t mid = (i + j) / 2, m = 0;
      for (size_t k : {mid, mid - 1, mid + 1}) {
        if (i < k && k < j && sign_of(k)) {
          m = k;
          break;
        }
      }
      if (!m) {
        std::optional<R> root =
            cell_crossing(c, nodes[i], sign_of(i), nodes[j], sign_of(j));
        if (root) {
          return root;
        }
        continue;
      }
      stack.push_back({m, j});
      stack.push_back({i, m});
    }
    return std::nullopt;
  }

  // wall_torus._quartic_first: smallest root in (_EPS_T, cap] where the
  // quartic changes sign. Float DK seeds polished by Newton are the fast path:
  // the smallest polished root counts when its rtol neighbours lie in the
  // window with decided signs that differ, and stands unless the window
  // before them holds a crossing (Descartes clears it, else the sign search
  // looks there); otherwise the sign search takes the whole window.
  static std::optional<R> quartic_first(const std::array<R, 5> &c,
                                        const R &cap) {
    std::array<R, 4> dc = {c[0] * 4, c[1] * 3, c[2] * 2, c[3]};
    // Seed material only, so Boost's faster conversion (an ulp off float() in
    // about one value per thousand).
    std::vector<double> cf(5);
    for (size_t i = 0; i < 5; ++i) {
      cf[i] = c[i].template convert_to<double>();
    }
    double t_capf = to_double(cap);
    std::optional<R> best;
    for (double seed : float_seeds(cf, t_capf)) {
      std::optional<R> t = newton_polish(c, dc, seed);
      if (t && *t > eps_t() && *t <= cap && (!best || *t < *best)) {
        best = std::move(t);
      }
    }
    if (best) {
      R delta = newton_rtol() * std::max(abs(*best), newton_tfloor());
      // the certifying bracket lies in the window: the upper probe never leaves it
      R lo = *best - delta, hi = std::min(*best + delta, cap);
      int s = lo > eps_t() ? decided_sign(c, lo) : 0;
      if (s && decided_sign(c, hi) == -s) {
        if (decided_count(c, eps_t(), lo) == 0) {
          return best;
        }
        std::optional<R> earlier = first_crossing(c, cap, t_capf, &lo);
        return earlier ? earlier : best;
      }
    }
    return first_crossing(c, cap, t_capf, nullptr);
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
        quartic_first(c, window_cap(t_exit));
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

  // wall_funnel.FunnelWall
  static bool inside(const Funnel &w, double xf, double yf, double zf) {
    double zr = zf - w.z0f;
    double gg = 1.0 + zr * (w.agf + zr * w.bgf);
    double ff = 1.0 + zr * (w.aff + zr * w.bff);
    double u = xf - w.cxf * gg;
    double v = yf - w.cyf * gg;
    double r = w.r0f * ff;
    return u * u + v * v < r * r * w.in_mult;
  }

  static OptHit hit(const Funnel &w, const V3 &O, const V3 &d,
                    const R &t_exit) {
    R zr0 = O.z - w.z0;
    R G0 = 1 + zr0 * (w.ag + zr0 * w.bg);
    R G1 = (w.ag + 2 * w.bg * zr0) * d.z;
    R G2 = w.bg * d.z * d.z;
    R F0 = 1 + zr0 * (w.af + zr0 * w.bf);
    R F1 = (w.af + 2 * w.bf * zr0) * d.z;
    R F2 = w.bf * d.z * d.z;
    R u0 = O.x - w.cx * G0, u1 = d.x - w.cx * G1, u2 = -(w.cx * G2);
    R v0 = O.y - w.cy * G0, v1 = d.y - w.cy * G1, v2 = -(w.cy * G2);
    R h0 = w.r0 * F0, h1 = w.r0 * F1, h2 = w.r0 * F2;
    std::array<R, 5> c = {u2 * u2 + v2 * v2 - h2 * h2,
                          2 * (u1 * u2 + v1 * v2 - h1 * h2),
                          u1 * u1 + v1 * v1 - h1 * h1 +
                              2 * (u0 * u2 + v0 * v2 - h0 * h2),
                          2 * (u0 * u1 + v0 * v1 - h0 * h1),
                          u0 * u0 + v0 * v0 - h0 * h0};
    // wall_funnel._poly_first: exact-zero leading coefficients (structural
    // degeneracies) shift out, the tail pads with zeros — q(t)*t^k keeps
    // every t>0 root and the padded t=0 roots sit below kEpsT.
    int lead = 0;
    while (lead < 4 && c[lead] == 0) {
      ++lead;
    }
    std::array<R, 5> m;
    for (int i = lead; i < 5; ++i) {
      m[i - lead] = c[i] / c[lead];
    }
    for (int i = 5 - lead; i < 5; ++i) {
      m[i] = 0;
    }
    std::optional<R> t =
        quartic_first(m, window_cap(t_exit));
    if (!t) {
      return std::nullopt;
    }
    V3 point = vadd(O, vscale(d, *t));
    R zr = point.z - w.z0;
    R gg = 1 + zr * (w.ag + zr * w.bg);
    R gp = w.ag + 2 * w.bg * zr;
    R ff = 1 + zr * (w.af + zr * w.bf);
    R fp = w.af + 2 * w.bf * zr;
    R u = point.x - w.cx * gg;
    R v = point.y - w.cy * gg;
    V3 n = {u, v, -((u * w.cx + v * w.cy) * gp + w.r02 * ff * fp)};
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

  // surfaces.Mirror.next_event: the hit's z against the span, in mp.
  static Event next_event(const Mirror &m, const V3 &O, const V3 &d) {
    if (O.x <= 0 || d.x >= 0) {
      return {kExit, R(), {}, {}};
    }
    R t = -O.x / d.x;
    R z = O.z + t * d.z;
    if (z > m.z1) {
      return {kExit, R(), {}, {}};
    }
    if (z < m.z0) {
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

  // surfaces.CapillaryBundle._locate: the point nudged along d, in mp and then
  // as doubles; the nudge disambiguates bore-bore tangency points (a
  // reflection there is on two walls at once).
  static const Wall *locate(const Bundle &b, const V3 &O, const V3 &d) {
    static const R eps_loc(kEpsLocStr);
    double xf = to_double(O.x + eps_loc * d.x);
    double yf = to_double(O.y + eps_loc * d.y);
    double zf = to_double(O.z + eps_loc * d.z);
    for (const Wall &w : b.walls) {
      if (wall_inside(w, xf, yf, zf)) {
        return &w;
      }
    }
    return nullptr;
  }

  // surfaces.CapillaryBundle.next_event: z against z0, z1 less kEpsT, in mp.
  static Event next_event(const Bundle &b, const V3 &O, const V3 &d) {
    if (d.z <= 0) {
      return {kAbsorb, R(), {}, {}};
    }
    if (O.z < b.z0 - eps_t()) {
      return {kPass, (b.z0 - O.z) / d.z, {}, {}};
    }
    if (O.z >= b.z1 - eps_t()) {
      return {kExit, R(), {}, {}};
    }
    const Wall *wall = locate(b, O, d);
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
    // trace.trace_ray: a non-finite input component is refused.
    const R *inputs[] = {&O.x, &O.y, &O.z, &d.x, &d.y, &d.z, &screen_z};
    for (const R *v : inputs) {
      if (!(boost::multiprecision::isfinite)(*v)) {
        throw std::invalid_argument("non-finite ray input");
      }
    }
    R opl;
    std::vector<Refl> refl;
    if (optic) {
      for (;;) {
        Event ev = next_event(*optic, O, d);
        if (ev.kind == kExit) {
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
        if (static_cast<long>(refl.size()) >= max_bounces) {
          // reflection budget spent at this wall hit
          return {fLost, ev.point, opl + ev.t, std::move(refl), d};
        }
        opl += ev.t;
        R dot = vdot(d, ev.normal);
        d = vsub(d, vscale(ev.normal, 2 * dot));
        refl.push_back({ev.point, abs(dot)});
        O = ev.point;
      }
    }
    if (d.z <= 0) {
      return {fLost, O, opl, std::move(refl), d};
    }
    R t = (screen_z - O.z) / d.z;
    if (t < 0) {
      // Treat backward steps within the tolerance as arrival, keeping O and opl.
      static const R screen_eps_t(kEpsTStr);
      if (t < -screen_eps_t) {
        return {fLost, O, opl, std::move(refl), d};
      }
      t = R();
    }
    V3 point = vadd(O, vscale(d, t));
    return {fScreen, point, opl + t, std::move(refl), d};
  }
};

}  // namespace cstrace

#endif  // CS_TRACE_HPP
