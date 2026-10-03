// Native CAPSYSred tracer bindings: NativeOptic twins of the Python optics,
// the trace/next_event/hit entry points, and debug hooks for the root-finder
// parity tests. Own translation unit to keep per-compile memory bounded.
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cseval/cseval.hpp>

#include "cstrace.hpp"

namespace py = pybind11;

namespace {

// Type-erased per-precision Tracer<P>::Optic; `precision` selects P back.
struct NativeOptic {
  unsigned precision = 0;
  std::string kind;
  std::shared_ptr<void> impl;
};

template <typename F, unsigned... Ps>
bool dispatch_ps(unsigned p, F &&f, std::integer_sequence<unsigned, Ps...>) {
  return (((p == Ps) ? (f(std::integral_constant<unsigned, Ps>{}), true)
                     : false) ||
          ...);
}

template <typename F>
void dispatch(unsigned p, F &&f) {
  if (!dispatch_ps(p, std::forward<F>(f), AllowedPrecisionsSeq{})) {
    throw std::invalid_argument("unsupported precision: " + std::to_string(p));
  }
}

template <unsigned... Ps>
unsigned precision_of_impl(py::handle h,
                           std::integer_sequence<unsigned, Ps...>) {
  unsigned out = 0;
  ((py::isinstance<mp_real<Ps>>(h) ? (out = Ps, true) : false) || ...);
  return out;
}

unsigned precision_of(py::handle h) {
  unsigned p = precision_of_impl(h, AllowedPrecisionsSeq{});
  if (!p) {
    throw std::invalid_argument("expected an mp_real_<P> value");
  }
  return p;
}

template <unsigned P>
typename cstrace::Tracer<P>::V3 to_v3(py::sequence seq) {
  return {py::cast<mp_real<P>>(seq[0]), py::cast<mp_real<P>>(seq[1]),
          py::cast<mp_real<P>>(seq[2])};
}

template <unsigned P>
py::tuple from_v3(const typename cstrace::Tracer<P>::V3 &v) {
  return py::make_tuple(py::cast(v.x), py::cast(v.y), py::cast(v.z));
}

// Wall spec: (kind, mp_values, float_values); layouts documented in
// capsysred/native.py next to the producer.
template <unsigned P>
typename cstrace::Tracer<P>::Wall make_wall(py::sequence spec) {
  using T = cstrace::Tracer<P>;
  const auto kind = py::cast<std::string>(spec[0]);
  auto mp = py::cast<py::sequence>(spec[1]);
  auto fl = py::cast<py::sequence>(spec[2]);
  auto M = [&](size_t i) { return py::cast<mp_real<P>>(mp[i]); };
  auto F = [&](size_t i) { return py::cast<double>(fl[i]); };
  if (kind == "revolution") {
    return typename T::Revolution{M(0), M(1), M(2), M(3), M(4),
                                  F(0), F(1), F(2), F(3), F(4), F(5)};
  }
  if (kind == "polygon") {
    typename T::Polygon w{{}, {}, M(0), M(1), M(2), F(0), F(1), F(2)};
    for (size_t i = 3; i + 1 < mp.size(); i += 2) {
      w.faces.emplace_back(M(i), M(i + 1));
    }
    for (size_t i = 3; i + 1 < fl.size(); i += 2) {
      w.facesf.emplace_back(F(i), F(i + 1));
    }
    if (w.faces.size() != w.facesf.size()) {
      throw std::invalid_argument("polygon mp/float face counts differ");
    }
    return w;
  }
  if (kind == "torus") {
    return typename T::Torus{{M(0), M(1), M(2)},
                             {M(3), M(4), M(5)},
                             M(6),
                             M(7),
                             M(8),
                             {F(0), F(1), F(2)},
                             {F(3), F(4)},
                             F(5),
                             F(6)};
  }
  if (kind == "funnel") {
    return typename T::Funnel{M(0), M(1), M(2), M(3), M(4), M(5), M(6),
                              M(7), M(8), F(0), F(1), F(2), F(3), F(4),
                              F(5), F(6), F(7), F(8)};
  }
  throw std::invalid_argument("unknown wall kind: " + kind);
}

template <unsigned P>
const typename cstrace::Tracer<P>::Optic *optic_of(const NativeOptic &nat) {
  return static_cast<const typename cstrace::Tracer<P>::Optic *>(
      nat.impl.get());
}

template <unsigned P>
const typename cstrace::Tracer<P>::Bundle &bundle_of(const NativeOptic &nat) {
  const auto *b = std::get_if<typename cstrace::Tracer<P>::Bundle>(
      optic_of<P>(nat));
  if (!b) {
    throw std::invalid_argument("optic is not a bundle");
  }
  return *b;
}

template <typename T>
py::bytes as_bytes(const std::vector<T> &v) {
  return py::bytes(reinterpret_cast<const char *>(v.data()),
                   v.size() * sizeof(T));
}

// One pixel of the delete-one-mode jackknife, shared by BeamletGrid::jackknife
// (rows retained in C++) and jackknife_tile (rows from the disk store):
// mu = min(|W|/sqrt(I*I_ref), 1); LOO over the modes whose removal leaves
// i_s > eps*I and iref_s > eps*I_ref; sigma from the centered LOO values;
// the don't-trust flag as in the Python fallback. row_w(s) -> const float*
// (re, im), row_i(s) -> float. With a boundary pointer, the pixel is marked
// when some mode's mask quantity lies within band_rel*I (resp. band_rel*I_ref)
// of its threshold: pixels whose LOO membership is rounding-sensitive.
template <typename RowW, typename RowI>
inline void jackknife_pixel(std::complex<double> w, double i_pix, double i_ref,
                            size_t nf, RowW row_w, RowI row_i,
                            const double *irefs, double eps_rel,
                            std::vector<double> &loo, double &mu, double &err,
                            unsigned char &dub, double band_rel = 0.0,
                            unsigned char *boundary = nullptr) {
  mu = std::min(std::abs(w) / std::sqrt(i_pix * i_ref), 1.0);
  const double eps_i = eps_rel * i_pix, eps_r = eps_rel * i_ref;
  loo.clear();
  for (size_t s = 0; s < nf; ++s) {
    const double i_s = i_pix - row_i(s);
    const double iref_s = i_ref - irefs[s];
    if (boundary && (std::fabs(i_s - eps_i) <= band_rel * i_pix ||
                     std::fabs(iref_s - eps_r) <= band_rel * i_ref)) {
      *boundary = 1;
    }
    if (i_s > eps_i && iref_s > eps_r) {
      const float *wr = row_w(s);
      const double dr = w.real() - wr[0];
      const double di = w.imag() - wr[1];
      loo.push_back(
          std::min(std::hypot(dr, di) / std::sqrt(i_s * iref_s), 1.0));
    }
  }
  err = 0.0;
  if (loo.size() > 1) {
    double mean = 0.0;
    for (const double v : loo) {
      mean += v;
    }
    mean /= double(loo.size());
    double ss = 0.0;
    for (const double v : loo) {
      ss += (v - mean) * (v - mean);
    }
    err = std::sqrt(ss * double(loo.size() - 1) / double(loo.size()));
  }
  dub = (err > 1.0 || loo.size() < 2 ||
         ((mu >= 1.0 - eps_rel || mu == 0.0) && err <= eps_rel))
            ? 1
            : 0;
}

// Stage-11 beamlet deposit: the hot window loop of BeamletField.add_ray,
// pure double (the beamlet field is float64 physics by design — no mp_real,
// no precision dispatch). One dense complex grid per spectral line.
class BeamletGrid {
 public:
  BeamletGrid(long nx, long ny, double x0, double y0, double ex, double ey,
              std::vector<double> kms, std::vector<double> zrs,
              std::vector<double> zrs_t, double ns)
      : nx_(nx),
        ny_(ny),
        x0_(x0),
        y0_(y0),
        ex_(ex),
        ey_(ey),
        dx_(ex / nx),
        dy_(ey / ny),
        ns_(ns),
        kms_(std::move(kms)),
        zrs_(std::move(zrs)),
        zrs_t_(std::move(zrs_t)),
        g_(kms_.size(), std::vector<std::complex<double>>(size_t(nx) * ny)),
        W_(size_t(nx) * ny),
        I_(size_t(nx) * ny) {}

  // Per-mode reset; the W/I fold totals persist across modes.
  void clear() {
    for (auto &g : g_) {
      std::fill(g.begin(), g.end(), std::complex<double>(0.0, 0.0));
    }
  }

  // Python-loop twin (pixel centers included): envelope
  // exp((k/2)*d^T Im(G) d), phase tilt + (k/2)*d^T Re(G) d. The complex
  // exponent is quadratic along a row, so the sweep is a geometric-ratio
  // recurrence — two complex mults per pixel instead of an exp; the
  // multiplicative drift stays ~n_row*eps, far under the deposit noise.
  void add(size_t m, double x, double y, std::complex<double> pref, double tx,
           double ty, std::complex<double> hxx, std::complex<double> hxy,
           std::complex<double> hyy, double rx, double ry) {
    // clamp in double: a blown-up spot overflows a 32-bit long
    const double fx_lo = std::floor((x - rx - x0_) / dx_);
    const double fx_hi = std::floor((x + rx - x0_) / dx_);
    const double fy_lo = std::floor((y - ry - y0_) / dy_);
    const double fy_hi = std::floor((y + ry - y0_) / dy_);
    if (!(fx_hi >= 0.0 && fx_lo <= double(nx_ - 1) && fy_hi >= 0.0 &&
          fy_lo <= double(ny_ - 1))) {
      return;
    }
    const long ix_lo = fx_lo < 0.0 ? 0L : long(fx_lo);
    const long ix_hi = fx_hi > double(nx_ - 1) ? nx_ - 1 : long(fx_hi);
    const long iy_lo = fy_lo < 0.0 ? 0L : long(fy_lo);
    const long iy_hi = fy_hi > double(ny_ - 1) ? ny_ - 1 : long(fy_hi);
    auto &g = g_.at(m);
    // exp arg as an analytic function of t = dx_off: c2*t^2 + b*t + a
    // with c2 = i*conj(hxx) (complex(quad.imag, quad.real) == i*conj(quad))
    const std::complex<double> im(0.0, 1.0);
    const double h = dx_;
    const std::complex<double> c2 = im * std::conj(hxx);
    const std::complex<double> M = std::exp(2.0 * c2 * (h * h));
    for (long iy = iy_lo; iy <= iy_hi; ++iy) {
      const double dy_off = y0_ + (iy + 0.5) * ey_ / ny_ - y;
      const std::complex<double> b =
          im * (std::conj(2.0 * hxy * dy_off) + tx);
      const std::complex<double> a =
          im * (std::conj(hyy * (dy_off * dy_off)) + ty * dy_off);
      // sweep outward from the row's envelope crest: every ratio then
      // stays <= 1 in magnitude (an edge start overflows on tail rays)
      const double vp =
          hxx.imag() == 0.0 ? 0.0 : -hxy.imag() * dy_off / hxx.imag();
      const long ip = long(std::max(
          double(ix_lo),
          std::min(double(ix_hi), std::floor((vp + x - x0_) / dx_))));
      const double tp = x0_ + (ip + 0.5) * ex_ / nx_ - x;
      const std::complex<double> v0 =
          pref * std::exp(a + b * tp + c2 * (tp * tp));
      std::complex<double> *row = g.data() + size_t(iy) * nx_;
      std::complex<double> v = v0;
      std::complex<double> K =
          std::exp(b * h + c2 * (h * h) + 2.0 * c2 * (h * tp));
      for (long ix = ip; ix <= ix_hi; ++ix) {
        row[ix] += v;
        v *= K;
        K *= M;
      }
      v = v0;
      K = std::exp(-b * h + c2 * (h * h) - 2.0 * c2 * (h * tp));
      for (long ix = ip - 1; ix >= ix_lo; --ix) {
        v *= K;
        K *= M;
        row[ix] += v;
      }
    }
  }

  // Whole-ray deposit: gamma.propagate + the spot per line, one call per
  // ray. lenses is flat [(phi, pxx, pxy, pyy), ...]; returns (spot width of
  // line 0, geometric mean of the axes; -1 when skipped) and the number of
  // lines whose Im(G) lost negative-definiteness (not deposited).
  py::tuple add_ray(double x, double y, double dxf, double dyf, double opl,
                    double psi, const std::vector<double> &segs,
                    const std::vector<double> &lenses,
                    const std::vector<std::complex<double>> &amps) {
    const size_t n_lens = lenses.size() / 4;
    if (segs.empty() || lenses.size() != 4 * (segs.size() - 1)) {
      throw std::invalid_argument("add_ray: lenses must be 4*(len(segs)-1)");
    }
    double w_spot = -1.0;
    long bad = 0;
    for (size_t m = 0; m < kms_.size(); ++m) {
      const double km = kms_[m];
      // gamma.propagate, op-for-op (adaptive sub-steps, principal sqrt);
      // elliptic launch: tangential axis at azimuth psi
      const double zrt = zrs_t_[m], zrsm = zrs_[m];
      std::complex<double> qxx, qxy, qyy;
      if (zrt == zrsm) {
        qxx = std::complex<double>(0.0, zrt);
        qxy = std::complex<double>(0.0, 0.0);
        qyy = std::complex<double>(0.0, zrt);
      } else {
        const double c = std::cos(psi), sn = std::sin(psi);
        qxx = std::complex<double>(0.0, zrt * c * c + zrsm * sn * sn);
        qxy = std::complex<double>(0.0, (zrt - zrsm) * c * sn);
        qyy = std::complex<double>(0.0, zrt * sn * sn + zrsm * c * c);
      }
      std::complex<double> a_geo(1.0, 0.0);
      for (size_t j = 0; j < segs.size(); ++j) {
        // closed-form drift amplitude (gamma.propagate twin): factor
        // det(Q + s) through its roots, principal sqrt per linear factor
        const std::complex<double> tr = qxx + qyy;
        const std::complex<double> det0 = qxx * qyy - qxy * qxy;
        const std::complex<double> disc = std::sqrt(tr * tr - 4.0 * det0);
        const std::complex<double> roots[2] = {0.5 * (disc - tr),
                                               -0.5 * (disc + tr)};
        for (const std::complex<double> &root : roots) {
          if (root == 0.0) {
            a_geo = 0.0;
          } else {
            a_geo *= 1.0 / std::sqrt(1.0 - segs[j] / root);
          }
        }
        qxx += segs[j];
        qyy += segs[j];
        if (j < n_lens) {
          // gamma.reflect twin: Gamma -= P, then the mirror M Q M; phi NaN
          // = unknown wall normal (no flip, no lens)
          const double phi = lenses[4 * j], pxx = lenses[4 * j + 1],
                       pxy = lenses[4 * j + 2], pyy = lenses[4 * j + 3];
          if (std::isnan(phi)) {
            continue;
          }
          if (pxx != 0.0 || pxy != 0.0 || pyy != 0.0) {
            std::complex<double> det = qxx * qyy - qxy * qxy;
            const std::complex<double> gxx = qyy / det - pxx;
            const std::complex<double> gxy = -qxy / det - pxy;
            const std::complex<double> gyy = qxx / det - pyy;
            det = gxx * gyy - gxy * gxy;
            qxx = gyy / det;
            qxy = -gxy / det;
            qyy = gxx / det;
          }
          const double c = std::cos(phi), sn = std::sin(phi);
          const double c2 = c * c - sn * sn, s2 = 2.0 * c * sn;
          const std::complex<double> fxx =
              c2 * c2 * qxx + 2.0 * c2 * s2 * qxy + s2 * s2 * qyy;
          const std::complex<double> fxy =
              c2 * s2 * (qxx - qyy) + (s2 * s2 - c2 * c2) * qxy;
          const std::complex<double> fyy =
              s2 * s2 * qxx - 2.0 * c2 * s2 * qxy + c2 * c2 * qyy;
          qxx = fxx;
          qxy = fxy;
          qyy = fyy;
        }
      }
      const std::complex<double> det = qxx * qyy - qxy * qxy;
      const std::complex<double> gxx = qyy / det;
      const std::complex<double> gxy = -qxy / det;
      const std::complex<double> gyy = qxx / det;
      const double mean = 0.5 * (gxx.imag() + gyy.imag());
      const double dev =
          std::hypot(0.5 * (gxx.imag() - gyy.imag()), gxy.imag());
      if (mean + dev >= 0.0) {   // beam blew up: no Gaussian to deposit
        ++bad;
        continue;
      }
      const double w_hi = std::sqrt(-2.0 / (km * (mean + dev)));
      if (m == 0) {
        const double w_lo = std::sqrt(-2.0 / (km * (mean - dev)));
        w_spot = std::sqrt(w_hi * w_lo);
      }
      const std::complex<double> pref =
          amps[m] * std::conj(a_geo) *
          std::exp(std::complex<double>(0.0, km * opl));
      // per-axis bounding box of the ns-sigma ellipse (det > 0 here)
      const double det_gi =
          gxx.imag() * gyy.imag() - gxy.imag() * gxy.imag();
      const double rx = ns_ * std::sqrt(-2.0 * gyy.imag() / (km * det_gi));
      const double ry = ns_ * std::sqrt(-2.0 * gxx.imag() / (km * det_gi));
      add(m, x, y, pref, km * dxf, km * dyf, 0.5 * km * gxx, 0.5 * km * gxy,
          0.5 * km * gyy, rx, ry);
    }
    return py::make_tuple(w_spot, bad);
  }

  // One mode's fold: W/I totals += wf*g*conj(g_ref) and wf*|g|^2 per line.
  // The float32 delete-one rows stay inside (jackknife runs native too);
  // returns the float32-consistent ref intensity of the mode.
  double fold(const std::vector<double> &wfs, long ref) {
    const size_t npix = size_t(nx_) * ny_;
    std::vector<float> w_row(2 * npix, 0.0f);
    std::vector<float> i_row(npix, 0.0f);
    const std::complex<double> zero(0.0, 0.0);
    for (size_t m = 0; m < g_.size(); ++m) {
      const double wf = wfs.at(m);
      const auto &g = g_.at(m);
      const std::complex<double> g_ref = g.at(size_t(ref));
      const bool has_ref = g_ref != zero;
      const std::complex<double> ref_c = std::conj(g_ref);
      for (size_t p = 0; p < npix; ++p) {
        const std::complex<double> v = g[p];
        if (v == zero) {
          continue;
        }
        const double a2 = wf * std::norm(v);
        I_[p] += a2;
        i_row[p] += a2;
        if (has_ref) {
          const std::complex<double> cross = v * ref_c;
          W_[p] += wf * cross;
          w_row[2 * p] += wf * cross.real();
          w_row[2 * p + 1] += wf * cross.imag();
        }
      }
    }
    const double i_ref = i_row[size_t(ref)];
    fw_.push_back(std::move(w_row));
    fi_.push_back(std::move(i_row));
    irefs_.push_back(i_ref);
    return i_ref;
  }

  // One mode's export for the shared any-jobs path: the float32 delete-one
  // rows exactly as fold builds them, the mode's float64 W/I contributions
  // (left fold over the lines, from zero) and the float32 ref intensity.
  // Nothing is retained; the totals stay untouched.
  py::tuple fold_export(const std::vector<double> &wfs, long ref) const {
    const size_t npix = size_t(nx_) * ny_;
    std::vector<float> w_row(2 * npix, 0.0f);
    std::vector<float> i_row(npix, 0.0f);
    std::vector<std::complex<double>> dw(npix);
    std::vector<double> di(npix, 0.0);
    const std::complex<double> zero(0.0, 0.0);
    for (size_t m = 0; m < g_.size(); ++m) {
      const double wf = wfs.at(m);
      const auto &g = g_.at(m);
      const std::complex<double> g_ref = g.at(size_t(ref));
      const bool has_ref = g_ref != zero;
      const std::complex<double> ref_c = std::conj(g_ref);
      for (size_t p = 0; p < npix; ++p) {
        const std::complex<double> v = g[p];
        if (v == zero) {
          continue;
        }
        const double a2 = wf * std::norm(v);
        di[p] += a2;
        i_row[p] += a2;
        if (has_ref) {
          const std::complex<double> cross = v * ref_c;
          dw[p] += wf * cross;
          w_row[2 * p] += wf * cross.real();
          w_row[2 * p + 1] += wf * cross.imag();
        }
      }
    }
    const double i_ref = i_row[size_t(ref)];
    return py::make_tuple(as_bytes(w_row), as_bytes(i_row), as_bytes(dw),
                          as_bytes(di), i_ref);
  }

  // Delete-one-mode jackknife over the fold rows, mirroring the Python
  // fallback op-for-op: dense row-major (mu float64, sigma float64,
  // don't-trust uint8) as bytes.
  py::tuple jackknife(long ref, double eps_rel) const {
    const size_t npix = size_t(nx_) * ny_;
    std::vector<double> mu(npix, 0.0), err(npix, 0.0);
    std::vector<unsigned char> dub(npix, 0);
    const double i_ref = I_[size_t(ref)];
    const size_t nf = fi_.size();
    std::vector<double> loo;
    loo.reserve(nf);
    if (!(i_ref > 0.0)) {
      // unlit reference: no pixel is estimable
      for (size_t p = 0; p < npix; ++p) {
        dub[p] = I_[p] > 0.0 ? 1 : 0;
      }
    } else {
      for (size_t p = 0; p < npix; ++p) {
        const double i_pix = I_[p];
        if (i_pix <= 0.0) {
          continue;
        }
        jackknife_pixel(
            W_[p], i_pix, i_ref, nf,
            [&](size_t s) { return fw_[s].data() + 2 * p; },
            [&](size_t s) { return fi_[s][p]; }, irefs_.data(), eps_rel, loo,
            mu[p], err[p], dub[p]);
      }
    }
    return py::make_tuple(as_bytes(mu), as_bytes(err), as_bytes(dub));
  }

  // The run's W (complex128 interleaved) and I (float64) fold totals.
  py::tuple totals() const {
    return py::make_tuple(
        py::bytes(reinterpret_cast<const char *>(W_.data()),
                  W_.size() * sizeof(std::complex<double>)),
        py::bytes(reinterpret_cast<const char *>(I_.data()),
                  I_.size() * sizeof(double)));
  }

  std::complex<double> at(size_t m, long pixel) const {
    return g_.at(m).at(size_t(pixel));
  }

  py::list items(size_t m) const {
    py::list out;
    const auto &g = g_.at(m);
    const std::complex<double> zero(0.0, 0.0);
    for (size_t p = 0; p < g.size(); ++p) {
      if (g[p] != zero) {
        out.append(py::make_tuple(long(p), g[p]));
      }
    }
    return out;
  }

 private:
  long nx_, ny_;
  double x0_, y0_, ex_, ey_, dx_, dy_, ns_;
  std::vector<double> kms_, zrs_, zrs_t_;
  std::vector<std::vector<std::complex<double>>> g_;
  std::vector<std::complex<double>> W_;
  std::vector<double> I_;
  std::vector<std::vector<float>> fw_, fi_;   // per-fold delete-one rows
  std::vector<double> irefs_;
};

// Exact, order-free totals of the shared Stage-11 path: per pixel and
// component (Re W, Im W, I) a 2176-bit two's-complement fixed-point integer
// in units of 2^-1074 (34 words: every binary position of a finite double
// plus 64 guard bits for up to 2^64 terms). Every finite double is added
// exactly; rounding to the nearest double (ties to even) happens once, in
// totals(). The density is an exact integer count.
class ExactAccumulator {
 public:
  static constexpr size_t kWords = 34;

  explicit ExactAccumulator(size_t npix)
      : npix_(npix), acc_(3 * npix * kWords, 0), density_(npix, 0),
        n_terms_(0) {}

  // dw: complex128[npix], di: float64[npix], density: uint32[npix].
  void add(py::buffer dw, py::buffer di, py::buffer density) {
    const py::buffer_info bw = dw.request(), bi = di.request(),
                          bd = density.request();
    if (size_t(bw.size) * size_t(bw.itemsize) != 16 * npix_ ||
        size_t(bi.size) * size_t(bi.itemsize) != 8 * npix_ ||
        size_t(bd.size) * size_t(bd.itemsize) != 4 * npix_) {
      throw std::invalid_argument(
          "ExactAccumulator.add: buffer sizes do not match the pixel count");
    }
    if (n_terms_ == UINT64_MAX) {
      throw std::overflow_error("ExactAccumulator: term counter exhausted");
    }
    const double *w = static_cast<const double *>(bw.ptr);
    const double *i = static_cast<const double *>(bi.ptr);
    const uint32_t *d = static_cast<const uint32_t *>(bd.ptr);
    for (size_t p = 0; p < npix_; ++p) {
      add_double(words(p, 0), w[2 * p]);
      add_double(words(p, 1), w[2 * p + 1]);
      add_double(words(p, 2), i[p]);
      density_[p] += d[p];
    }
    ++n_terms_;
  }

  // W (complex128) and I (float64), each the correctly rounded exact sum.
  py::tuple totals() const {
    std::vector<std::complex<double>> W(npix_);
    std::vector<double> I(npix_);
    for (size_t p = 0; p < npix_; ++p) {
      W[p] = std::complex<double>(round_to_double(words(p, 0)),
                                  round_to_double(words(p, 1)));
      I[p] = round_to_double(words(p, 2));
    }
    return py::make_tuple(as_bytes(W), as_bytes(I));
  }

  py::bytes density() const { return as_bytes(density_); }   // uint64[npix]
  unsigned long long n_terms() const { return n_terms_; }
  size_t npix() const { return npix_; }

 private:
  uint64_t *words(size_t p, size_t c) {
    return acc_.data() + (3 * p + c) * kWords;
  }
  const uint64_t *words(size_t p, size_t c) const {
    return acc_.data() + (3 * p + c) * kWords;
  }

  // x = m * 2^(shift - 1074) with m the 53-bit significand; add or subtract
  // m << shift into the little-endian word array (carry/borrow out of the
  // top word is the two's-complement wrap of a signed result).
  static void add_double(uint64_t *a, double x) {
    if (x == 0.0) {
      return;
    }
    if (!std::isfinite(x)) {
      throw std::invalid_argument(
          "ExactAccumulator: non-finite contribution");
    }
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    const bool neg = (bits >> 63) != 0;
    const unsigned e = unsigned((bits >> 52) & 0x7FFu);
    uint64_t m = bits & ((uint64_t(1) << 52) - 1);
    unsigned shift = 0;
    if (e != 0) {
      m |= uint64_t(1) << 52;
      shift = e - 1;
    }
    const size_t k = shift / 64;
    const unsigned b = shift % 64;
    const uint64_t lo = m << b;
    const uint64_t hi = b ? (m >> (64 - b)) : 0;
    if (!neg) {
      uint64_t s = a[k] + lo;
      uint64_t carry = s < lo ? 1 : 0;
      a[k] = s;
      size_t j = k + 1;
      const uint64_t t = hi + carry;
      s = a[j] + t;
      carry = s < t ? 1 : 0;
      a[j] = s;
      for (++j; carry && j < kWords; ++j) {
        a[j] += 1;
        carry = a[j] == 0 ? 1 : 0;
      }
    } else {
      uint64_t borrow = a[k] < lo ? 1 : 0;
      a[k] -= lo;
      size_t j = k + 1;
      const uint64_t t = hi + borrow;
      borrow = a[j] < t ? 1 : 0;
      a[j] -= t;
      for (++j; borrow && j < kWords; ++j) {
        borrow = a[j] == 0 ? 1 : 0;
        a[j] -= 1;
      }
    }
  }

  static bool bit_at(const uint64_t *v, long pos) {
    return ((v[size_t(pos) / 64] >> (size_t(pos) % 64)) & 1u) != 0;
  }

  static bool any_below(const uint64_t *v, long pos) {
    const size_t k = size_t(pos) / 64, b = size_t(pos) % 64;
    for (size_t j = 0; j < k; ++j) {
      if (v[j]) {
        return true;
      }
    }
    return b ? (v[k] & ((uint64_t(1) << b) - 1)) != 0 : false;
  }

  // 53 bits starting at bit pos (pos + 52 < 64 * kWords).
  static uint64_t bits_at(const uint64_t *v, long pos) {
    const size_t k = size_t(pos) / 64, b = size_t(pos) % 64;
    uint64_t out = v[k] >> b;
    if (b && k + 1 < kWords) {
      out |= v[k + 1] << (64 - b);
    }
    return out & ((uint64_t(1) << 53) - 1);
  }

  static double round_to_double(const uint64_t *a) {
    uint64_t v[kWords];
    std::memcpy(v, a, sizeof(v));
    const bool neg = (v[kWords - 1] >> 63) != 0;
    if (neg) {
      uint64_t carry = 1;
      for (size_t j = 0; j < kWords; ++j) {
        const uint64_t t = ~v[j];
        v[j] = t + carry;
        carry = (carry && t == UINT64_MAX) ? 1 : 0;
      }
    }
    int top = -1;
    for (int j = int(kWords) - 1; j >= 0; --j) {
      if (v[j]) {
        top = j;
        break;
      }
    }
    if (top < 0) {
      return 0.0;
    }
    int msb = 63;
    while (((v[top] >> msb) & 1u) == 0) {
      --msb;
    }
    const long h = long(top) * 64 + msb;   // highest set bit
    double mag;
    if (h <= 52) {
      mag = std::ldexp(double(v[0]), -1074);   // exact: fits 53 bits
    } else {
      uint64_t M = bits_at(v, h - 52);
      const bool R = bit_at(v, h - 53);
      const bool S = any_below(v, h - 53);
      if (R && (S || (M & 1u))) {
        ++M;
      }
      long hh = h;
      if (M == (uint64_t(1) << 53)) {
        M >>= 1;
        ++hh;
      }
      if (hh > 2097) {
        throw std::overflow_error(
            "ExactAccumulator: exact sum exceeds the double range");
      }
      mag = std::ldexp(double(M), int(hh - 52 - 1074));
    }
    return neg ? -mag : mag;
  }

  size_t npix_;
  std::vector<uint64_t> acc_;
  std::vector<uint64_t> density_;
  uint64_t n_terms_;
};

// Delete-one-mode jackknife for a pixel tile of the shared path: totals from
// ExactAccumulator, rows from the disk store (mode-major float32: per mode
// 2*npix re/im values, then npix intensities), the same pixel arithmetic as
// BeamletGrid::jackknife. Returns (mu, sigma, dubious, boundary) bytes for the
// tile; boundary marks pixels with a LOO mask quantity within band_rel of its
// threshold (0 when band_rel <= 0).
py::tuple jackknife_tile(py::buffer w_tot, py::buffer i_tot, double i_ref,
                         py::buffer w_rows, py::buffer i_rows,
                         const std::vector<double> &irefs, double eps_rel,
                         double band_rel) {
  const py::buffer_info bw = w_tot.request(), bi = i_tot.request(),
                        brw = w_rows.request(), bri = i_rows.request();
  const size_t npix = size_t(bi.size) * size_t(bi.itemsize) / 8;
  const size_t nf = irefs.size();
  if (size_t(bw.size) * size_t(bw.itemsize) != 16 * npix ||
      size_t(brw.size) * size_t(brw.itemsize) != nf * 8 * npix ||
      size_t(bri.size) * size_t(bri.itemsize) != nf * 4 * npix) {
    throw std::invalid_argument(
        "jackknife_tile: buffer sizes do not match the tile and mode count");
  }
  const auto *W = static_cast<const std::complex<double> *>(bw.ptr);
  const double *I = static_cast<const double *>(bi.ptr);
  const float *RW = static_cast<const float *>(brw.ptr);
  const float *RI = static_cast<const float *>(bri.ptr);
  std::vector<double> mu(npix, 0.0), err(npix, 0.0);
  std::vector<unsigned char> dub(npix, 0), boundary(npix, 0);
  std::vector<double> loo;
  loo.reserve(nf);
  if (!(i_ref > 0.0)) {
    for (size_t p = 0; p < npix; ++p) {
      dub[p] = I[p] > 0.0 ? 1 : 0;
    }
  } else {
    for (size_t p = 0; p < npix; ++p) {
      const double i_pix = I[p];
      if (i_pix <= 0.0) {
        continue;
      }
      jackknife_pixel(
          W[p], i_pix, i_ref, nf,
          [&](size_t s) { return RW + (s * npix + p) * 2; },
          [&](size_t s) { return RI[s * npix + p]; }, irefs.data(), eps_rel,
          loo, mu[p], err[p], dub[p], band_rel,
          band_rel > 0.0 ? &boundary[p] : nullptr);
    }
  }
  return py::make_tuple(as_bytes(mu), as_bytes(err), as_bytes(dub),
                        as_bytes(boundary));
}

}  // namespace

void register_trace(py::module_ &m) {
  py::class_<BeamletGrid>(m, "BeamletGrid",
                          "Stage-11 beamlet deposit grids (one per spectral "
                          "line): the hot window loop of BeamletField.")
      .def(py::init<long, long, double, double, double, double,
                    std::vector<double>, std::vector<double>,
                    std::vector<double>, double>(),
           py::arg("nx"), py::arg("ny"), py::arg("x0"), py::arg("y0"),
           py::arg("ex"), py::arg("ey"), py::arg("kms"), py::arg("zrs"),
           py::arg("zrs_t"), py::arg("ns"))
      .def("clear", &BeamletGrid::clear)
      .def("add_ray", &BeamletGrid::add_ray, py::arg("x"), py::arg("y"),
           py::arg("dx"), py::arg("dy"), py::arg("opl"), py::arg("psi"),
           py::arg("segs"), py::arg("lenses"), py::arg("amps"),
           "propagate + deposit for every line of one ray.")
      .def("add", &BeamletGrid::add, py::arg("m"), py::arg("x"), py::arg("y"),
           py::arg("pref"), py::arg("tx"), py::arg("ty"), py::arg("hxx"),
           py::arg("hxy"), py::arg("hyy"), py::arg("rx"), py::arg("ry"))
      .def("fold", &BeamletGrid::fold, py::arg("wfs"), py::arg("ref"),
           "Fold one mode into the W/I totals and delete-one rows; "
           "returns the mode's float32 ref intensity.")
      .def("fold_export", &BeamletGrid::fold_export, py::arg("wfs"),
           py::arg("ref"),
           "One mode's (w_row f32, i_row f32, dW f64, dI f64, i_ref) without "
           "retaining anything; the totals stay untouched.")
      .def("jackknife", &BeamletGrid::jackknife, py::arg("ref"),
           py::arg("eps_rel"),
           "Dense (mu, sigma, dubious) bytes from the fold rows.")
      .def("totals", &BeamletGrid::totals,
           "W (complex128) and I (float64) fold totals as bytes.")
      .def("at", &BeamletGrid::at, "Cell value: (line, pixel) -> complex.")
      .def("items", &BeamletGrid::items,
           "Nonzero cells of one line: [(pixel, complex), ...].")
      .attr("lens_stride") = 4;

  py::class_<ExactAccumulator>(
      m, "ExactAccumulator",
      "Exact order-free fixed-point totals (Re W, Im W, I) and integer "
      "density per pixel for the Stage-11 shared path.")
      .def(py::init<size_t>(), py::arg("npix"))
      .def("add", &ExactAccumulator::add, py::arg("dw"), py::arg("di"),
           py::arg("density"),
           "Add one mode: complex128 dW, float64 dI, uint32 density.")
      .def("totals", &ExactAccumulator::totals,
           "(W complex128, I float64) bytes: correctly rounded exact sums.")
      .def("density", &ExactAccumulator::density, "uint64 counts as bytes.")
      .def_property_readonly("n_terms", &ExactAccumulator::n_terms)
      .def_property_readonly("npix", &ExactAccumulator::npix);
  m.def("jackknife_tile", &jackknife_tile, py::arg("w_tot"), py::arg("i_tot"),
        py::arg("i_ref"), py::arg("w_rows"), py::arg("i_rows"),
        py::arg("irefs"), py::arg("eps_rel"), py::arg("band_rel") = 0.0,
        "Delete-one-mode jackknife (mu, sigma, dubious, boundary) for one "
        "pixel tile; boundary marks LOO masks within band_rel of a threshold.");

  py::class_<NativeOptic>(m, "NativeOptic")
      .def_readonly("precision", &NativeOptic::precision)
      .def_readonly("kind", &NativeOptic::kind);

  m.def(
      "trace_make_mirror",
      [](py::object z0, py::object z1, double z0f, double z1f) {
        NativeOptic out{precision_of(z0), "mirror", nullptr};
        dispatch(out.precision, [&](auto ic) {
          constexpr unsigned P = decltype(ic)::value;
          using T = cstrace::Tracer<P>;
          out.impl = std::make_shared<typename T::Optic>(typename T::Mirror{
              py::cast<mp_real<P>>(z0), py::cast<mp_real<P>>(z1), z0f, z1f});
        });
        return out;
      },
      "NativeOptic twin of surfaces.Mirror.", py::arg("z0"), py::arg("z1"),
      py::arg("z0f"), py::arg("z1f"));

  m.def(
      "trace_make_bundle",
      [](py::object z0, py::object z1, double z0f, double z1f,
         py::sequence walls) {
        NativeOptic out{precision_of(z0), "bundle", nullptr};
        dispatch(out.precision, [&](auto ic) {
          constexpr unsigned P = decltype(ic)::value;
          using T = cstrace::Tracer<P>;
          typename T::Bundle b{py::cast<mp_real<P>>(z0),
                               py::cast<mp_real<P>>(z1), z0f, z1f, {}};
          for (auto spec : walls) {
            b.walls.push_back(make_wall<P>(py::cast<py::sequence>(spec)));
          }
          out.impl = std::make_shared<typename T::Optic>(std::move(b));
        });
        return out;
      },
      "NativeOptic twin of surfaces.CapillaryBundle.", py::arg("z0"),
      py::arg("z1"), py::arg("z0f"), py::arg("z1f"), py::arg("walls"));

  m.def(
      "trace_ray_native",
      [](py::object optic, py::sequence origin, py::sequence direction,
         py::object screen_z, long max_bounces) {
        const unsigned p = precision_of(origin[0]);
        const NativeOptic *nat = nullptr;
        if (!optic.is_none()) {
          nat = py::cast<const NativeOptic *>(optic);
          if (nat->precision != p) {
            throw std::invalid_argument("optic/ray precision mismatch");
          }
        }
        py::tuple out;
        dispatch(p, [&](auto ic) {
          constexpr unsigned P = decltype(ic)::value;
          using T = cstrace::Tracer<P>;
          auto res = T::trace(nat ? optic_of<P>(*nat) : nullptr,
                              to_v3<P>(origin), to_v3<P>(direction),
                              py::cast<mp_real<P>>(screen_z), max_bounces);
          static const char *fates[] = {"screen", "absorbed", "lost"};
          py::list refl;
          for (const auto &r : res.reflections) {
            refl.append(
                py::make_tuple(from_v3<P>(r.point), py::cast(r.sin_g)));
          }
          out = py::make_tuple(fates[res.fate], from_v3<P>(res.point),
                               py::cast(res.opl), refl,
                               from_v3<P>(res.direction));
        });
        return out;
      },
      "trace.trace_ray twin: (fate, point, opl, [(point, sin), ...], dir).",
      py::arg("optic"), py::arg("origin"), py::arg("direction"),
      py::arg("screen_z"), py::arg("max_bounces"));

  m.def(
      "trace_next_event",
      [](const NativeOptic &nat, py::sequence O, py::sequence d) {
        py::tuple out;
        dispatch(nat.precision, [&](auto ic) {
          constexpr unsigned P = decltype(ic)::value;
          using T = cstrace::Tracer<P>;
          auto ev = T::next_event(*optic_of<P>(nat), to_v3<P>(O), to_v3<P>(d));
          switch (ev.kind) {
            case T::kExit:
              out = py::make_tuple("exit", py::none());
              break;
            case T::kPass:
              out = py::make_tuple("pass", py::cast(ev.t));
              break;
            case T::kAbsorb:
              out = py::make_tuple("absorb", py::cast(ev.t));
              break;
            default:
              out = py::make_tuple("reflect", py::cast(ev.t),
                                   from_v3<P>(ev.point),
                                   from_v3<P>(ev.normal));
          }
        });
        return out;
      },
      "next_event twin, same event tuples as the Python optics.");

  m.def(
      "trace_wall_hit",
      [](const NativeOptic &nat, size_t index, py::sequence O, py::sequence d,
         py::object t_exit) {
        py::object out = py::none();
        dispatch(nat.precision, [&](auto ic) {
          constexpr unsigned P = decltype(ic)::value;
          using T = cstrace::Tracer<P>;
          auto h = T::wall_hit(bundle_of<P>(nat).walls.at(index), to_v3<P>(O),
                               to_v3<P>(d), py::cast<mp_real<P>>(t_exit));
          if (h) {
            out = py::make_tuple(py::cast(h->t), from_v3<P>(h->point),
                                 from_v3<P>(h->normal));
          }
        });
        return out;
      },
      "Wall.hit twin for bundle wall #index: None or (t, point, normal).");

  m.def("trace_wall_inside",
        [](const NativeOptic &nat, size_t index, double x, double y,
           double z) {
          bool out = false;
          dispatch(nat.precision, [&](auto ic) {
            constexpr unsigned P = decltype(ic)::value;
            using T = cstrace::Tracer<P>;
            out = T::wall_inside(bundle_of<P>(nat).walls.at(index), x, y, z);
          });
          return out;
        },
        "Wall.inside twin for bundle wall #index.");

  // Root-finder debug hooks for the parity tests.
  m.def("trace_dbg_dk_roots", [](const std::vector<double> &cf) {
    return cstrace::dk_roots(cf);
  });
  m.def("trace_dbg_quartic_first", [](py::sequence c, double t_capf) {
    py::object out = py::none();
    dispatch(precision_of(c[0]), [&](auto ic) {
      constexpr unsigned P = decltype(ic)::value;
      using T = cstrace::Tracer<P>;
      std::array<mp_real<P>, 5> arr = {
          py::cast<mp_real<P>>(c[0]), py::cast<mp_real<P>>(c[1]),
          py::cast<mp_real<P>>(c[2]), py::cast<mp_real<P>>(c[3]),
          py::cast<mp_real<P>>(c[4])};
      auto t = T::quartic_first(arr, t_capf);
      if (t) {
        out = py::cast(*t);
      }
    });
    return out;
  });
}
