// Native CAPSYSred tracer bindings: NativeOptic twins of the Python optics,
// the trace/next_event/hit entry points, and debug hooks for the root-finder
// parity tests. Own translation unit to keep per-compile memory bounded.
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <memory>
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
        g_(kms_.size(), std::vector<std::complex<double>>(size_t(nx) * ny)) {}

  void clear() {
    for (auto &g : g_) {
      std::fill(g.begin(), g.end(), std::complex<double>(0.0, 0.0));
    }
  }

  // Mirrors the Python window loop op-for-op (pixel centers included):
  // envelope exp((k/2)*d^T Im(G) d), phase tilt + (k/2)*d^T Re(G) d.
  void add(size_t m, double x, double y, std::complex<double> pref, double tx,
           double ty, std::complex<double> hxx, std::complex<double> hxy,
           std::complex<double> hyy, double r) {
    const long ix_lo = std::max(0L, long(std::floor((x - r - x0_) / dx_)));
    const long ix_hi = std::min(nx_ - 1, long(std::floor((x + r - x0_) / dx_)));
    const long iy_lo = std::max(0L, long(std::floor((y - r - y0_) / dy_)));
    const long iy_hi = std::min(ny_ - 1, long(std::floor((y + r - y0_) / dy_)));
    if (ix_lo > ix_hi || iy_lo > iy_hi) {
      return;
    }
    auto &g = g_.at(m);
    for (long iy = iy_lo; iy <= iy_hi; ++iy) {
      const double dy_off = y0_ + (iy + 0.5) * ey_ / ny_ - y;
      const std::complex<double> cy = hyy * (dy_off * dy_off);
      const double phase_y = ty * dy_off;
      std::complex<double> *row = g.data() + size_t(iy) * nx_;
      for (long ix = ix_lo; ix <= ix_hi; ++ix) {
        const double dx_off = x0_ + (ix + 0.5) * ex_ / nx_ - x;
        const std::complex<double> quad =
            hxx * (dx_off * dx_off) + 2.0 * hxy * (dx_off * dy_off) + cy;
        row[ix] += pref * std::exp(std::complex<double>(
                              quad.imag(),
                              quad.real() + tx * dx_off + phase_y));
      }
    }
  }

  // Whole-ray deposit: gamma.propagate + the spot per line, one call per
  // ray. lenses is flat [(phi, 1/f_t, 1/f_s), ...]; returns (spot width of
  // line 0, geometric mean of the axes; -1 when skipped) and the number of
  // lines whose Im(G) lost negative-definiteness (not deposited).
  py::tuple add_ray(double x, double y, double dxf, double dyf, double opl,
                    double psi, const std::vector<double> &segs,
                    const std::vector<double> &lenses,
                    const std::vector<std::complex<double>> &amps) {
    const size_t n_lens = lenses.size() / 3;
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
        // adaptive sub-steps (gamma.propagate twin): step <= min axis z_R
        const double mean = 0.5 * (qxx.imag() + qyy.imag());
        const double dev =
            std::hypot(0.5 * (qxx.imag() - qyy.imag()), qxy.imag());
        const double zr_min = mean - dev;
        const int nsub =
            (zr_min > 0.0 && segs[j] > 0.0)
                ? std::max(2, std::min(256, int(std::ceil(segs[j] / zr_min))))
                : 2;
        const double step = segs[j] / nsub;
        for (int sub = 0; sub < nsub; ++sub) {
          const std::complex<double> pre = qxx * qyy - qxy * qxy;
          qxx += step;
          qyy += step;
          a_geo *= std::sqrt(pre / (qxx * qyy - qxy * qxy));
        }
        if (j < n_lens) {
          const double phi = lenses[3 * j], ift = lenses[3 * j + 1],
                       ifs = lenses[3 * j + 2];
          if (ift == 0.0 && ifs == 0.0) {
            continue;
          }
          const double c = std::cos(phi), sn = std::sin(phi);
          const double pxx = ift * c * c + ifs * sn * sn;
          const double pxy = (ift - ifs) * c * sn;
          const double pyy = ift * sn * sn + ifs * c * c;
          std::complex<double> det = qxx * qyy - qxy * qxy;
          const std::complex<double> gxx = qyy / det - pxx;
          const std::complex<double> gxy = -qxy / det - pxy;
          const std::complex<double> gyy = qxx / det - pyy;
          det = gxx * gyy - gxy * gxy;
          qxx = gyy / det;
          qxy = -gxy / det;
          qyy = gxx / det;
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
      add(m, x, y, pref, km * dxf, km * dyf, 0.5 * km * gxx, 0.5 * km * gxy,
          0.5 * km * gyy, ns_ * w_hi);
    }
    return py::make_tuple(w_spot, bad);
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
};

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
           py::arg("hxy"), py::arg("hyy"), py::arg("r"))
      .def("at", &BeamletGrid::at, "Cell value: (line, pixel) -> complex.")
      .def("items", &BeamletGrid::items,
           "Nonzero cells of one line: [(pixel, complex), ...].");

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
