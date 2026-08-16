// CAPSYSred Stage 14: an exact, disk-backed delete-one-mode jackknife.
//
// This translation unit is intentionally independent from BeamletGrid.  The
// cache payload is a headerless canonical little-endian stream with shape
// [mode, pixel, 3] and fields (W.real, W.imag, Ic), all binary64.  It is read
// with ordinary buffered streams (no NumPy and no mmap).
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace py = pybind11;

namespace {

constexpr std::size_t kCellBytes = 3 * sizeof(double);

std::size_t checked_mul(std::size_t a, std::size_t b,
                        const char *what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::overflow_error(std::string(what) + " size overflow");
  }
  return a * b;
}

std::uint64_t checked_u64_add(std::uint64_t a, std::uint64_t b,
                              const char *what) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    throw std::overflow_error(std::string(what) + " overflows uint64");
  }
  return a + b;
}

void checked_u32_inc(std::uint32_t &value, const char *what) {
  if (value == std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(std::string(what) + " overflows uint32");
  }
  ++value;
}

// A small self-contained SHA-256 implementation.  Keeping it here lets the
// mandatory first finalize pass validate each multi-gigabyte payload without
// a third Python read or an external crypto dependency.
class Sha256 {
 public:
  Sha256() { reset(); }

  void update(const void *data, std::size_t size) {
    const auto *p = static_cast<const std::uint8_t *>(data);
    constexpr std::uint64_t max_bytes =
        std::numeric_limits<std::uint64_t>::max() / 8;
    if (size > max_bytes - total_bytes_) {
      throw std::length_error(
          "SHA-256 input is too long for its 64-bit bit-length field");
    }
    total_bytes_ += static_cast<std::uint64_t>(size);
    while (size != 0) {
      const std::size_t take = std::min(size, block_.size() - used_);
      std::memcpy(block_.data() + used_, p, take);
      used_ += take;
      p += take;
      size -= take;
      if (used_ == block_.size()) {
        transform(block_.data());
        used_ = 0;
      }
    }
  }

  std::string hex_digest() const {
    Sha256 copy = *this;
    const std::uint64_t bit_count = copy.total_bytes_ * std::uint64_t(8);
    const std::uint8_t one = 0x80;
    copy.append_padding(&one, 1);
    const std::uint8_t zero = 0;
    while (copy.used_ != 56) {
      copy.append_padding(&zero, 1);
    }
    std::array<std::uint8_t, 8> length{};
    for (unsigned i = 0; i != 8; ++i) {
      length[7 - i] = static_cast<std::uint8_t>(bit_count >> (8 * i));
    }
    copy.append_padding(length.data(), length.size());

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto word : copy.state_) {
      out << std::setw(8) << word;
    }
    return out.str();
  }

 private:
  static std::uint32_t rotr(std::uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
  }

  void reset() {
    state_ = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
              0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    block_.fill(0);
    used_ = 0;
    total_bytes_ = 0;
  }

  void append_padding(const void *data, std::size_t size) {
    // Padding must not change total_bytes_, because the encoded length is the
    // length of the original message.
    const auto *p = static_cast<const std::uint8_t *>(data);
    while (size != 0) {
      const std::size_t take = std::min(size, block_.size() - used_);
      std::memcpy(block_.data() + used_, p, take);
      used_ += take;
      p += take;
      size -= take;
      if (used_ == block_.size()) {
        transform(block_.data());
        used_ = 0;
      }
    }
  }

  void transform(const std::uint8_t *p) {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> w{};
    for (unsigned i = 0; i != 16; ++i) {
      const unsigned j = 4 * i;
      w[i] = (std::uint32_t(p[j]) << 24) |
             (std::uint32_t(p[j + 1]) << 16) |
             (std::uint32_t(p[j + 2]) << 8) | std::uint32_t(p[j + 3]);
    }
    for (unsigned i = 16; i != 64; ++i) {
      const std::uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (unsigned i = 0; i != 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> block_{};
  std::size_t used_ = 0;
  std::uint64_t total_bytes_ = 0;
};

void put_u64_le(std::uint8_t *dst, std::uint64_t value) {
  for (unsigned i = 0; i != 8; ++i) {
    dst[i] = static_cast<std::uint8_t>(value >> (8 * i));
  }
}

std::uint64_t get_u64_le(const std::uint8_t *src) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i != 8; ++i) {
    value |= std::uint64_t(src[i]) << (8 * i);
  }
  return value;
}

void put_double_le(std::uint8_t *dst, double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "Stage 14 requires IEEE-754 binary64-sized doubles");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  put_u64_le(dst, bits);
}

double get_double_le(const std::uint8_t *src) {
  const std::uint64_t bits = get_u64_le(src);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

template <typename T>
py::bytes le_bytes(const std::vector<T> &values);

template <>
py::bytes le_bytes<double>(const std::vector<double> &values) {
  std::string raw(checked_mul(values.size(), sizeof(double), "float64 array"),
                  '\0');
  auto *out = reinterpret_cast<std::uint8_t *>(raw.data());
  for (std::size_t i = 0; i != values.size(); ++i) {
    put_double_le(out + i * sizeof(double), values[i]);
  }
  return py::bytes(raw);
}

template <>
py::bytes le_bytes<std::uint64_t>(const std::vector<std::uint64_t> &values) {
  std::string raw(checked_mul(values.size(), sizeof(std::uint64_t),
                              "uint64 array"),
                  '\0');
  auto *out = reinterpret_cast<std::uint8_t *>(raw.data());
  for (std::size_t i = 0; i != values.size(); ++i) {
    put_u64_le(out + i * sizeof(std::uint64_t), values[i]);
  }
  return py::bytes(raw);
}

template <>
py::bytes le_bytes<std::uint32_t>(const std::vector<std::uint32_t> &values) {
  std::string raw(checked_mul(values.size(), sizeof(std::uint32_t),
                              "uint32 array"),
                  '\0');
  auto *out = reinterpret_cast<std::uint8_t *>(raw.data());
  for (std::size_t i = 0; i != values.size(); ++i) {
    const std::uint32_t v = values[i];
    for (unsigned j = 0; j != 4; ++j) {
      out[i * 4 + j] = static_cast<std::uint8_t>(v >> (8 * j));
    }
  }
  return py::bytes(raw);
}

template <>
py::bytes le_bytes<std::uint8_t>(const std::vector<std::uint8_t> &values) {
  return py::bytes(reinterpret_cast<const char *>(values.data()), values.size());
}

std::vector<double> doubles_from_bytes(const py::bytes &value,
                                       std::size_t count,
                                       const char *name) {
  const std::string raw = value;
  const std::size_t expected = checked_mul(count, sizeof(double), name);
  if (raw.size() != expected) {
    throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                std::to_string(expected) + " bytes");
  }
  std::vector<double> out(count);
  const auto *p = reinterpret_cast<const std::uint8_t *>(raw.data());
  for (std::size_t i = 0; i != count; ++i) {
    out[i] = get_double_le(p + i * sizeof(double));
    if (!std::isfinite(out[i])) {
      throw std::invalid_argument(std::string(name) +
                                  " contains a non-finite value at pixel " +
                                  std::to_string(i));
    }
  }
  return out;
}

std::FILE *open_exclusive_payload(const std::filesystem::path &path,
                                  const std::string &display) {
  int fd = -1;
#ifdef _WIN32
  const HANDLE handle = CreateFileW(
      path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const DWORD error = GetLastError();
    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
      throw std::runtime_error("Stage-14 payload already exists: " + display +
                               "; remove it or choose another cache directory");
    }
    throw std::runtime_error("cannot create Stage-14 payload " + display +
                             ": " +
                             std::error_code(error, std::system_category())
                                 .message());
  }
  fd = _open_osfhandle(reinterpret_cast<std::intptr_t>(handle),
                       _O_WRONLY | _O_BINARY);
  if (fd < 0) {
    const int error = errno;
    CloseHandle(handle);
    // As below, leave the exclusively-created pathname in place rather than
    // risk deleting an external replacement after releasing our handle.
    throw std::runtime_error("cannot attach Stage-14 payload " + display +
                             ": " +
                             std::error_code(error, std::generic_category())
                                 .message());
  }
  std::FILE *out = _fdopen(fd, "wb");
#else
  int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
  fd = ::open(path.c_str(), flags, 0666);
  if (fd < 0) {
    const int error = errno;
    if (error == EEXIST) {
      throw std::runtime_error("Stage-14 payload already exists: " + display +
                               "; remove it or choose another cache directory");
    }
    throw std::runtime_error("cannot create Stage-14 payload " + display +
                             ": " +
                             std::error_code(error, std::generic_category())
                                 .message());
  }
  std::FILE *out = fdopen(fd, "wb");
#endif
  if (out == nullptr) {
    const int error = errno;
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
    // The exclusively-created file is deliberately left in place.  It is a
    // fail-closed partial, and removing by pathname after closing the handle
    // would introduce another race with an external replacement.
    throw std::runtime_error("cannot buffer Stage-14 payload " + display +
                             ": " +
                             std::error_code(error, std::generic_category())
                                 .message());
  }
  return out;
}

double product_scaled(double a, double b, double c, double d) {
  if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) {
    return 0.0;
  }
  int exponent = 0;
  int part = 0;
  double mantissa = std::frexp(a, &part);
  exponent += part;
  mantissa *= std::frexp(b, &part);
  exponent += part;
  mantissa *= std::frexp(c, &part);
  exponent += part;
  mantissa *= std::frexp(d, &part);
  exponent += part;
  return std::scalbn(mantissa, exponent);
}

double weighted_norm(const std::complex<double> &value, double weight) {
  if (weight == 0.0 || value == std::complex<double>(0.0, 0.0)) {
    return 0.0;
  }
  const double direct_norm = std::norm(value);
  const double direct = weight * direct_norm;
  if (std::isfinite(direct) && direct_norm != 0.0) {
    return direct;
  }
  const double scale =
      std::max(std::abs(value.real()), std::abs(value.imag()));
  const double real = value.real() / scale;
  const double imag = value.imag() / scale;
  return product_scaled(weight, scale, scale, real * real + imag * imag);
}

std::complex<double> weighted_cross(const std::complex<double> &left,
                                    const std::complex<double> &right_conj,
                                    double weight) {
  if (weight == 0.0 || left == std::complex<double>(0.0, 0.0) ||
      right_conj == std::complex<double>(0.0, 0.0)) {
    return {0.0, 0.0};
  }
  const std::complex<double> direct_product = left * right_conj;
  const std::complex<double> direct = weight * direct_product;
  if (std::isfinite(direct.real()) && std::isfinite(direct.imag())) {
    return direct;
  }

  const double left_scale =
      std::max(std::abs(left.real()), std::abs(left.imag()));
  const double right_scale =
      std::max(std::abs(right_conj.real()), std::abs(right_conj.imag()));
  const std::complex<double> normalized =
      (left / left_scale) * (right_conj / right_scale);
  return {
      product_scaled(weight, left_scale, right_scale, normalized.real()),
      product_scaled(weight, left_scale, right_scale, normalized.imag()),
  };
}

// |W| / sqrt(Ic * Ic_ref), without ever materializing either |W| or the
// denominator outside binary64 range.  The mantissas remain O(1); scalbn is
// the only operation allowed to under/overflow, exactly when the final ratio
// itself is outside the representable range.
double range_safe_mu(double wr, double wi, double ic, double ic_ref) {
  const double scale = std::max(std::abs(wr), std::abs(wi));
  if (scale == 0.0) {
    return 0.0;
  }
  int ew = 0;
  double mw = std::frexp(scale, &ew);
  mw *= std::hypot(wr / scale, wi / scale);
  if (mw >= 1.0) {
    mw *= 0.5;
    ++ew;
  }

  int ei = 0, er = 0;
  const double mi = std::frexp(ic, &ei);
  const double mr = std::frexp(ic_ref, &er);
  const int combined = ei + er;
  int denominator_exp = combined / 2;
  if (combined < 0 && combined % 2 != 0) {
    --denominator_exp;
  }
  const int remainder = combined - 2 * denominator_exp;
  const double denominator_mantissa =
      std::sqrt(std::scalbn(mi * mr, remainder));
  return std::scalbn(mw / denominator_mantissa, ew - denominator_exp);
}

class Stage14Store {
 public:
  Stage14Store(std::string path, std::size_t n_modes, std::size_t n_pixels,
               std::size_t ref, std::vector<double> kms,
               std::vector<double> weights)
      : path_(std::move(path)),
        n_modes_(n_modes),
        n_pixels_(n_pixels),
        ref_(ref),
        kms_(std::move(kms)),
        weights_(std::move(weights)),
        g_(checked_mul(kms_.size(), n_pixels_, "current-mode field")),
        sq_(checked_mul(kms_.size(), n_pixels_, "current-mode square sum")),
        next_g_(kms_.size()),
        next_sq_(kms_.size()),
        count_(n_pixels_),
        row_w_(n_pixels_),
        row_ic_(n_pixels_),
        total_i_(n_pixels_),
        total_w_(n_pixels_),
        total_ic_(n_pixels_),
        n_rays_(n_pixels_),
        m_realizations_(n_pixels_),
        m_pair_realizations_(n_pixels_),
        m_ref_realizations_(n_pixels_),
        max_rays_per_realization_(n_pixels_),
        row_bytes_(checked_mul(n_pixels_, kCellBytes, "Stage-14 row")) {
    if (n_modes_ == 0) {
      throw std::invalid_argument("n_modes must be positive");
    }
    if (n_pixels_ == 0) {
      throw std::invalid_argument("n_pixels must be positive");
    }
    if (ref_ >= n_pixels_) {
      throw std::invalid_argument("reference pixel is outside the grid");
    }
    if (kms_.empty() || kms_.size() != weights_.size()) {
      throw std::invalid_argument(
          "kms and weights must have the same non-zero length");
    }
    for (std::size_t i = 0; i != kms_.size(); ++i) {
      if (!std::isfinite(kms_[i])) {
        throw std::invalid_argument("kms contains a non-finite value");
      }
      if (!std::isfinite(weights_[i]) || weights_[i] < 0.0) {
        throw std::invalid_argument(
            "weights must contain finite non-negative values");
      }
    }
    out_ = open_exclusive_payload(std::filesystem::u8path(path_), path_);
  }

  ~Stage14Store() {
    if (out_ != nullptr) {
      std::fclose(out_);
    }
  }

  Stage14Store(const Stage14Store &) = delete;
  Stage14Store &operator=(const Stage14Store &) = delete;

  void begin_mode(std::int64_t mode) {
    require_open();
    if (mode_open_) {
      throw std::logic_error("fold_mode() is required before begin_mode()");
    }
    if (folded_ >= n_modes_) {
      throw std::logic_error("more modes supplied than declared");
    }
    if (!mode_ids_.empty()) {
      if (mode_ids_.back() == std::numeric_limits<std::int64_t>::max() ||
          mode != mode_ids_.back() + 1) {
        throw std::invalid_argument(
            "Stage-14 modes must be consecutive and strictly increasing");
      }
    }
    current_mode_ = mode;
    mode_open_ = true;
  }

  void add_ray(std::size_t pixel, double opl,
               const std::vector<std::complex<double>> &amps) {
    require_open();
    if (!mode_open_) {
      throw std::logic_error("begin_mode() is required before add_ray()");
    }
    if (pixel >= n_pixels_) {
      throw std::out_of_range("ray pixel is outside the Stage-14 grid");
    }
    if (!std::isfinite(opl)) {
      throw std::invalid_argument("ray optical path length is non-finite");
    }
    if (amps.size() != kms_.size()) {
      throw std::invalid_argument("ray amplitude count differs from line count");
    }
    if (count_[pixel] == std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("rays per pixel/mode overflow uint32");
    }
    // Prepare every line first.  No store state changes until all inputs and
    // candidate accumulators have passed validation, so a rejected ray cannot
    // leave ghost field/square contributions behind its unincremented count.
    for (std::size_t line = 0; line != kms_.size(); ++line) {
      const auto amp = amps[line];
      if (!std::isfinite(amp.real()) || !std::isfinite(amp.imag())) {
        throw std::invalid_argument("ray amplitudes contain a non-finite value");
      }
      const std::size_t at = line * n_pixels_ + pixel;
      if (weights_[line] == 0.0) {
        next_g_[line] = g_[at];
        next_sq_[line] = sq_[at];
        continue;
      }
      const double phase = kms_[line] * opl;
      if (!std::isfinite(phase)) {
        throw std::invalid_argument("ray spectral phase is non-finite");
      }
      const std::complex<double> term =
          amp * std::complex<double>(std::cos(phase), std::sin(phase));
      next_g_[line] = g_[at] + term;
      next_sq_[line] = sq_[at] + weighted_norm(amp, weights_[line]);
      if (!std::isfinite(next_g_[line].real()) ||
          !std::isfinite(next_g_[line].imag()) ||
          !std::isfinite(next_sq_[line])) {
        throw std::overflow_error("current-mode accumulator became non-finite");
      }
    }
    for (std::size_t line = 0; line != kms_.size(); ++line) {
      if (weights_[line] == 0.0) {
        continue;
      }
      const std::size_t at = line * n_pixels_ + pixel;
      g_[at] = next_g_[line];
      sq_[at] = next_sq_[line];
    }
    ++count_[pixel];
  }

  void fold_mode() {
    require_open();
    if (!mode_open_) {
      throw std::logic_error("begin_mode() is required before fold_mode()");
    }

    std::fill(row_w_.begin(), row_w_.end(), std::complex<double>(0.0, 0.0));
    std::fill(row_ic_.begin(), row_ic_.end(), 0.0);
    const bool ref_occupied = count_[ref_] != 0;

    for (std::size_t pixel = 0; pixel != n_pixels_; ++pixel) {
      const std::uint32_t n = count_[pixel];
      n_rays_[pixel] = checked_u64_add(n_rays_[pixel], n, "n_rays");
      if (n != 0) {
        checked_u32_inc(m_realizations_[pixel], "m_realizations");
        max_rays_per_realization_[pixel] =
            std::max(max_rays_per_realization_[pixel], n);
        if (n >= 2) {
          checked_u32_inc(m_pair_realizations_[pixel],
                          "m_pair_realizations");
        }
        if (ref_occupied) {
          checked_u32_inc(m_ref_realizations_[pixel],
                          "m_ref_realizations");
        }
      }
    }

    for (std::size_t line = 0; line != kms_.size(); ++line) {
      const std::size_t base = line * n_pixels_;
      const std::complex<double> ref_conj = std::conj(g_[base + ref_]);
      const double weight = weights_[line];
      for (std::size_t pixel = 0; pixel != n_pixels_; ++pixel) {
        const std::size_t at = base + pixel;
        const double a2 = weighted_norm(g_[at], weight);
        total_i_[pixel] += a2;
        const double ic = a2 - sq_[at];
        row_ic_[pixel] += ic;
        std::complex<double> cross =
            weighted_cross(g_[at], ref_conj, weight);
        if (pixel == ref_) {
          cross -= sq_[at];
        }
        row_w_[pixel] += cross;
      }
    }

    for (std::size_t pixel = 0; pixel != n_pixels_; ++pixel) {
      total_w_[pixel] += row_w_[pixel];
      total_ic_[pixel] += row_ic_[pixel];
      if (!std::isfinite(total_i_[pixel]) ||
          !std::isfinite(row_w_[pixel].real()) ||
          !std::isfinite(row_w_[pixel].imag()) ||
          !std::isfinite(row_ic_[pixel]) ||
          !std::isfinite(total_w_[pixel].real()) ||
          !std::isfinite(total_w_[pixel].imag()) ||
          !std::isfinite(total_ic_[pixel])) {
        throw std::overflow_error(
            "Stage-14 fold produced a non-finite row or total");
      }
      auto *dst = row_bytes_.data() + pixel * kCellBytes;
      put_double_le(dst, row_w_[pixel].real());
      put_double_le(dst + 8, row_w_[pixel].imag());
      put_double_le(dst + 16, row_ic_[pixel]);
    }
    const std::size_t written =
        std::fwrite(row_bytes_.data(), 1, row_bytes_.size(), out_);
    if (written != row_bytes_.size()) {
      throw std::runtime_error("failed to write Stage-14 payload: " + path_);
    }
    payload_sha256_.update(row_bytes_.data(), row_bytes_.size());
    payload_bytes_ = checked_u64_add(payload_bytes_, row_bytes_.size(),
                                     "Stage-14 payload");
    mode_ids_.push_back(current_mode_);
    ++folded_;
    mode_open_ = false;
    std::fill(g_.begin(), g_.end(), std::complex<double>(0.0, 0.0));
    std::fill(sq_.begin(), sq_.end(), 0.0);
    std::fill(count_.begin(), count_.end(), 0);
  }

  py::dict finish() {
    require_open();
    if (mode_open_) {
      throw std::logic_error("fold_mode() is required before finish()");
    }
    if (folded_ != n_modes_) {
      throw std::logic_error("Stage-14 payload has " +
                             std::to_string(folded_) + " modes; expected " +
                             std::to_string(n_modes_));
    }
    const std::uint64_t expected = static_cast<std::uint64_t>(checked_mul(
        checked_mul(n_modes_, n_pixels_, "Stage-14 payload"), kCellBytes,
        "Stage-14 payload"));
    if (payload_bytes_ != expected) {
      throw std::runtime_error("Stage-14 payload byte count mismatch");
    }
    if (std::fflush(out_) != 0) {
      throw std::runtime_error("failed to flush Stage-14 payload: " + path_);
    }
    // fclose consumes the stream even when it reports a delayed write error;
    // make every such finish attempt terminal before dropping the pointer.
    finished_ = true;
    if (std::fclose(out_) != 0) {
      out_ = nullptr;
      throw std::runtime_error("failed to close Stage-14 payload: " + path_);
    }
    out_ = nullptr;

    std::vector<double> w_re(n_pixels_), w_im(n_pixels_);
    for (std::size_t pixel = 0; pixel != n_pixels_; ++pixel) {
      // Rows and totals were checked during fold; repeat at the publication
      // boundary so this invariant remains local and unmistakable.
      if (!std::isfinite(total_i_[pixel]) ||
          !std::isfinite(total_w_[pixel].real()) ||
          !std::isfinite(total_w_[pixel].imag()) ||
          !std::isfinite(total_ic_[pixel])) {
        throw std::runtime_error(
            "cannot publish non-finite Stage-14 aggregates");
      }
      w_re[pixel] = total_w_[pixel].real();
      w_im[pixel] = total_w_[pixel].imag();
    }

    py::dict result;
    result["I"] = le_bytes(total_i_);
    result["w_re"] = le_bytes(w_re);
    result["w_im"] = le_bytes(w_im);
    result["ic"] = le_bytes(total_ic_);
    result["n_rays"] = le_bytes(n_rays_);
    result["m_realizations"] = le_bytes(m_realizations_);
    result["m_pair_realizations"] = le_bytes(m_pair_realizations_);
    result["m_ref_realizations"] = le_bytes(m_ref_realizations_);
    result["max_rays_per_realization"] =
        le_bytes(max_rays_per_realization_);
    result["payload_bytes"] = py::int_(payload_bytes_);
    result["payload_sha256"] = payload_sha256_.hex_digest();
    result["n_modes"] = py::int_(folded_);
    result["mode_ids"] = mode_ids_;
    return result;
  }

 private:
  void require_open() const {
    if (finished_) {
      throw std::logic_error("Stage14Store is already finished");
    }
  }

  std::string path_;
  std::size_t n_modes_ = 0;
  std::size_t n_pixels_ = 0;
  std::size_t ref_ = 0;
  std::vector<double> kms_;
  std::vector<double> weights_;
  std::vector<std::complex<double>> g_;
  std::vector<double> sq_;
  std::vector<std::complex<double>> next_g_;
  std::vector<double> next_sq_;
  std::vector<std::uint32_t> count_;
  std::vector<std::complex<double>> row_w_;
  std::vector<double> row_ic_;
  std::vector<double> total_i_;
  std::vector<std::complex<double>> total_w_;
  std::vector<double> total_ic_;
  std::vector<std::uint64_t> n_rays_;
  std::vector<std::uint32_t> m_realizations_;
  std::vector<std::uint32_t> m_pair_realizations_;
  std::vector<std::uint32_t> m_ref_realizations_;
  std::vector<std::uint32_t> max_rays_per_realization_;
  std::vector<std::uint8_t> row_bytes_;
  std::FILE *out_ = nullptr;
  Sha256 payload_sha256_;
  std::uint64_t payload_bytes_ = 0;
  std::size_t folded_ = 0;
  std::int64_t current_mode_ = 0;
  std::vector<std::int64_t> mode_ids_;
  bool mode_open_ = false;
  bool finished_ = false;
};

// Kahan addition is inexpensive relative to disk I/O and preserves the
// centered-sum contract even for many thousands of jackknife units.
void kahan_add(double value, double &sum, double &correction) {
  const double y = value - correction;
  const double t = sum + y;
  correction = (t - sum) - y;
  sum = t;
}

class ScaledSumSquares {
 public:
  void add(double value) {
    const double magnitude = std::abs(value);
    if (magnitude == 0.0) {
      return;
    }
    if (scale_ < magnitude) {
      if (scale_ == 0.0) {
        scale_ = magnitude;
        sum_ = 1.0;
        correction_ = 0.0;
        return;
      }
      const double ratio = scale_ / magnitude;
      const double ratio2 = ratio * ratio;
      sum_ *= ratio2;
      correction_ *= ratio2;
      scale_ = magnitude;
      kahan_add(1.0, sum_, correction_);
      return;
    }
    const double ratio = magnitude / scale_;
    kahan_add(ratio * ratio, sum_, correction_);
  }

  double root(double factor) const {
    if (scale_ == 0.0) {
      return 0.0;
    }
    return scale_ * std::sqrt(factor * sum_);
  }

 private:
  double scale_ = 0.0;
  double sum_ = 0.0;
  double correction_ = 0.0;
};

struct FinalizeResult {
  std::vector<double> ic_err;
  std::vector<double> w_err;
  std::vector<double> mu_raw;
  std::vector<double> mu_raw_err;
  std::vector<std::uint32_t> n_mu_loo_valid;
  std::vector<std::uint8_t> mu_raw_defined;
  std::vector<std::uint8_t> mu_raw_err_defined;
  std::vector<double> ic_ref_loo;
  std::vector<std::string> payload_sha256;
  std::vector<std::uint64_t> payload_bytes;
  std::uint64_t bytes_read = 0;
  double pass1_seconds = 0.0;
  double pass2_seconds = 0.0;
};

std::uint64_t file_size_strict(const std::string &path) {
  std::ifstream in(std::filesystem::u8path(path),
                   std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::runtime_error("cannot open Stage-14 payload: " + path);
  }
  const std::streamoff end = in.tellg();
  if (end < 0) {
    throw std::runtime_error("cannot determine Stage-14 payload size: " + path);
  }
  return static_cast<std::uint64_t>(end);
}

std::ifstream open_payload(const std::string &path) {
  std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open Stage-14 payload: " + path);
  }
  return in;
}

void read_exact(std::ifstream &in, std::vector<std::uint8_t> &row,
                const std::string &path, std::size_t mode_index) {
  in.read(reinterpret_cast<char *>(row.data()),
          static_cast<std::streamsize>(row.size()));
  if (in.gcount() != static_cast<std::streamsize>(row.size()) || !in) {
    throw std::runtime_error("short read in Stage-14 payload " + path +
                             " at local mode " +
                             std::to_string(mode_index));
  }
}

std::string lower_ascii(std::string value) {
  for (char &ch : value) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }
  return value;
}

FinalizeResult finalize_impl(
    const std::vector<std::string> &row_paths,
    const std::vector<std::size_t> &mode_counts, std::size_t n_pixels,
    std::size_t ref, const std::vector<double> &total_i,
    const std::vector<double> &total_w_re,
    const std::vector<double> &total_w_im,
    const std::vector<double> &total_ic,
    const std::vector<std::string> &expected_sha256) {
  if (row_paths.empty() || row_paths.size() != mode_counts.size()) {
    throw std::invalid_argument(
        "row_paths and mode_counts must have the same non-zero length");
  }
  if (n_pixels == 0 || ref >= n_pixels) {
    throw std::invalid_argument("invalid Stage-14 grid/reference");
  }
  if (!expected_sha256.empty() && expected_sha256.size() != row_paths.size()) {
    throw std::invalid_argument(
        "expected_sha256 must be empty or match row_paths");
  }
  std::size_t n_modes = 0;
  for (const auto count : mode_counts) {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() - n_modes) {
      throw std::invalid_argument("mode counts must be positive and fit size_t");
    }
    n_modes += count;
  }
  if (n_modes < 2) {
    throw std::invalid_argument("Stage-14 jackknife requires at least 2 modes");
  }
  const std::size_t row_size =
      checked_mul(n_pixels, kCellBytes, "Stage-14 row");

  FinalizeResult result;
  result.payload_bytes.resize(row_paths.size());
  for (std::size_t part = 0; part != row_paths.size(); ++part) {
    const std::size_t expected_size =
        checked_mul(mode_counts[part], row_size, "Stage-14 payload");
    const std::uint64_t actual_size = file_size_strict(row_paths[part]);
    if (actual_size != static_cast<std::uint64_t>(expected_size)) {
      throw std::runtime_error(
          "Stage-14 payload size mismatch for " + row_paths[part] +
          ": got " + std::to_string(actual_size) + ", expected " +
          std::to_string(expected_size));
    }
    result.payload_bytes[part] = actual_size;
  }

  const double factor = static_cast<double>(n_modes - 1) /
                        static_cast<double>(n_modes);
  result.ic_err.assign(n_pixels, 0.0);
  result.w_err.assign(n_pixels, 0.0);
  result.mu_raw.assign(n_pixels,
                       std::numeric_limits<double>::quiet_NaN());
  result.mu_raw_err.assign(n_pixels,
                           std::numeric_limits<double>::quiet_NaN());
  result.n_mu_loo_valid.assign(n_pixels, 0);
  result.mu_raw_defined.assign(n_pixels, 0);
  result.mu_raw_err_defined.assign(n_pixels, 0);
  result.ic_ref_loo.reserve(n_modes);
  result.payload_sha256.reserve(row_paths.size());

  std::vector<ScaledSumSquares> ic_sumsq(n_pixels);
  std::vector<ScaledSumSquares> w_sumsq(n_pixels);
  std::vector<double> mu_loo_mean(n_pixels, 0.0);
  std::vector<std::uint8_t> row(row_size);

  const double inv_n = 1.0 / static_cast<double>(n_modes);
  const double ic_ref = total_ic[ref];
  const auto pass1_start = std::chrono::steady_clock::now();
  for (std::size_t part = 0; part != row_paths.size(); ++part) {
    auto in = open_payload(row_paths[part]);
    Sha256 digest;
    for (std::size_t mode = 0; mode != mode_counts[part]; ++mode) {
      read_exact(in, row, row_paths[part], mode);
      digest.update(row.data(), row.size());
      const double row_ic_ref =
          get_double_le(row.data() + ref * kCellBytes + 16);
      if (!std::isfinite(row_ic_ref)) {
        throw std::runtime_error("non-finite reference Ic row in " +
                                 row_paths[part]);
      }
      const double loo_ic_ref = ic_ref - row_ic_ref;
      result.ic_ref_loo.push_back(loo_ic_ref);
      for (std::size_t pixel = 0; pixel != n_pixels; ++pixel) {
        const auto *cell = row.data() + pixel * kCellBytes;
        const double wr = get_double_le(cell);
        const double wi = get_double_le(cell + 8);
        const double ic = get_double_le(cell + 16);
        if (!std::isfinite(wr) || !std::isfinite(wi) || !std::isfinite(ic)) {
          throw std::runtime_error("non-finite Stage-14 row value in " +
                                   row_paths[part] + " at local mode " +
                                   std::to_string(mode) + ", pixel " +
                                   std::to_string(pixel));
        }

        const double dic = ic - total_ic[pixel] * inv_n;
        const double dwr = wr - total_w_re[pixel] * inv_n;
        const double dwi = wi - total_w_im[pixel] * inv_n;
        ic_sumsq[pixel].add(dic);
        w_sumsq[pixel].add(dwr);
        w_sumsq[pixel].add(dwi);

        const double loo_ic = total_ic[pixel] - ic;
        const double loo_wr = total_w_re[pixel] - wr;
        const double loo_wi = total_w_im[pixel] - wi;
        if (std::isfinite(loo_ic) && std::isfinite(loo_ic_ref) &&
            std::isfinite(loo_wr) && std::isfinite(loo_wi) && loo_ic > 0.0 &&
            loo_ic_ref > 0.0) {
          const double mu =
              range_safe_mu(loo_wr, loo_wi, loo_ic, loo_ic_ref);
          if (std::isfinite(mu)) {
            auto &count = result.n_mu_loo_valid[pixel];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
              throw std::overflow_error("n_mu_loo_valid overflows uint32");
            }
            ++count;
            mu_loo_mean[pixel] +=
                (mu - mu_loo_mean[pixel]) / static_cast<double>(count);
          }
        }
      }
    }
    if (in.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error("trailing bytes in Stage-14 payload: " +
                               row_paths[part]);
    }
    const std::string actual_hash = digest.hex_digest();
    if (!expected_sha256.empty() &&
        lower_ascii(expected_sha256[part]) != actual_hash) {
      throw std::runtime_error("Stage-14 payload SHA-256 mismatch for " +
                               row_paths[part]);
    }
    result.payload_sha256.push_back(actual_hash);
  }
  const auto pass1_end = std::chrono::steady_clock::now();
  result.pass1_seconds =
      std::chrono::duration<double>(pass1_end - pass1_start).count();

  for (std::size_t pixel = 0; pixel != n_pixels; ++pixel) {
    const double ic_error = ic_sumsq[pixel].root(factor);
    const double w_error = w_sumsq[pixel].root(factor);
    if (!std::isfinite(ic_error) || !std::isfinite(w_error)) {
      throw std::overflow_error("Stage-14 centered variance is non-finite");
    }
    result.ic_err[pixel] = ic_error;
    result.w_err[pixel] = w_error;
    if (total_ic[pixel] > 0.0 && ic_ref > 0.0) {
      const double mu = range_safe_mu(total_w_re[pixel], total_w_im[pixel],
                                      total_ic[pixel], ic_ref);
      if (std::isfinite(mu)) {
        result.mu_raw[pixel] = mu;
        result.mu_raw_defined[pixel] = 1;
      }
    }
  }

  std::vector<ScaledSumSquares> mu_sumsq(n_pixels);
  const auto pass2_start = std::chrono::steady_clock::now();
  for (std::size_t part = 0; part != row_paths.size(); ++part) {
    const std::uint64_t pass2_size = file_size_strict(row_paths[part]);
    if (pass2_size != result.payload_bytes[part]) {
      throw std::runtime_error(
          "Stage-14 payload size changed between finalize passes for " +
          row_paths[part]);
    }
    auto in = open_payload(row_paths[part]);
    Sha256 digest;
    for (std::size_t mode = 0; mode != mode_counts[part]; ++mode) {
      read_exact(in, row, row_paths[part], mode);
      digest.update(row.data(), row.size());
      const double row_ic_ref =
          get_double_le(row.data() + ref * kCellBytes + 16);
      const double loo_ic_ref = ic_ref - row_ic_ref;
      for (std::size_t pixel = 0; pixel != n_pixels; ++pixel) {
        if (result.n_mu_loo_valid[pixel] != n_modes) {
          continue;
        }
        const auto *cell = row.data() + pixel * kCellBytes;
        const double wr = get_double_le(cell);
        const double wi = get_double_le(cell + 8);
        const double ic = get_double_le(cell + 16);
        const double loo_ic = total_ic[pixel] - ic;
        const double loo_wr = total_w_re[pixel] - wr;
        const double loo_wi = total_w_im[pixel] - wi;
        if (!(std::isfinite(loo_ic) && std::isfinite(loo_ic_ref) &&
              std::isfinite(loo_wr) && std::isfinite(loo_wi) && loo_ic > 0.0 &&
              loo_ic_ref > 0.0)) {
          throw std::runtime_error(
              "LOO validity changed between Stage-14 finalize passes");
        }
        const double mu =
            range_safe_mu(loo_wr, loo_wi, loo_ic, loo_ic_ref);
        if (!std::isfinite(mu)) {
          throw std::runtime_error(
              "LOO numeric validity changed between Stage-14 finalize passes");
        }
        mu_sumsq[pixel].add(mu - mu_loo_mean[pixel]);
      }
    }
    if (in.peek() != std::char_traits<char>::eof()) {
      throw std::runtime_error(
          "trailing bytes in Stage-14 payload during pass 2: " +
          row_paths[part]);
    }
    const std::string pass2_hash = digest.hex_digest();
    if (pass2_hash != result.payload_sha256[part] ||
        (!expected_sha256.empty() &&
         pass2_hash != lower_ascii(expected_sha256[part]))) {
      throw std::runtime_error(
          "Stage-14 payload SHA-256 changed between finalize passes for " +
          row_paths[part]);
    }
  }
  const auto pass2_end = std::chrono::steady_clock::now();
  result.pass2_seconds =
      std::chrono::duration<double>(pass2_end - pass2_start).count();

  for (std::size_t pixel = 0; pixel != n_pixels; ++pixel) {
    if (result.n_mu_loo_valid[pixel] == n_modes) {
      const double error = mu_sumsq[pixel].root(factor);
      if (!std::isfinite(error)) {
        throw std::overflow_error("Stage-14 raw-mu variance is non-finite");
      }
      result.mu_raw_err[pixel] = error;
      result.mu_raw_err_defined[pixel] = 1;
    }
  }
  const std::uint64_t one_pass = [&]() {
    std::uint64_t total = 0;
    for (const auto size : result.payload_bytes) {
      total = checked_u64_add(total, size, "Stage-14 bytes read");
    }
    return total;
  }();
  result.bytes_read = checked_u64_add(one_pass, one_pass,
                                      "Stage-14 bytes read");
  return result;
}

}  // namespace

void register_stage14(py::module_ &m) {
  py::class_<Stage14Store>(m, "Stage14Store")
      .def(py::init<std::string, std::size_t, std::size_t, std::size_t,
                    std::vector<double>, std::vector<double>>(),
           py::arg("path"), py::arg("n_modes"), py::arg("n_pixels"),
           py::arg("ref"), py::arg("kms"), py::arg("weights"))
      .def("begin_mode", &Stage14Store::begin_mode, py::arg("mode"))
      .def("add_ray", &Stage14Store::add_ray, py::arg("pixel"),
           py::arg("opl"), py::arg("amps"))
      .def("fold_mode", &Stage14Store::fold_mode)
      .def("finish", &Stage14Store::finish);

  m.def(
      "stage14_finalize",
      [](const std::vector<std::string> &row_paths,
         const std::vector<std::size_t> &mode_counts, std::size_t n_pixels,
         std::size_t ref, const py::bytes &i_bytes,
         const py::bytes &w_re_bytes, const py::bytes &w_im_bytes,
         const py::bytes &ic_bytes,
         const std::vector<std::string> &expected_sha256) {
        const auto total_i = doubles_from_bytes(i_bytes, n_pixels, "I");
        const auto total_w_re =
            doubles_from_bytes(w_re_bytes, n_pixels, "w_re");
        const auto total_w_im =
            doubles_from_bytes(w_im_bytes, n_pixels, "w_im");
        const auto total_ic = doubles_from_bytes(ic_bytes, n_pixels, "ic");
        FinalizeResult result;
        {
          py::gil_scoped_release release;
          result = finalize_impl(row_paths, mode_counts, n_pixels, ref, total_i,
                                 total_w_re, total_w_im, total_ic,
                                 expected_sha256);
        }
        py::dict out;
        out["ic_err"] = le_bytes(result.ic_err);
        out["w_err"] = le_bytes(result.w_err);
        out["mu_raw"] = le_bytes(result.mu_raw);
        out["mu_raw_err"] = le_bytes(result.mu_raw_err);
        out["n_mu_loo_valid"] = le_bytes(result.n_mu_loo_valid);
        out["mu_raw_defined"] = le_bytes(result.mu_raw_defined);
        out["mu_raw_err_defined"] = le_bytes(result.mu_raw_err_defined);
        out["ic_ref_loo"] = le_bytes(result.ic_ref_loo);
        out["payload_sha256"] = result.payload_sha256;
        out["payload_bytes"] = result.payload_bytes;
        out["bytes_read"] = py::int_(result.bytes_read);
        out["pass1_seconds"] = result.pass1_seconds;
        out["pass2_seconds"] = result.pass2_seconds;
        return out;
      },
      py::arg("row_paths"), py::arg("mode_counts"), py::arg("n_pixels"),
      py::arg("ref"), py::arg("I"), py::arg("w_re"), py::arg("w_im"),
      py::arg("ic"),
      py::arg("expected_sha256") = std::vector<std::string>{},
      R"pbdoc(Finalize compatible Stage-14 payload parts in two centered passes.

All dense inputs and outputs are canonical little-endian byte strings.  The
first pass validates exact file sizes, finite row values, and (when supplied)
each SHA-256.  Undefined nonlinear values contain NaN internally and are
paired with explicit uint8 masks; the Python result serializer must emit JSON
null for them.)pbdoc");
}
