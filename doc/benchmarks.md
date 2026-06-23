🇬🇧 **English** · [🇷🇺 Русский](benchmarks.ru.md)

# Benchmarks

Speed measurements for `formula`, each in its own file with the code to reproduce
it, the environment, and the date.

- [Raw arithmetic vs the field](benchmark-raw-arithmetic.md) — one multiply
  against `gmpy2`, `decimal`, `mpmath` at matched decimal digits.
- [Transcendentals](benchmark-transcendentals.md) — `sin` is the real
  high-precision cost, not multiplication (~100x behind `gmpy2`-backed `mpmath`).
- [End to end vs SymPy](benchmark-end-to-end-sympy.md) — parse a string, get the
  value and the derivative at a point to N digits.
- [cpp_dec_float scaling](benchmark-dec-float-scaling.md) — how the Boost type
  scales with precision; the data behind the ladder's ceiling.

Design rationale that uses these numbers:
[the precision ladder](precision-ladder.md).
