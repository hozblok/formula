"""Stage 16 provider `fb`: Fourier-Bessel modes of circular bores (cylinder, torus arc).

Paraxial scalar wave in the co-moving frame of the bent axis c(z) = c0 + t z^2 / (2R)
(t = unit vector toward the bend centre):  i d/dz psi = -(1/2k) Lap psi + (k/R) xi psi,
xi along t; Dirichlet wall at r = a + Re(ell), ell = i / (k q0), q0 = sqrt(-2 delta + 2 i beta)
(DIR-ell); Im(ell) enters as the first-order per-mode damping kappa^2 Im(ell) / (a k).
Longitudinal step: a Chebyshev series of exp(-iHL) (no z error; z-independent H).  Exit field in the lab frame on a Cartesian lattice with the phase
exp(i k c'(L).x' + i g), g = k L^3 / (6 R^2); screens by angular-spectrum drift on the
lattice and exact cell integrals of the band-limited field (Fourier box), reference cell as
Stage 14.  Conventions as the uisk provider: W = <E(P) E*(P_ref)>, e^{-i omega t}, carrier
e^{+ikz} dropped; entrance field exp(ik|x - s|^2 / 2 z_in) / (i lambda z_in).
Options: wall `exact-roots` (per-mode roots of the J_m / K_m characteristic equation of the hollow
cylinder instead of DIR-ell), a wall figure dr(phi, z) to first order (Hadamard) with the q = 1
part as an exact co-moving axis path, the far field from the exit modes (Lommel integrals) and
a dump of the entrance / exit mode coefficients.
"""
