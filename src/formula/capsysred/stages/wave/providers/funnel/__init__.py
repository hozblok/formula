"""Stage 16 provider fb for conformal funnel bores (kind funnel, f = g): every bore of the scene scales about the device
axis with G(z) = 1 + g0 z + g1 z^2 (z from the entrance): the axis of a bore is G(z) c0, its radius a G(z). The field
of a bore lives in the frame of the bore, rho = (r - G c0) / S(z), scaled with its effective Dirichlet radius,
S = (a G + Re ell) / A0, so the wall is the fixed disc A0 = a + Re ell of the Fourier-Bessel basis:
  E(r, z) = S^-1 exp(i k [S' |x|^2 / 2S + G' c0.x + G G' |c0|^2 / 2]) w(rho, z),   x = r - G c0,
  i dw/dz = -Lap_rho w / (2 k S^2) + [V + (k/2) S S'' |rho|^2 + k S G'' c0.rho + (k/2) G G'' |c0|^2] w.
The kinetic part with the wall has the eigenvalues of the physical bore of radius a G(z): fb_wall exact-roots takes the
roots of the scalar characteristic equation, effective-radius the zero field at the radius a G + ell, ell the complex
penetration depth, kappa = j / (a G + ell).
fb_integrator cf4: the lens and the axis term enter as Galerkin blocks (r^2 within m, r e^{+-i phi} between m and
m +- 1); a step is the commutator-free exponential integrator of order 4 (two exponentials of combinations of H at
the Gauss points), each exponential a Chebyshev series.
fb_integrator split-step: a step is exp(-i dz/2 D) exp(-i dz nu K) exp(-i dz/2 D), order 2: D the diagonal of the modes,
nu K = k g1 S [(a / A0) |rho|^2 + 2 c0.rho] the lens and the axis term as one multiplication on the (r, phi) grid; no
Galerkin block and no matrix exponential. The projection back on the modes drops what the product puts above jmax; that
power (projection_loss, summed over the steps) falls as 1 / steps only, so the order is 2 while it stays below the
wanted error: the top of the basis has to be empty. The basis keeps every zero j_mn < jmax; its outer fraction
`layer` absorbs (amplitude rate gamma_max ((j - j_s) / (jmax - j_s))^2). The Kirchhoff edge tail of the entrance is
fixed in j (the frame keeps j while the physical angle falls as 1 / G), so jmax has to cover the guided modes of the
entrance, k theta_c a.
Kirchhoff entrance: the paraxial point source inside the geometric aperture a. Exit plane: the field inside the
geometric exit core a G(L) at the screen pixels (observable point)."""
