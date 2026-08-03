"""
SG_helpers.py
=============
Helper functions for Savitzky-Golay (SG) derivative estimation and the
associated correction factor Phi_SG for the MDCK force inference model.

Four functions
--------------
1. stencil_coefficients(m, p)
2. calculate_derivatives_sg(data, dt, m, p, S=1)
3. calculate_integral(k, l, S, dt, gamma=-1e-4)
4. correction_term_sg(m, p, S=1, dt=1.0)

Typical usage
-------------
    from SG_helpers import calculate_derivatives_sg, correction_term_sg

    m = int((smooth_window - 1) / 2)   # e.g. smooth_window=9 → m=4
    p = smooth_polyorder                # e.g. 3

    data_clean = calculate_derivatives_sg(data, dt=dt, m=m, p=p, S=1)
    phi_sg     = correction_term_sg(m=m, p=p, S=1)
    model      = create_model(hidden_dim=64, tau_init=tau_mem0, phi_sg=phi_sg)
"""

import numpy as np
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.integrate import dblquad
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. stencil_coefficients
# ──────────────────────────────────────────────────────────────────────────────

def stencil_coefficients(m, p):
    """
    Return the SG velocity and acceleration stencil coefficients.

    For a window of length 2m+1 and polynomial degree p, the SG filter
    computes analytical polynomial derivatives. Coefficients are indexed so
    that coeff[m + k] corresponds to frame offset k ∈ {-m, …, m}.

    Parameters
    ----------
    m : int   SG half-window size; window = 2m+1
    p : int   SG polynomial degree (must satisfy p < 2m+1)

    Returns
    -------
    cv : ndarray, shape (2m+1,)
        Velocity stencil (deriv=1, delta=1). Antisymmetric: cv[m+k] = -cv[m-k].
    ca : ndarray, shape (2m+1,)
        Acceleration stencil (deriv=2, delta=1). Symmetric: ca[m+k] = ca[m-k].
    """
    window = 2 * m + 1
    cv = savgol_coeffs(window, p, deriv=1, delta=1.0, use='dot')
    ca = savgol_coeffs(window, p, deriv=2, delta=1.0, use='dot')
    return cv, ca


# ──────────────────────────────────────────────────────────────────────────────
# 2. calculate_derivatives_sg
# ──────────────────────────────────────────────────────────────────────────────

def calculate_derivatives_sg(data, dt, m, p, S=1):
    """
    Drop-in replacement for calculate_derivatives() using a Savitzky-Golay filter.

    Fits a local polynomial of degree p over a sliding window of 2m+1 frames,
    then extracts analytical derivatives from the polynomial:

        x, y   — SG-smoothed positions for all frames  (deriv=0)
        vx, vy — SG velocity at frame t-S              (deriv=1)
        ax, ay — SG acceleration at frame t            (deriv=2)

    All frames for each particle are kept in the output. Boundary frames that
    lack a valid SG window (local index t < m+S or t > n-1-m) have their
    derivatives set to zero. build_graph() masks out any node with zero
    ax/ay/vx/vy, so boundary nodes act as neighbours only (contributing
    forces on others) without entering the training loss.

    Particles shorter than the window length (2m+1) are dropped entirely
    since no smoothed position estimate is possible.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: frame, particle, x, y.
    dt   : float  Time between consecutive frames.
    m    : int    SG half-window size; window = 2m+1.
    p    : int    SG polynomial degree.
    S    : int    Velocity shift in frames (default 1).

    Returns
    -------
    pd.DataFrame
        All rows kept (except particles shorter than window), with columns:
        x, y (smoothed), vx, vy, ax, ay (zero at boundary frames).
    """
    if S == 0:
        print('WARNING: S=0 is degenerate — bias estimator is zero by parity. '
              'Use S >= 1.')

    window = 2 * m + 1
    sg_kw  = dict(window_length=window, polyorder=p, delta=dt)

    parts_ok      = 0
    parts_skip    = 0
    n_boundary    = 0
    n_interior    = 0
    records       = []

    for pid, grp in data.groupby('particle'):
        grp = grp.sort_values('frame')
        n   = len(grp)

        if n < window:
            # Can't smooth at all — drop entirely
            parts_skip += 1
            continue

        xs = grp['x'].to_numpy(dtype=float)
        ys = grp['y'].to_numpy(dtype=float)

        # Smoothed positions for all frames
        x_smooth = savgol_filter(xs, deriv=0, **sg_kw)
        y_smooth = savgol_filter(ys, deriv=0, **sg_kw)

        # Derivative arrays (only valid in interior, but computed for all frames)
        ax_all = savgol_filter(xs, deriv=2, **sg_kw)
        ay_all = savgol_filter(ys, deriv=2, **sg_kw)
        vx_all = savgol_filter(xs, deriv=1, **sg_kw)
        vy_all = savgol_filter(ys, deriv=1, **sg_kw)

        # Interior range: a[t] valid at t ∈ [m, n-1-m]
        #                 v[t-S] valid at t ∈ [m+S, n-1-m+S]
        #                 Combined: t ∈ [m+S, n-1-m]
        t_min = m + S
        t_max = n - 1 - m

        rows = grp.copy()
        rows['x']  = x_smooth
        rows['y']  = y_smooth
        rows['vx'] = 0.0
        rows['vy'] = 0.0
        rows['ax'] = 0.0
        rows['ay'] = 0.0

        if t_min <= t_max:
            interior = rows.index[t_min:t_max + 1]
            rows.loc[interior, 'ax'] = ax_all[t_min:t_max + 1]
            rows.loc[interior, 'ay'] = ay_all[t_min:t_max + 1]
            rows.loc[interior, 'vx'] = vx_all[t_min - S:t_max + 1 - S]
            rows.loc[interior, 'vy'] = vy_all[t_min - S:t_max + 1 - S]
            n_interior += (t_max - t_min + 1)
            n_boundary += n - (t_max - t_min + 1)
        else:
            # Track too short for any valid interior frame — all boundary
            n_boundary += n

        records.append(rows)
        parts_ok += 1

    if not records:
        raise ValueError(
            f'No particles had enough frames for the SG filter '
            f'(need >= {window} frames per particle).'
        )

    data_clean = (pd.concat(records)
                    .sort_values(['frame', 'particle'])
                    .reset_index(drop=True))

    interior_mask = (data_clean['ax'] != 0) | (data_clean['ay'] != 0)
    print(f'SG derivatives: window={window}, polyorder={p}, shift S={S}')
    print(f'  Particles processed: {parts_ok}  (skipped too-short: {parts_skip})')
    print(f'  Interior frames (train): {n_interior}  '
          f'Boundary frames (neighbour-only): {n_boundary}')
    print(f'  ax: mean={data_clean.loc[interior_mask, "ax"].mean():.4f},  '
          f'std={data_clean.loc[interior_mask, "ax"].std():.4f}  (interior only)')
    return data_clean


# ──────────────────────────────────────────────────────────────────────────────
# 3. calculate_integral
# ──────────────────────────────────────────────────────────────────────────────

def calculate_integral(k, l, S, dt, gamma=-1e-4):
    """
    Compute I(k, l, S) — the double integral entering the Phi_SG correction.

    This is the G-function from Appendix 5, evaluated at a small |gamma| to
    give the eps = |gamma|·dt → 0 limit:

        I(k, l, S) = sign(k) · sign(l)
                     · ∫_0^{|k|·dt} ∫_0^{|l|·dt}
                         R_v(S·dt + sign(k)·u − sign(l)·w) du dw

    where R_v(τ) = exp(−|gamma|·|τ|) is the normalised OU velocity
    autocorrelation (with σ²/(2|gamma|) = 1).

    Returns 0 when k=0 or l=0 (no stencil contribution from those offsets).

    Parameters
    ----------
    k, l  : int    Stencil offsets in {-m, …, m}.
    S     : int    Velocity shift (frames).
    dt    : float  Frame interval.
    gamma : float  Small negative value for the eps→0 limit (default -1e-4).

    Returns
    -------
    float   I(k, l, S)
    """
    if k == 0 or l == 0:
        return 0.0

    def R_v(tau):
        # Normalised OU autocorrelation with σ²/(2|γ|) = 1
        return np.exp(-abs(gamma) * abs(tau))

    sk = int(np.sign(k))
    sl = int(np.sign(l))

    val, _ = dblquad(
        lambda w, u: R_v(S * dt + sk * u - sl * w),
        0.0, abs(k) * dt,
        0.0, abs(l) * dt,
        epsabs=1e-9, epsrel=1e-9,
    )
    return sk * sl * val


# ──────────────────────────────────────────────────────────────────────────────
# 4. correction_term_sg
# ──────────────────────────────────────────────────────────────────────────────

def correction_term_sg(m, p, S=1, dt=1.0):
    """
    Compute the leading-order SG correction factor Phi_SG.

    Phi_SG = gamma_star / gamma is the ratio of the effective drag recovered
    by the SG-based estimator to the true drag, in the eps → 0 limit.
    Passing phi_sg to create_model() ensures the forward model applies the
    correct bias correction.

    Derivation (Appendix 5):
        D = Σ_{l1,l2} c_v(l1)·c_v(l2)·I(l1, l2, 0) / dt²   (velocity variance)
        N = Σ_{k,l}   c_a(k) ·c_v(l) ·I(k,  l,  S) / dt³   (a-v cross-term)
        Phi_SG = (N / D) / gamma

    Reference values:
        Naive finite differences:  Phi_SG = 2/3 ≈ 0.667
        m=2, p=2, S=1:             Phi_SG ≈ 0.510  (≈ 107/210)
        m=4, p=3, S=1:             Phi_SG ≈ 0.5xx  (computed numerically)

    Parameters
    ----------
    m  : int    SG half-window size (window = 2m+1).
    p  : int    SG polynomial degree.
    S  : int    Velocity shift in frames (default 1; must be >= 1).
    dt : float  Frame interval (default 1.0). Phi_SG is independent of dt
                to leading order, so the default is fine for any dt.

    Returns
    -------
    phi_sg : float

    Example
    -------
        phi_sg = correction_term_sg(m=int((smooth_window - 1) / 2),
                                    p=smooth_polyorder, S=1)
        model  = create_model(hidden_dim=64, tau_init=tau_mem0, phi_sg=phi_sg)
    """
    if S == 0:
        raise ValueError('S=0 is degenerate (parity cancellation). Use S >= 1.')

    gamma = -1e-4   # small |gamma| for eps → 0 limit; shared with calculate_integral

    cv, ca = stencil_coefficients(m, p)

    def c_v(l_): return float(cv[m + l_])
    def c_a(k_): return float(ca[m + k_])

    rng = range(-m, m + 1)

    # ── Denominator D: velocity variance ─────────────────────────────────────
    D = sum(
        c_v(l1) * c_v(l2) * calculate_integral(l1, l2, 0, dt, gamma)
        for l1 in rng for l2 in rng
        if c_v(l1) != 0 and c_v(l2) != 0
    ) / dt**2

    if abs(D) < 1e-15:
        raise ValueError(
            f'D ≈ 0 for m={m}, p={p}. Check that p < 2m+1 and m >= 1.'
        )

    # ── Numerator N: acceleration-velocity cross-correlation ─────────────────
    N = sum(
        c_a(k) * c_v(l) * calculate_integral(k, l, S, dt, gamma)
        for k in rng for l in rng
        if c_a(k) != 0 and c_v(l) != 0
    ) / dt**3

    gamma_star = N / D
    phi_sg     = gamma_star / gamma

    if abs(phi_sg) < 1e-6:
        raise ValueError(
            f'Phi_SG ≈ 0 for S={S} — degenerate result. Use S >= 1.'
        )

    print(f'Phi_SG = {phi_sg:.6f}  '
          f'(m={m}, p={p}, S={S};  1/Phi_SG = {1/phi_sg:.4f})')
    return float(phi_sg)
