"""
inference_helpers_simple.py
===========================
Data pipeline + stencil constants for the pairwise overdamped-with-memory
inference on (non-periodic) cell-tracking data, e.g. MDCK.

    load_data                 read a CSV with columns frame, particle, x, y
    calculate_derivatives_sg  gap-aware Savitzky-Golay position/velocity/acceleration
    estimate_lambda           localization-noise variance from raw positions
    build_graph               one frame -> nodes, edges, edge features, loss mask
    prepare_dataset           all frames -> list of graphs + normalisation constants
    split_dataset             seeded train/val/test split over frames
    sg_stencils, sg_position_stencil, Kav, Kaa, phi_factor, phi_sg
                              S-G stencil coefficients and the correction constants
                              K_av, K_aa (Method.pdf SI VI C, Eqs. 22, 29, 13)

Conventions
-----------
* Lengths are in whatever unit x, y are given in (microns in the MDCK notebook),
  times in the unit of dt (hours).
* Edge [j, i] (row 0 = source j, row 1 = target i) carries r_ij = |x_i - x_j| and
  the unit vector r_hat_ij = (x_i - x_j)/r_ij; the force f(r) r_hat_ij acts on i.
* A node enters the loss only if it has valid derivatives ('valid' column) and is
  more than `edge_margin` from the field-of-view edge. All other nodes are kept in
  the graph as neighbours.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(datadir, dataname):
    """Load a trajectory CSV (columns: frame, particle, x, y)."""
    data = pd.read_csv(Path(datadir) / dataname)
    print(f"Data loaded: {data.shape}, columns {list(data.columns)}, "
          f"{data['frame'].nunique()} frames")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Contiguous track segments (shared by the S-G filter and the Lambda estimator)
# ─────────────────────────────────────────────────────────────────────────────

def _frame_step(data):
    d = data.sort_values(['particle', 'frame']).groupby('particle')['frame'].diff().dropna()
    return d.mode().iloc[0] if len(d) else 1


def _segments(data, frame_step):
    """Yield contiguous (no missing frame) pieces of every track, sorted by frame."""
    for _, grp in data.groupby('particle'):
        grp = grp.sort_values('frame').reset_index(drop=True)
        gaps = np.where(np.diff(grp['frame'].to_numpy()) != frame_step)[0]
        bounds = [0] + list(gaps + 1) + [len(grp)]
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            yield grp.iloc[b0:b1].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Savitzky-Golay derivatives
# ─────────────────────────────────────────────────────────────────────────────

def calculate_derivatives_sg(data, dt, m, p, S=1):
    """
    Gap-aware S-G derivatives (window 2m+1, degree p, velocity shift S).

    Each track is split at missing frames and every contiguous segment is filtered
    on its own, so no stencil straddles a gap. For a segment of length n, frame t
    gets
        x, y   : S-G smoothed position at t
        vx, vy : S-G velocity at t - S
        ax, ay : S-G acceleration at t
    and is marked valid for t in [m+S, n-1-m]. Other frames (segment ends, and
    whole segments shorter than the window) keep zero derivatives and valid=False;
    segments shorter than the window also keep their raw (unsmoothed) position.
    The raw positions are always kept in x_raw, y_raw (used by estimate_lambda).
    """
    window = 2 * m + 1
    sg_kw = dict(window_length=window, polyorder=p, delta=dt)
    step = _frame_step(data)

    records, n_seg, n_short = [], 0, 0
    for seg in _segments(data, step):
        n_seg += 1
        rows = seg.copy()
        rows['x_raw'] = seg['x'].to_numpy(dtype=float)
        rows['y_raw'] = seg['y'].to_numpy(dtype=float)
        for c in ('vx', 'vy', 'ax', 'ay'):
            rows[c] = 0.0
        rows['valid'] = False
        n = len(seg)
        if n < window:
            n_short += 1
            records.append(rows)
            continue

        xs, ys = rows['x_raw'].to_numpy(), rows['y_raw'].to_numpy()
        rows['x'] = savgol_filter(xs, deriv=0, **sg_kw)
        rows['y'] = savgol_filter(ys, deriv=0, **sg_kw)
        vx, vy = savgol_filter(xs, deriv=1, **sg_kw), savgol_filter(ys, deriv=1, **sg_kw)
        ax, ay = savgol_filter(xs, deriv=2, **sg_kw), savgol_filter(ys, deriv=2, **sg_kw)

        t0, t1 = m + S, n - 1 - m
        if t0 <= t1:
            sl = slice(t0, t1 + 1)
            rows.loc[t0:t1, 'ax'] = ax[sl]
            rows.loc[t0:t1, 'ay'] = ay[sl]
            rows.loc[t0:t1, 'vx'] = vx[t0 - S:t1 + 1 - S]
            rows.loc[t0:t1, 'vy'] = vy[t0 - S:t1 + 1 - S]
            rows.loc[t0:t1, 'valid'] = True
        records.append(rows)

    out = pd.concat(records).sort_values(['frame', 'particle']).reset_index(drop=True)
    v = out['valid']
    print(f"S-G derivatives (window={window}, p={p}, S={S}): {n_seg} contiguous segments "
          f"({n_short} shorter than the window, kept as neighbour-only); "
          f"valid frames {int(v.sum())}, neighbour-only frames {int((~v).sum())}; "
          f"ax std (valid) = {out.loc[v, 'ax'].std():.4f}")
    return out


def estimate_lambda(data_clean, dt):
    """
    Localization-noise variance per Cartesian component from the RAW positions
    (x_raw, y_raw), Bruckner et al. PRL 125, 058103 (2020), SM Eq. (S71), pooled over
    every contiguous segment of >= 4 frames with weight 2(n-3). Same units as x^2.
    (dt is not needed by the Lambda estimator itself; kept for a uniform signature.)
    """
    step = _frame_step(data_clean)
    num = den = 0.0
    for seg in _segments(data_clean, step):
        n = len(seg)
        if n < 4:
            continue
        pos = np.vstack([seg['x_raw'].to_numpy(float), seg['y_raw'].to_numpy(float)])
        dm = pos[:, 1:-2] - pos[:, :-3]
        d0 = pos[:, 2:-1] - pos[:, 1:-2]
        dp = pos[:, 3:] - pos[:, 2:-1]
        lam = np.mean(10 * d0 * d0 + dm * dm + dp * dp + 8 * dp * dm
                      - 10 * d0 * dp - 10 * d0 * dm) / 44.0
        w = 2 * (n - 3)
        num += lam * w
        den += w
    return num / den if den > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Graphs
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(frame_data, cutoff, edge_margin=0.0, fov_bounds=None):
    """
    One frame -> (nodes [x, y, vx, vy], edges (2, E) [source j, target i],
    edge_feat (E, 3) [r_ij, r_hat_ij], mask (N,)). Pairs with 0 < r < cutoff are
    connected in both directions.
    """
    pos = frame_data[['x', 'y']].to_numpy(dtype=float)
    nodes = frame_data[['x', 'y', 'vx', 'vy']].to_numpy(dtype=np.float32)

    pairs = cKDTree(pos).query_pairs(cutoff, output_type='ndarray')
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        d = pos[b] - pos[a]
        r = np.linalg.norm(d, axis=1)
        keep = (r > 0) & (r < cutoff)
        a, b, d, r = a[keep], b[keep], d[keep], r[keep]
        u = d / r[:, None]                        # unit vector a -> b  (= r_hat_{b a})
        edges = np.concatenate([np.stack([a, b]), np.stack([b, a])], axis=1)
        edge_feat = np.concatenate([np.column_stack([r, u]), np.column_stack([r, -u])])
    else:
        edges = np.zeros((2, 0), dtype=np.int64)
        edge_feat = np.zeros((0, 3))

    mask = frame_data['valid'].to_numpy(dtype=bool)
    if edge_margin > 0:
        x0, x1, y0, y1 = fov_bounds
        mask &= ((pos[:, 0] > x0 + edge_margin) & (pos[:, 0] < x1 - edge_margin) &
                 (pos[:, 1] > y0 + edge_margin) & (pos[:, 1] < y1 - edge_margin))

    return nodes, edges.astype(np.int64), edge_feat.astype(np.float32), mask.astype(np.float32)


def prepare_dataset(data_clean, cutoff, max_frames=None, edge_margin=0.0, fov_bounds=None):
    """
    Build one graph per frame and normalise.

    fov_bounds = (x_min, x_max, y_min, y_max) of the field of view for the edge mask
    (default: extent of data_clean). Normalisation, pooled over all frames:
        positions  (x - x_mean) / x_std,  x_std = sqrt(std_x^2 + std_y^2)
        distances  r / r_std               (not centred: the model needs f(r)/r)
        targets    a / y_std               (single scalar over both components)
    velocities stay in physical units.

    Returns (dataset, norm): dataset is a list of dicts with keys
    frame, nodes, edges, edge_feat, targets, mask; norm = dict(x_mean, x_std, r_std, y_std).
    """
    frames = sorted(data_clean['frame'].unique())[:max_frames]
    if edge_margin > 0 and fov_bounds is None:
        fov_bounds = (data_clean['x'].min(), data_clean['x'].max(),
                      data_clean['y'].min(), data_clean['y'].max())

    dataset = []
    for fr in frames:
        fd = data_clean[data_clean['frame'] == fr].reset_index(drop=True)
        nodes, edges, edge_feat, mask = build_graph(fd, cutoff, edge_margin, fov_bounds)
        dataset.append(dict(frame=fr, nodes=nodes, edges=edges, edge_feat=edge_feat,
                            targets=fd[['ax', 'ay']].to_numpy(dtype=np.float32), mask=mask))

    all_xy = np.vstack([d['nodes'][:, :2] for d in dataset])
    all_r = np.concatenate([d['edge_feat'][:, 0] for d in dataset])
    all_a = np.vstack([d['targets'] for d in dataset])
    x_mean = all_xy.mean(axis=0)
    x_std = float(np.sqrt((all_xy.std(axis=0) ** 2).sum())) or 1.0
    r_std = float(all_r.std()) or 1.0
    y_std = float((all_a - all_a.mean(axis=0)).std()) or 1.0
    for d in dataset:
        d['nodes'][:, :2] = (d['nodes'][:, :2] - x_mean) / x_std
        d['edge_feat'][:, 0] /= r_std
        d['targets'] = d['targets'] / y_std
    norm = dict(x_mean=x_mean, x_std=x_std, r_std=r_std, y_std=y_std)

    n_in = sum(d['mask'].sum() for d in dataset)
    n_all = sum(len(d['mask']) for d in dataset)
    msg = f"{len(dataset)} frames, {n_all} nodes, {n_in:.0f} in the loss ({100 * n_in / n_all:.1f}%)"
    if edge_margin > 0:
        msg += (f"; edge mask {edge_margin} from FOV x=[{fov_bounds[0]:.1f}, {fov_bounds[1]:.1f}], "
                f"y=[{fov_bounds[2]:.1f}, {fov_bounds[3]:.1f}]")
    print("Dataset: " + msg)
    print(f"  max r = {all_r.max():.2f}, x_std = {x_std:.4f}, r_std = {r_std:.4f}, y_std = {y_std:.4f}")
    return dataset, norm


def split_dataset(dataset, train_frac=0.7, val_frac=0.15, seed=None):
    """Shuffle frames with RandomState(seed) (without touching `dataset`) and split."""
    order = list(dataset)
    np.random.RandomState(seed).shuffle(order)
    n_tr, n_va = int(train_frac * len(order)), int(val_frac * len(order))
    train, val, test = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    print(f"Split: train {len(train)}, val {len(val)}, test {len(test)} frames")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Stencils and correction constants
# ─────────────────────────────────────────────────────────────────────────────

def sg_stencils(window, d, S):
    """S-G acceleration stencil (offsets -m..m) and velocity stencil shifted by S
    (offsets -m-S..m-S), unit spacing: returns ca, qa, cv, qv."""
    m = (window - 1) // 2
    ks = list(range(-m, m + 1))
    ca = list(savgol_coeffs(window, d, deriv=2, delta=1.0, use='dot'))
    cv = list(savgol_coeffs(window, d, deriv=1, delta=1.0, use='dot'))
    return ca, ks, cv, [k - S for k in ks]


def sg_position_stencil(window, d):
    """S-G smoothing (0th-derivative) stencil and its offsets -m..m: returns c0, q0."""
    m = (window - 1) // 2
    return list(savgol_coeffs(window, d, deriv=0, delta=1.0, use='dot')), list(range(-m, m + 1))


def _G(n, k):
    """Integrated-Wiener covariance kernel, Method.pdf Eq. (22)."""
    q, p = min(n, k), max(n, k)
    return q * q * (3 * p - q) / 6.0


def _K(c1, q1, c2, q2):
    shift = -min(list(q1) + list(q2))          # offsets measured from the earliest time
    return sum(a * b * _G(k + shift, l + shift) for a, k in zip(c1, q1) for b, l in zip(c2, q2))


def Kav(ca, qa, cv, qv):
    """K_av, Method.pdf Eq. (29): 229/756 for S-G (9, 3, S=1), 1/6 for naive."""
    return _K(ca, qa, cv, qv)


def Kaa(ca, qa):
    """K_aa, Method.pdf Eq. (13): acceleration-noise variance factor."""
    return _K(ca, qa, ca, qa)


def phi_factor(ca, qa, cv, qv):
    """Multiplicative drag correction Phi = -(1/6) sum_{k,l} c^a_k c^v_l |q^a_k - q^v_l|^3
    (Method.pdf App. A 4-5): 2/3 for naive differences, 149/378 for S-G (9, 3, S=1)."""
    phi = -sum(a * v * abs(k - l) ** 3 for a, k in zip(ca, qa) for v, l in zip(cv, qv)) / 6.0
    if abs(phi) < 1e-9:
        raise ValueError("Phi = 0 for these stencils (degenerate; for S-G use S >= 1).")
    return float(phi)


def phi_sg(window, d, S):
    """Phi_SG for S-G derivatives (window, degree d, velocity shift S)."""
    return phi_factor(*sg_stencils(window, d, S))
