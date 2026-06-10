"""
Export the trained ForceGNN pair-interaction MLP F_pair(r) to a
LAMMPS `pair_style table` file.

Conventions matched to Fetch_ForceFields.ipynb:
    r_norm = r / r_std           (NO mean subtraction, matches notebook)
    F_phys  = y_std * F_net(r_norm)

Output table format (LAMMPS "pair_style table" linear):
    # comment line(s)
    KEYWORD
    N <Npts> R <rmin> <rmax>
    <blank line>
    1   r_1   U(r_1)   F(r_1)
    2   r_2   U(r_2)   F(r_2)
    ...

U(r) is obtained by integrating  U(r) = U(r_max) + integral_r^{rmax} F(r') dr'
(taking U(r_max) = 0 so the potential vanishes at the cutoff).

Sign convention: LAMMPS expects F(r) = -dU/dr to be the radial force on
particle i due to particle j, positive = repulsive (pushes i away from j).
The trained network was fitted with the same convention used by the
Underdamped LAMMPS sim it learned from, so we pass F through unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "Models"  # .../MakeSimsFromForcefields/Models
EXPORT_DIR = HERE
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "model_UD_Playing_ReDo_2D_3Tanh"


# ---------------------------------------------------------------------------
# Re-declare ForceGNN here so we don't depend on the notebook being importable.
# Architecture must match Models/Fetch_ForceFields.ipynb exactly.
# ---------------------------------------------------------------------------
class ForceGNN(nn.Module):
    def __init__(self, hidden_dim: int = 32, gamma_init: float = 0.0):
        super().__init__()
        self.env_net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.interaction_net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))


def load_trained_model(model_stem: str = MODEL_NAME) -> tuple[ForceGNN, dict]:
    """Load weights + config + normalisation stats."""
    base = MODELS_DIR / model_stem
    with open(base.with_name(base.name + "_config.json")) as f:
        config = json.load(f)
    with open(base.with_name(base.name + "_norm.json")) as f:
        norm = json.load(f)

    model = ForceGNN(hidden_dim=int(config["hidden_dim"]))
    state = torch.load(str(base) + ".pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, norm


# ---------------------------------------------------------------------------
# F_pair evaluation (must match get_force_at_distance in the notebook)
# ---------------------------------------------------------------------------
def f_pair(model: ForceGNN, norm: dict, r: np.ndarray) -> np.ndarray:
    """Evaluate physical pair force F(r) for a 1D array of distances r."""
    r_std = float(norm.get("r_std", 1.0)) or 1.0
    y_std_raw = norm.get("y_std", 1.0)
    y_std = (
        float(np.asarray(y_std_raw).flat[0])
        if np.ndim(y_std_raw) != 0
        else float(y_std_raw)
    )
    if y_std == 0.0:
        y_std = 1.0

    r = np.asarray(r, dtype=np.float64)
    r_norm = r / r_std  # match notebook (no mean subtraction)
    with torch.no_grad():
        inp = torch.tensor(r_norm.reshape(-1, 1), dtype=torch.float32)
        f_norm = model.interaction_net(inp).cpu().numpy().ravel()
    return y_std * f_norm.astype(np.float64)


# ---------------------------------------------------------------------------
# Table writer
# ---------------------------------------------------------------------------
def write_lammps_pair_table(
    out_path: Path,
    keyword: str,
    r: np.ndarray,
    U: np.ndarray,
    F: np.ndarray,
    comment_lines: list[str] | None = None,
) -> None:
    """
    Write a LAMMPS pair_style table file (linear-spaced r, format 'R').

    Parameters
    ----------
    out_path : where to save
    keyword  : the section keyword used by `pair_coeff * * <file> <KEYWORD>`
    r, U, F  : 1D arrays of equal length; r must be monotonically increasing
    """
    r = np.asarray(r, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    assert r.ndim == 1 and r.shape == U.shape == F.shape
    N = r.size

    # sanity: linear spacing required by 'R' header
    dr = np.diff(r)
    assert np.allclose(dr, dr[0], rtol=1e-6), "r must be uniformly spaced for 'R'"

    lines: list[str] = []
    if comment_lines:
        for c in comment_lines:
            lines.append(f"# {c}")
    else:
        lines.append(f"# LAMMPS pair_style table generated from ForceGNN")
    lines.append("")  # blank line is fine between header comments
    lines.append(keyword)
    lines.append(f"N {N} R {r[0]:.10g} {r[-1]:.10g}")
    lines.append("")
    for i in range(N):
        lines.append(f"{i+1} {r[i]:.10g} {U[i]:.10g} {F[i]:.10g}")
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main: build U(r) by trapezoidal integration of F(r)
# ---------------------------------------------------------------------------
def main(
    rmin: float = 0.05,
    rmax: float = 3.0,
    n_points: int = 2000,
    keyword: str = "FORCE_GNN_PAIR",
    out_name: str = "pair_table_UD_Playing_3Tanh.table",
) -> Path:
    model, norm = load_trained_model()

    r = np.linspace(rmin, rmax, n_points)
    F = f_pair(model, norm, r)

    # U(r) = integral_r^{rmax} F(r') dr', i.e. U(rmax) = 0 by construction.
    # cumulative_trapezoid from the right:
    U = np.zeros_like(r)
    # integrate F over [r_i, r_max] using trapezoid
    # walking from the right end inward
    for i in range(n_points - 2, -1, -1):
        U[i] = U[i + 1] + 0.5 * (F[i] + F[i + 1]) * (r[i + 1] - r[i])

    out_path = EXPORT_DIR / out_name
    comments = [
        f"Source model: {MODEL_NAME}.pt",
        f"hidden_dim=64, interaction MLP only",
        f"Normalisation: r_norm = r / r_std (no mean sub, matches notebook)",
        f"r_std={float(norm['r_std']):.6g}, y_std={float(norm['y_std']):.6g}",
        f"r in [{rmin}, {rmax}], N={n_points} (linear)",
        f"U(rmax)=0 by construction (cutoff = {rmax})",
    ]
    write_lammps_pair_table(out_path, keyword, r, U, F, comment_lines=comments)
    print(f"Wrote {out_path}")
    print(f"  keyword: {keyword}")
    print(f"  r range: [{rmin}, {rmax}]   N = {n_points}")
    print(f"  F(rmin) = {F[0]:.6g}   F(rmax) = {F[-1]:.6g}")
    print(f"  U(rmin) = {U[0]:.6g}   U(rmax) = {U[-1]:.6g}")

    # also save the raw arrays as .npz for easy plotting/verification
    npz_path = out_path.with_suffix(".npz")
    np.savez(npz_path, r=r, U=U, F=F)
    print(f"  also saved arrays to {npz_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stitched table: LJ/soft core spliced onto the NN for r < stitch_r
# ---------------------------------------------------------------------------
def lj_soft_F(r: np.ndarray,
               epsilon: float = 0.5,
               sigma: float = 1.5,
               lam: float = 0.1,
               alpha: float = 0.5,
               n: int = 1) -> np.ndarray:
    """
    LAMMPS lj/cut/soft force F(r) = -dU/dr (positive = repulsive).
    D = alpha*(1-lam)^2 + (r/sigma)^6
    U = 4*eps*lam^n * (1/D^2 - 1/D)
    F = -dU/dr = 4*eps*lam^n * (6r^5/sigma^6) * (2/D^3 - 1/D^2)
    """
    r = np.asarray(r, dtype=np.float64)
    D = alpha * (1.0 - lam) ** 2 + (r / sigma) ** 6
    dD_dr = 6.0 * r ** 5 / sigma ** 6
    return 4.0 * epsilon * lam ** n * dD_dr * (2.0 / D ** 3 - 1.0 / D ** 2)


def lj_soft_U(r: np.ndarray,
               epsilon: float = 0.5,
               sigma: float = 1.5,
               lam: float = 0.1,
               alpha: float = 0.5,
               n: int = 1,
               rcut: float = 3.0) -> np.ndarray:
    """LJ/cut/soft potential shifted to zero at rcut."""
    r = np.asarray(r, dtype=np.float64)
    def _u(rr):
        D = alpha * (1.0 - lam) ** 2 + (rr / sigma) ** 6
        return 4.0 * epsilon * lam ** n * (1.0 / D ** 2 - 1.0 / D)
    return _u(r) - _u(rcut)


def write_stitched_pair_table(
    src_table: str | Path,
    out_path: str | Path,
    stitch_r: float = 0.7,
    blend_width: float = 0.15,
    lj_kwargs: dict | None = None,
    keyword: str = "FORCE_GNN_PAIR_STITCHED",
) -> Path:
    """
    Read the existing NN pair table and write a new one where:
      - r < stitch_r            : pure LJ/soft (analytic)
      - stitch_r <= r < stitch_r + blend_width : cosine blend LJ -> NN
      - r >= stitch_r + blend_width : pure NN

    This fixes the unphysical near-zero repulsive core in the NN table
    without touching the region the NN was trained on.

    Parameters
    ----------
    src_table   : path to the original NN .table file
    out_path    : where to write the stitched table
    stitch_r    : transition starts here (pure LJ below, blend above)
    blend_width : width of the cosine crossover region
    lj_kwargs   : dict of kwargs forwarded to lj_soft_F / lj_soft_U
                  (epsilon, sigma, lam, alpha, n, rcut)
    keyword     : LAMMPS section keyword for the output table
    """
    src_table = Path(src_table)
    out_path = Path(out_path)
    lj_kw = {"epsilon": 0.5, "sigma": 1.5, "lam": 0.1,
              "alpha": 0.5, "n": 1, "rcut": 3.0}
    if lj_kwargs:
        lj_kw.update(lj_kwargs)

    # --- load NN table ---
    r_nn, U_nn, F_nn = [], [], []
    with open(src_table) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lstrip("-").isdigit():
                r_nn.append(float(parts[1]))
                U_nn.append(float(parts[2]))
                F_nn.append(float(parts[3]))
    r = np.array(r_nn)
    U_nn = np.array(U_nn)
    F_nn = np.array(F_nn)

    # --- analytic LJ on same grid ---
    F_lj = lj_soft_F(r, **{k: v for k, v in lj_kw.items() if k != "rcut"})
    U_lj = lj_soft_U(r, **lj_kw)

    # --- cosine blend weight (1 = pure NN, 0 = pure LJ) ---
    w = np.ones_like(r)
    lo, hi = stitch_r, stitch_r + blend_width
    in_blend = (r >= lo) & (r < hi)
    below = r < lo
    t = (r[in_blend] - lo) / blend_width          # 0 at lo, 1 at hi
    w[in_blend] = 0.5 * (1.0 - np.cos(np.pi * t)) # 0 at lo, 1 at hi
    w[below] = 0.0

    F_out = w * F_nn + (1.0 - w) * F_lj
    U_out = w * U_nn + (1.0 - w) * U_lj

    # ensure U(rmax) = 0 (it already should be from both sides, but tidy up)
    U_out -= U_out[-1]

    comments = [
        f"Stitched table: LJ/soft core (r < {stitch_r}) blended onto NN (r > {stitch_r + blend_width:.3g})",
        f"Source NN table: {src_table.name}",
        f"LJ params: {lj_kw}",
        f"Blend: cosine over [{stitch_r}, {stitch_r + blend_width:.3g}]",
    ]
    write_lammps_pair_table(out_path, keyword, r, U_out, F_out, comment_lines=comments)
    print(f"[stitch] wrote {out_path}")
    print(f"  pure LJ for r < {stitch_r}, blend [{stitch_r}, {stitch_r+blend_width:.3g}], pure NN beyond")
    return out_path


if __name__ == "__main__":
    main()
