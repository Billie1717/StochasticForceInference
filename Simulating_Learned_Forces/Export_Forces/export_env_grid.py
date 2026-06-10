"""
Export the trained ForceGNN environmental MLP F_env(x, y) on a 2D grid.

Conventions matched to Fetch_ForceFields.ipynb (get_force_environment):
    x_norm = (x - x_mean[0]) / x_std
    y_norm = (y - x_mean[1]) / x_std
    F_phys = y_std * F_env_net([x_norm, y_norm])

The grid is saved as a .npz with:
    x_centers : (Nx,) float64    -- physical x coordinates of grid columns
    y_centers : (Ny,) float64    -- physical y coordinates of grid rows
    Fx        : (Nx, Ny) float64 -- x-component of F_env at (x_centers[i], y_centers[j])
    Fy        : (Nx, Ny) float64 -- y-component of F_env at (x_centers[i], y_centers[j])

Indexing is [i, j] = [x_index, y_index] (same 'ij' meshgrid convention as the
notebook). This matches what `np.meshgrid(x, y, indexing='ij')` produces.

A runtime helper (`bilinear_interp_field`) is included so the LAMMPS callback
module can `from export_env_grid import bilinear_interp_field` without having
to redeclare it.
"""

from __future__ import annotations

import json
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
# Re-declare ForceGNN (same as export_pair_table.py, kept self-contained)
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


def load_trained_model(model_stem: str = MODEL_NAME):
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
# F_env evaluation (must match get_force_environment in the notebook)
# ---------------------------------------------------------------------------
def f_env(model: ForceGNN, norm: dict, xy: np.ndarray) -> np.ndarray:
    """
    Evaluate physical environmental force F_env(x,y) for a (M, 2) array of
    positions. Returns (M, 2) array of [Fx, Fy].
    """
    x_mean = np.asarray(norm.get("x_mean", [0.0, 0.0]), dtype=np.float64)
    x_std = float(norm.get("x_std", 1.0)) or 1.0
    y_std_raw = norm.get("y_std", 1.0)
    y_std = (
        float(np.asarray(y_std_raw).flat[0])
        if np.ndim(y_std_raw) != 0
        else float(y_std_raw)
    )
    if y_std == 0.0:
        y_std = 1.0

    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy_norm = (xy - x_mean[None, :]) / x_std

    with torch.no_grad():
        inp = torch.tensor(xy_norm, dtype=torch.float32)
        F_norm = model.env_net(inp).cpu().numpy()
    return y_std * F_norm.astype(np.float64)


# ---------------------------------------------------------------------------
# Bilinear interpolation helper for the runtime callback
# ---------------------------------------------------------------------------
def bilinear_interp_field(
    x: np.ndarray,
    y: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    Fx: np.ndarray,
    Fy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear interpolation of (Fx, Fy) at arbitrary points (x, y).

    The grid is assumed to be uniformly spaced. Out-of-range points are
    clamped to the grid edge (so a particle that wandered outside still
    feels *some* force; we could also return 0 there if preferred).

    Parameters
    ----------
    x, y : (M,) arrays of query positions
    x_centers, y_centers : (Nx,) and (Ny,) sorted, uniform grid coordinates
    Fx, Fy : (Nx, Ny) field components

    Returns
    -------
    Fx_q, Fy_q : (M,) arrays
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    dx = x_centers[1] - x_centers[0]
    dy = y_centers[1] - y_centers[0]
    Nx = x_centers.size
    Ny = y_centers.size

    # fractional index into grid
    fi = (x - x_centers[0]) / dx
    fj = (y - y_centers[0]) / dy

    # clamp to interior so i+1, j+1 stay in range
    fi = np.clip(fi, 0.0, Nx - 1 - 1e-12)
    fj = np.clip(fj, 0.0, Ny - 1 - 1e-12)

    i0 = np.floor(fi).astype(np.int64)
    j0 = np.floor(fj).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1

    tx = fi - i0
    ty = fj - j0

    # bilinear weights
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty

    def interp(F):
        return (
            w00 * F[i0, j0]
            + w10 * F[i1, j0]
            + w01 * F[i0, j1]
            + w11 * F[i1, j1]
        )

    return interp(Fx), interp(Fy)


# ---------------------------------------------------------------------------
# Main: build and save the grid
# ---------------------------------------------------------------------------
def main(
    x_range: tuple[float, float] = (-20.0, 20.0),
    y_range: tuple[float, float] = (-20.0, 20.0),
    nx: int = 201,
    ny: int = 201,
    out_name: str = "env_grid_UD_Playing_3Tanh.npz",
) -> Path:
    model, norm = load_trained_model()

    x_centers = np.linspace(x_range[0], x_range[1], nx)
    y_centers = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")  # (nx, ny)

    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    F = f_env(model, norm, pts)  # (nx*ny, 2)
    Fx = F[:, 0].reshape(nx, ny)
    Fy = F[:, 1].reshape(nx, ny)

    out_path = EXPORT_DIR / out_name
    np.savez(
        out_path,
        x_centers=x_centers,
        y_centers=y_centers,
        Fx=Fx,
        Fy=Fy,
        # provenance / convention reminder
        meta=np.array(
            [
                f"model={MODEL_NAME}",
                f"x_mean={list(norm.get('x_mean', [0.0, 0.0]))}",
                f"x_std={float(norm.get('x_std', 1.0))}",
                f"y_std={float(norm.get('y_std', 1.0))}",
                f"x_range={list(x_range)}, y_range={list(y_range)}, nx={nx}, ny={ny}",
                "indexing='ij': Fx[i,j] at (x_centers[i], y_centers[j])",
            ],
            dtype=object,
        ),
    )

    Fmag = np.sqrt(Fx ** 2 + Fy ** 2)
    print(f"Wrote {out_path}")
    print(f"  x in [{x_range[0]}, {x_range[1]}]  nx={nx}  dx={x_centers[1]-x_centers[0]:.4g}")
    print(f"  y in [{y_range[0]}, {y_range[1]}]  ny={ny}  dy={y_centers[1]-y_centers[0]:.4g}")
    print(f"  |F| min={Fmag.min():.4g}  max={Fmag.max():.4g}  mean={Fmag.mean():.4g}")
    print(f"  Fx range=[{Fx.min():.4g}, {Fx.max():.4g}]")
    print(f"  Fy range=[{Fy.min():.4g}, {Fy.max():.4g}]")
    return out_path


if __name__ == "__main__":
    main()
