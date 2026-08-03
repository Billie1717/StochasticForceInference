"""
Pytorch Inference Module
=========================

This is part of an adapted version of Billie's force inference code meant to convert from 
tensorflow to pytorch. The code for the actual models and training steps are in the Jupyter
notebooks and overlapping functions are mainly present here. The plan is to move the desired
model code into this script, but at the moment, the actual pytorch code is not present here.
"""


import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sympy import symbols, lambdify, diff

from tqdm.notebook import tqdm
from typing import Iterable, List, Optional, Sequence, Tuple, Dict



class VoxelNeighborList:
    """Maintain a voxel-based neighbour list for sequential frames.

    The neighbour list is rebuilt using a voxel (cell) decomposition of the
    spatial domain every ``rebuild_interval`` frames. Between rebuilds the
    cached neighbour pairs are reused, which significantly reduces the number
    of distance calculations when particle counts are large.

    Parameters
    ----------
    cutoff : float
        Cutoff radius for interactions (same as used in ``build_graph``).
    n_boxes : int, default 8
        Number of voxels per domain edge. The voxel size is set to
        ``L_box / n_boxes`` where ``L_box`` is the maximal domain extent.
    rebuild_interval : int, default 5
        Number of frames between full neighbour-list rebuilds.
    domain_bounds : tuple(float, float, float, float), optional
        Precomputed domain bounds ``(xmin, xmax, ymin, ymax)``. If ``None`` the
        bounds are inferred on the first call.
    domain_margin : float, default 0.0
        Extra padding added to each domain bound when (re)computing bounds.
    particle_column : str or None, default 'particle'
        Column used to detect changes in particle identities between frames.
        If ``None`` the neighbour list tracks only by row index.
    """

    def __init__(
        self,
        cutoff: float,
        n_boxes: int = 8,
        rebuild_interval: int = 5,
        domain_bounds: Optional[Tuple[float, float, float, float]] = None,
        domain_margin: float = 0.0,
        particle_column: Optional[str] = 'particle',
    ) -> None:
        if n_boxes < 1:
            raise ValueError("n_boxes must be a positive integer")

        self.cutoff = float(cutoff)
        self.n_boxes = int(n_boxes)
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.domain_bounds = domain_bounds
        self.domain_margin = float(domain_margin)
        self.particle_column = particle_column

        self._cached_pairs: List[Tuple[int, int]] = []
        self._frames_since_rebuild: Optional[int] = None
        self._last_particle_ids: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_neighbor_pairs(self, frame_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Return candidate neighbour pairs for the given frame.

        Depending on the ``rebuild_interval`` this either reuses the cached
        pairs or rebuilds the voxel list and updates the cache.
        """

        if self._needs_rebuild(frame_data):
            self._rebuild(frame_data)
            self._frames_since_rebuild = 0
        else:
            self._frames_since_rebuild = (self._frames_since_rebuild or 0) + 1

        return self._cached_pairs

    def force_rebuild(self, frame_data: pd.DataFrame) -> None:
        """Explicitly rebuild the neighbour list for ``frame_data``."""

        self._rebuild(frame_data)
        self._frames_since_rebuild = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _needs_rebuild(self, frame_data: pd.DataFrame) -> bool:
        if self._cached_pairs is None or self._frames_since_rebuild is None:
            return True

        if self._frames_since_rebuild >= self.rebuild_interval:
            return True

        current_ids = self._extract_particle_ids(frame_data)
        if self._last_particle_ids != current_ids:
            return True

        if self.domain_bounds is None:
            return True

        xmin, xmax, ymin, ymax = self.domain_bounds
        x_vals = frame_data['x']
        y_vals = frame_data['y']

        if (
            x_vals.min() < xmin + self.domain_margin
            or x_vals.max() > xmax - self.domain_margin
            or y_vals.min() < ymin + self.domain_margin
            or y_vals.max() > ymax - self.domain_margin
        ):
            # If particles approach the boundaries we expand the domain and rebuild.
            return True

        return False

    def _rebuild(self, frame_data: pd.DataFrame) -> None:
        self._ensure_domain_bounds(frame_data)
        self._cached_pairs = self._build_pairs(frame_data)
        self._last_particle_ids = self._extract_particle_ids(frame_data)

    def _extract_particle_ids(self, frame_data: pd.DataFrame) -> Tuple:
        if self.particle_column and self.particle_column in frame_data.columns:
            return tuple(frame_data[self.particle_column].tolist())
        return tuple(range(len(frame_data)))

    def _ensure_domain_bounds(self, frame_data: pd.DataFrame) -> None:
        if self.domain_bounds is None:
            x_vals = frame_data['x']
            y_vals = frame_data['y']
            self.domain_bounds = (
                float(x_vals.min() - self.domain_margin),
                float(x_vals.max() + self.domain_margin),
                float(y_vals.min() - self.domain_margin),
                float(y_vals.max() + self.domain_margin),
            )
            return

        # Expand bounds if necessary (do not shrink).
        xmin, xmax, ymin, ymax = self.domain_bounds
        x_vals = frame_data['x']
        y_vals = frame_data['y']
        self.domain_bounds = (
            float(min(xmin, x_vals.min() - self.domain_margin)),
            float(max(xmax, x_vals.max() + self.domain_margin)),
            float(min(ymin, y_vals.min() - self.domain_margin)),
            float(max(ymax, y_vals.max() + self.domain_margin)),
        )

    def _build_pairs(self, frame_data: pd.DataFrame) -> List[Tuple[int, int]]:
        if self.domain_bounds is None:
            raise RuntimeError("Domain bounds must be set before building neighbour list")

        xmin, xmax, ymin, ymax = self.domain_bounds
        lx = max(xmax - xmin, 1e-8)
        ly = max(ymax - ymin, 1e-8)
        l_box = max(lx, ly)
        cell_size = l_box / self.n_boxes
        if cell_size <= 0:
            cell_size = max(self.cutoff, 1e-3)

        cells: defaultdict = defaultdict(list)
        positions = frame_data[['x', 'y']].to_numpy()

        for idx, (x_pos, y_pos) in enumerate(positions):
            cx = int(np.clip(np.floor((x_pos - xmin) / cell_size), 0, self.n_boxes - 1))
            cy = int(np.clip(np.floor((y_pos - ymin) / cell_size), 0, self.n_boxes - 1))
            cells[(cx, cy)].append(idx)

        neighbor_pairs = set()
        offsets = (-1, 0, 1)

        for (cx, cy), members in cells.items():
            for dx in offsets:
                nx = cx + dx
                if nx < 0 or nx >= self.n_boxes:
                    continue
                for dy in offsets:
                    ny = cy + dy
                    if ny < 0 or ny >= self.n_boxes:
                        continue

                    neighbours = cells.get((nx, ny))
                    if not neighbours:
                        continue

                    for i in members:
                        for j in neighbours:
                            if i >= j:
                                continue
                            neighbor_pairs.add((i, j))

        return sorted(neighbor_pairs)






def load_data(datadir, dataname):
    filepath = Path(datadir) / dataname
    data = pd.read_csv(filepath)
    
    print(f"Data loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Number of frames: {len(data['frame'].unique())}")
    
    return data



t = symbols('t')

# Define the function w which is a sympy expression involving t
w_sym = (t**2-1)**2
def load_data(datadir, dataname):
    """
    Load particle trajectory data from CSV.
    
    """
    filepath = Path(datadir) / dataname
    data = pd.read_csv(filepath)
    
    print(f"Data loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Number of frames: {len(data['frame'].unique())}")
    
    return data


def compute_weak_derivatives(
    data: pd.DataFrame,
    dt: float,
    tau: int,
    w_sym,                         # sympy expression in symbol tbar over [-1,1]
    particle_col='particle',
    frame_col='frame',
    x_col='x',
    y_col='y',
    vx_col='Wvx',
    vy_col='Wvy',
    ax_col='Wax',
    ay_col='Way',
    inplace=True,
):
    """
    Compute weak-form velocity and acceleration for each particle trajectory.

    Parameters
    ----------
    data : DataFrame
        Must contain columns [particle_col, frame_col, x_col, y_col].
        Frames are assumed uniformly spaced by `dt` within each particle.
    dt : float
        Time step between consecutive frames.
    tau : int
        Even integer. Number of *subintervals* in [-1,1] and also number of
        time steps spanned by the window. The window uses tau+1 samples.
        For center index i, we sample frames i - tau//2 ... i + tau//2.
    w_sym : sympy expression
        Test function w(tbar) with tbar symbol 't'. Should satisfy
        w(±1)=0 and w'(±1)=0 to eliminate boundary terms.

    Notes
    -----
    For velocity:
        v ≈ - (∫ x(t̄) w'(t̄) d t̄) / (∫ w(t̄) d t̄)
    For acceleration:
        a ≈ (4 / T^2) * (∫ x(t̄) w''(t̄) d t̄) / (∫ w(t̄) d t̄),  T = tau*dt

    Returns
    -------
    DataFrame (if inplace=False), otherwise modifies `data` in place.
    """
    if not inplace:
        data = data.copy()

    # Prepare columns - reset to 0.0 for all rows
    # Only valid centers (from half to n-half) will be updated with computed values
    # Boundary points will remain at 0.0
    for c in [vx_col, vy_col, ax_col, ay_col]:
        data[c] = 0.0

    print("In Compute Weak Derivatives: ",data.head())
    # Sympy -> numpy callables
    tbar = symbols('t')
    w = lambdify(tbar, w_sym, 'numpy')
    wp = lambdify(tbar, diff(w_sym, tbar), 'numpy')
    wpp = lambdify(tbar, diff(w_sym, (tbar, 2)), 'numpy')

    # Simpson grid on [-1,1]
    if tau % 2 != 0 or tau < 2:
        raise ValueError("tau must be an even integer ≥ 2.")
    tbars = np.linspace(-1.0, 1.0, tau + 1)             # length tau+1
    delta = 2.0 / tau                                   # Δt̄
    # Simpson weights include the Δ factor: (Δ/3) * [1, 4, 2, 4, ..., 4, 1]
    pattern = np.array([1] + [4, 2] * (tau // 2 - 1) + [4] + [1]) if tau >= 4 else np.array([1, 4, 1])
    simpson_weights = (delta / 3.0) * pattern

    # Precompute w, w', w'' and the normalizer ∫ w
    w_vals = w(tbars)
    wp_vals = wp(tbars)
    wpp_vals = wpp(tbars)

    Iw = np.sum(w_vals * simpson_weights)               # ∫ w d t̄  (dimensionless)
    if np.isclose(Iw, 0.0):
        raise ZeroDivisionError("∫ w(t̄) d t̄ is ~0; choose a different test function.")

    T = tau * dt                                        # window duration

    # Track statistics for debugging
    total_points = 0
    total_computed = 0
    total_skipped_particles = 0

    # Work per particle
    for pid, dfp in data.sort_values([particle_col, frame_col]).groupby(particle_col, sort=False):
        idx = dfp.index.to_numpy()
        n = len(idx)
        half = tau // 2
        total_points += n
        
        if n < tau + 1:
            total_skipped_particles += 1
            continue  # not enough points for any center

        # We'll slide a centered window; centers from half .. n-1-half
        centers = np.arange(half, n - half, dtype=int)
        total_computed += len(centers)

        # Extract x,y arrays in numeric form for speed
        x_arr = dfp[x_col].to_numpy()
        y_arr = dfp[y_col].to_numpy()

        # For each center, compute integrals using the precomputed t̄ weights
        for c in centers:
            win = slice(c - half, c + half + 1)         # length tau+1
            x_win = x_arr[win]
            y_win = y_arr[win]

            # Handle NaNs (skip if any in window)
            if (np.any(~np.isfinite(x_win)) or np.any(~np.isfinite(y_win))):
                continue

            # ∫ x w' d t̄, ∫ x w'' d t̄, and same for y
            Ix_wp   = np.sum(x_win * wp_vals  * simpson_weights)
            Ix_wpp  = np.sum(x_win * wpp_vals * simpson_weights)
            Iy_wp   = np.sum(y_win * wp_vals  * simpson_weights)
            Iy_wpp  = np.sum(y_win * wpp_vals * simpson_weights)

            # Weak velocity (boundary terms vanish for your polynomial w)
            scale_v = 2.0 / T
            vx = scale_v * (- Ix_wp / Iw)
            vy = scale_v * (- Iy_wp / Iw)
            # Weak acceleration: factor 4/T^2
            fac = 4.0 / (T * T)
            ax = fac * (Ix_wpp / Iw)
            ay = fac * (Iy_wpp / Iw)

            data.loc[idx[c], vx_col] = vx
            data.loc[idx[c], vy_col] = vy
            data.loc[idx[c], ax_col] = ax
            data.loc[idx[c], ay_col] = ay
    
    # Print statistics
    total_boundary = total_points - total_computed
    boundary_fraction = total_boundary / total_points if total_points > 0 else 0.0
    print(f"  Weak derivative statistics (tau={tau}):")
    print(f"    Total points: {total_points}")
    print(f"    Computed (valid centers): {total_computed}")
    print(f"    Boundary points (zeros): {total_boundary}")
    print(f"    Fraction boundary: {boundary_fraction:.4f}")
    print(f"    Skipped particles (n < tau+1): {total_skipped_particles}")
    print("Out of Compute Weak Derivatives: ",data.head())
    return data #if not inplace else None


def calculate_derivatives(data, dt=0.1,weak_form = False,naive_form = False,tau = 16):
    """
    Calculate velocities and accelerations from positions using finite differences.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with columns: frame, x, y, and optionally 'id'
    dt : float
        Time step between frames
    
    Returns:
    --------
    pd.DataFrame
        Data with added columns: vx, vy, ax, ay (boundary frames removed)
    """
    # Check if derivatives already exist
    if 'ax' in data.columns and 'ay' in data.columns:
        print("Derivatives already in data, skipping calculation...")
        frames = sorted(data['frame'].unique())
        data_clean = data[(data['frame'] > frames[0]) & (data['frame'] < frames[-1])].copy()
        return data_clean
    
    print("Calculating derivatives...")    
    # Initialize derivative columns
    data['vx'] = 0.0
    data['vy'] = 0.0
    data['ax'] = 0.0
    data['ay'] = 0.0
    
    # Get particle IDs
    if 'particle' in data.columns:
        particles = data['particle'].unique()
    else:
        print("With variable node data, particle ID must be in column of data")
    print(f"Processing {len(particles)} particles...")
    
    # Calculate derivatives for each particle using central differences
    
    if naive_form:
        print("Calculating naïve derivatives...")
        for pid in particles:
            particle_mask = data['particle'] == pid
            particle_data = data[particle_mask].sort_values('frame')
            indices = particle_data.index.tolist()
            
            # Central differences for interior points
            for i in range(1, len(indices) - 1):
                idx = indices[i]
                idx_prev = indices[i-1]
                idx_next = indices[i+1]
                
                # Velocity
                #data.loc[idx, 'vx'] = (data.loc[idx, 'x'] - data.loc[idx_prev, 'x']) / (dt)
                #data.loc[idx, 'vy'] = (data.loc[idx, 'y'] - data.loc[idx_prev, 'y']) / (dt)
                data.loc[idx, 'vx'] = (data.loc[idx, 'x'] - data.loc[idx_prev, 'x']) / (dt)
                data.loc[idx, 'vy'] = (data.loc[idx, 'y'] - data.loc[idx_prev, 'y']) / (dt)
                
                # Acceleration
                data.loc[idx, 'ax'] = (data.loc[idx_next, 'x'] - 2*data.loc[idx, 'x'] + data.loc[idx_prev, 'x']) / (dt**2)
                data.loc[idx, 'ay'] = (data.loc[idx_next, 'y'] - 2*data.loc[idx, 'y'] + data.loc[idx_prev, 'y']) / (dt**2)
    if weak_form and not naive_form:
        print("Calculating weak form derivatives...")
        compute_weak_derivatives(
            data=data,
            dt=dt,         # e.g. 0.1
            tau=tau,             # even integer; window spans tau*dt in time and uses tau+1 samples
            w_sym=w_sym,
            inplace=True,
            vx_col='vx',
            vy_col='vy',
            ax_col='ax',
            ay_col='ay',
        )
    if weak_form and naive_form:
        compute_weak_derivatives(
            data=data,
            dt=dt,         # e.g. 0.1
            tau=tau,             # even integer; window spans tau*dt in time and uses tau+1 samples
            w_sym=w_sym,
            inplace=True,
            vx_col='Wvx',
            vy_col='Wvy',
            ax_col='Wax',
            ay_col='Way',
        )
    data_clean = data.copy()
    print(f"Acceleration statistics:")
    print(f"  ax: mean={data_clean['ax'].mean():.3f}, std={data_clean['ax'].std():.3f}")
    print(f"  ay: mean={data_clean['ay'].mean():.3f}, std={data_clean['ay'].std():.3f}")
    return data_clean


def build_graph(frame_data, cutoff=3.0, neighbor_pairs: Optional[Sequence[Tuple[int, int]]] = None):
    """
    Build graph structure from particle data for one frame.
    
    Parameters:
    -----------
    frame_data : pd.DataFrame
        Single frame data with columns 'x', 'y', 'vx', 'vy' and optionally neighbour columns
        (e.g. 'Neighbor1', 'Neighbor2', ...).
        Optionally may contain 'mask' column (1 = include in training, 0 = neighbor only)
    cutoff : float
        Maximum distance for edge connections when neighbour columns are absent.
    
    neighbor_pairs : sequence of tuple(int, int), optional
        Candidate neighbour pairs (typically from ``VoxelNeighborList``). If
        provided, distance checks are only computed for these pairs instead of
        all O(N^2) combinations.

    Returns:
    --------
    tuple
        (nodes, num_neighbors, edges, edge_features, vel_edge_features, mask, nn_r)
        - nodes: (N, 4) array of [x, y, vx, vy]
        - num_neighbors: (N,) array of number of neighbors per node
        - edges: (2, E) array of edge indices [source, target]
        - edge_features: (E, 3) array of [distance, dx/r, dy/r]
        - vel_edge_features: (E, 3) array of [|dv|, dvx/|dv|, dvy/|dv|]
        - mask: (N,) array of mask values (1 = train, 0 = neighbor only), defaults to all 1s if not provided
        - nn_r: (N,) array of nearest neighbor distance per node (np.nan for nodes with no neighbors)
    """
    N = len(frame_data)
    nodes = frame_data[['x', 'y', 'vx', 'vy']].values
    edges = []
    edge_features = []
    vel_edge_features = []
    # Initialize nearest neighbor distance per node
    nn_r_per_node = np.full(N, np.inf, dtype=np.float32)

    # Extract mask if present, otherwise default to all 1s (all nodes included in training)
    if 'mask' in frame_data.columns:
        mask = frame_data['mask'].values.astype(np.float32)
    else:
        mask = np.ones(N, dtype=np.float32)
    # now make mask a zero if ax, ay, vx, vy are zero
    mask = mask & (frame_data['ax'] != 0) & (frame_data['ay'] != 0) & (frame_data['vx'] != 0) & (frame_data['vy'] != 0)
    num_neighbors_arr = np.zeros(N, dtype=np.int32)

    positions  = frame_data[['x', 'y']].to_numpy()
    velocities = frame_data[['vx', 'vy']].to_numpy()
    neighbor_cols = [col for col in frame_data.columns if col.lower().startswith('neighbor')]
    use_neighbor_columns = len(neighbor_cols) > 0

    if use_neighbor_columns:
        if 'particle' not in frame_data.columns:
            raise ValueError(
                "Neighbor columns detected but 'particle' column is missing; "
                "unable to map neighbour IDs to row indices."
            )
        particle_ids = frame_data['particle'].to_numpy()
        id_to_index = {int(pid): idx for idx, pid in enumerate(particle_ids)}

        for row_idx, row in frame_data.reset_index(drop=True).iterrows():
            source_pid = int(row['particle'])
            NumNeighbors = 0
            for col in neighbor_cols:
                neighbor_val = row[col]

                if pd.isna(neighbor_val):
                    continue
                try:
                    neighbor_pid = int(neighbor_val)
                except (TypeError, ValueError):
                    continue
                if neighbor_pid == source_pid:
                    continue
                target_idx = id_to_index.get(neighbor_pid)
                if target_idx is None:
                    continue

                dx, dy = positions[target_idx] - positions[row_idx]
                r = np.sqrt(dx**2 + dy**2)
                if r == 0:
                    continue
                NumNeighbors = NumNeighbors + 1
                num_neighbors_arr[row_idx] = NumNeighbors
                unit_dx = dx / r
                unit_dy = dy / r
                edges.append([row_idx, target_idx])
                edge_features.append([r, unit_dx, unit_dy])
                dvx = velocities[target_idx, 0] - velocities[row_idx, 0]
                dvy = velocities[target_idx, 1] - velocities[row_idx, 1]
                dv = np.sqrt(dvx**2 + dvy**2)
                if dv > 0:
                    vel_edge_features.append([dv, dvx / dv, dvy / dv])
                else:
                    vel_edge_features.append([0.0, 0.0, 0.0])
                # Update nearest neighbor distance
                if r < nn_r_per_node[row_idx]:
                    nn_r_per_node[row_idx] = r
    else:
        if neighbor_pairs is None:
            pair_iter: Iterable[Tuple[int, int]] = (
                (i, j)
                for i in range(N)
                for j in range(i + 1, N)
            )
        else:
            pair_iter = neighbor_pairs

        for pair in pair_iter:
            i, j = pair
            if i == j:
                continue

            xi, yi = positions[i]
            xj, yj = positions[j]
            dx, dy = xj - xi, yj - yi
            r = np.sqrt(dx**2 + dy**2)

            if r >= cutoff or r == 0:
                continue

            unit_dx = dx / r
            unit_dy = dy / r

            dvx = velocities[j, 0] - velocities[i, 0]
            dvy = velocities[j, 1] - velocities[i, 1]
            dv  = np.sqrt(dvx**2 + dvy**2)
            if dv > 0:
                vel_ij = [dv,  dvx / dv,  dvy / dv]
                vel_ji = [dv, -dvx / dv, -dvy / dv]
            else:
                vel_ij = [0.0, 0.0, 0.0]
                vel_ji = [0.0, 0.0, 0.0]

            # Directed edge i -> j
            edges.append([i, j])
            edge_features.append([r, unit_dx, unit_dy])
            vel_edge_features.append(vel_ij)
            # Update nearest neighbor distance
            if r < nn_r_per_node[i]:
                nn_r_per_node[i] = r
            # Count neighbor for node i (use numpy array for speed)
            num_neighbors_arr[i] += 1

            # Directed edge j -> i (negative direction)
            edges.append([j, i])
            edge_features.append([r, -unit_dx, -unit_dy])
            vel_edge_features.append(vel_ji)
            # Update nearest neighbor distance
            if r < nn_r_per_node[j]:
                nn_r_per_node[j] = r
            # Count neighbor for node j (use numpy array for speed)
            num_neighbors_arr[j] += 1
    x_min = frame_data['x'].min()
    x_max = frame_data['x'].max()
    y_min = frame_data['y'].min()
    y_max = frame_data['y'].max()
    #mask cells which are 20 away from the edge of the domain, we assume these may be governed by forces we can't observe
    mask = mask & (frame_data['x'] > x_min + 20) & (frame_data['x'] < x_max - 20) & (frame_data['y'] > y_min + 20) & (frame_data['y'] < y_max - 20)

    # Convert infinity to NaN for nodes with no neighbors
    nn_r_per_node[nn_r_per_node == np.inf] = np.nan
    
    return (
        nodes.astype(np.float32),
        num_neighbors_arr.astype(np.float32),
        np.array(edges, dtype=np.int32).T if edges else np.zeros((2, 0), dtype=np.int32),
        np.array(edge_features,     dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32),
        np.array(vel_edge_features, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32),
        mask,
        nn_r_per_node
    )


def prepare_dataset(
    data_clean,
    cutoff=3.0,
    max_frames=None,
    use_neighbor_list: bool = True,
    n_boxes: int = 8,
    neighbor_rebuild_interval: int = 5,
    neighbor_domain_margin: float = 0.0,
):
    """
    Prepare training dataset from all frames.
    
    Parameters:
    -----------
    data_clean : pd.DataFrame
        Cleaned data with derivatives (must have columns: x, y, vx, vy, ax, ay)
        Optionally may contain 'mask' column (1 = include in training, 0 = neighbor only)
    cutoff : float
        Graph connectivity cutoff distance
    max_frames : int, optional
        Maximum number of frames to use (None = use all)
    use_neighbor_list : bool, default True
        If True, reuse a voxel-based neighbour list between frames to avoid
        O(N^2) distance checks.
    n_boxes : int, default 8
        Number of voxels per domain edge used for the neighbour list.
    neighbor_rebuild_interval : int, default 5
        Rebuild the neighbour list after this many frames.
    neighbor_domain_margin : float, default 0.0
        Padding added to the inferred domain bounds when constructing voxels.
    
    Returns:
    --------
    list of dict
        Each dict contains: frame, nodes (N,4) [x,y,vx,vy], edges, edge_feat, targets, mask
        Optionally may contain 'mask' column (1 = include in training, 0 = neighbor only)
    """
    frames = sorted(data_clean['frame'].unique())
    if max_frames:
        frames = frames[:max_frames]
    
    dataset = []
    print(f"Preparing dataset from {len(frames)} frames...")
    
    neighbor_manager: Optional[VoxelNeighborList] = None
    if use_neighbor_list:
        if not {'x', 'y'}.issubset(data_clean.columns):
            raise ValueError("Data must contain 'x' and 'y' columns for neighbour list construction.")

        domain_bounds = (
            float(data_clean['x'].min() - neighbor_domain_margin),
            float(data_clean['x'].max() + neighbor_domain_margin),
            float(data_clean['y'].min() - neighbor_domain_margin),
            float(data_clean['y'].max() + neighbor_domain_margin),
        )

        particle_column = 'particle' if 'particle' in data_clean.columns else None

        neighbor_manager = VoxelNeighborList(
            cutoff=cutoff,
            n_boxes=n_boxes,
            rebuild_interval=neighbor_rebuild_interval,
            domain_bounds=domain_bounds,
            domain_margin=neighbor_domain_margin,
            particle_column=particle_column,
        )

    for frame_id in frames:
        print("processing frame ", frame_id)
        frame_data = data_clean[data_clean['frame'] == frame_id].reset_index(drop=True)
        
        neighbor_pairs = None
        if neighbor_manager is not None:
            neighbor_pairs = neighbor_manager.get_neighbor_pairs(frame_data)

        # Build graph (now returns mask and nn_r as well)
        nodes, num_neighbors, edges, edge_feat, vel_edge_feat, mask, nn_r = build_graph(frame_data, cutoff, neighbor_pairs=neighbor_pairs)
        
        # Targets (accelerations)
        targets = frame_data[['ax', 'ay']].values.astype(np.float32)
        

        dataset.append({
            'frame': frame_id,
            'nodes': nodes,
            'num_neighbors': num_neighbors,
            'edges': edges,
            'edge_feat': edge_feat,
            'vel_edge_feat': vel_edge_feat,
            'targets': targets,
            'mask': mask,
            'particle_ids': frame_data['particle'].to_numpy(dtype=np.int32) if 'particle' in frame_data.columns else np.arange(len(frame_data)),
            'nn_r': nn_r
        })
    # now begin normalisation of the data
    all_pairwise_distances = []
    all_x_vals = []
    all_y_vals = []
    all_targets = []
    for d in dataset:
        edge_data = d['edge_feat']
        edge_distances = edge_data[:,0]
        all_pairwise_distances.append(edge_distances)
        all_x_vals.append(d['nodes'][:,0])
        all_y_vals.append(d['nodes'][:,1])
        all_targets.append(d['targets'])
    #print(np.shape(np.concatenate(all_pairwise_distances)))
    print("Max pairwise distance: ",np.max(np.concatenate(all_pairwise_distances)))
    all_r = np.concatenate(all_pairwise_distances) if len(all_pairwise_distances) else np.array([0.0], dtype=np.float32)
    all_targets = np.vstack(all_targets) if len(all_targets) else np.zeros((1,2), dtype=np.float32)
    x_mean = np.array([float(np.mean(np.concatenate(all_x_vals))),
                   float(np.mean(np.concatenate(all_y_vals)))], dtype=np.float32)

    xx_std  = np.array([float(np.std(np.concatenate(all_x_vals))),
                    float(np.std(np.concatenate(all_y_vals)))], dtype=np.float32)

    xx_std[xx_std < 1e-12] = 1.0
    x_std = np.sqrt(xx_std[0]**2 + xx_std[1]**2)
    if x_std == 0.0:
        x_std = 1.0

    print("x_mean:", x_mean, "x_std:", x_std)

    r_mean = float(np.mean(all_r))
    r_std  = float(np.std(all_r))
    if r_std < 1e-12:
        r_std = 1.0  # avoid divide-by-zero
    # y in this case is our targets, accelerations
    y_mean = np.mean(all_targets, axis=0).astype(np.float32)  # (2,)
    # single scalar std across both components after mean subtraction
    y_std = float(np.std((all_targets - y_mean).reshape(-1)))
    if y_std < 1e-12:
        y_std = 1.0
    print("y_mean:", y_mean, "y_std:", y_std)
    print("r_mean:", r_mean, "r_std:", r_std)
    
    for d in dataset:
        d['edge_feat'][:, 0] = (d['edge_feat'][:, 0]) / r_std
        d['targets'] = (d['targets']) / y_std
        d['nodes'][:, 0] = (d['nodes'][:, 0] - x_mean[0]) / x_std
        d['nodes'][:, 1] = (d['nodes'][:, 1] - x_mean[1]) / x_std
        # store stats so train_model can attach them to the model
        d['x_mean'] = x_mean
        d['x_std']  = x_std

        # store stats so train_model can attach them to the model
        d['y_mean'] = y_mean
        d['y_std']  = y_std
        # --- NEW: store stats so training can attach them to model ---
        d['r_mean'] = r_mean
        d['r_std']  = r_std

    print(f"Dataset prepared: {len(dataset)} frames")
    print(f"Example - Nodes: {dataset[0]['nodes'].shape}, Edges: {dataset[0]['edges'].shape}")
    mask_stats = np.sum([np.sum(d['mask']) for d in dataset])
    total_nodes = np.sum([len(d['mask']) for d in dataset])
    print(f"Mask statistics: {mask_stats:.0f}/{total_nodes} nodes included in training ({100*mask_stats/total_nodes:.1f}%)")
    
    return dataset


def split_dataset(dataset, train_frac=0.7, val_frac=0.15,randomize=True):
    """
    Split dataset into train/validation/test sets.
    
    Parameters:
    -----------
    dataset : list
        Dataset from prepare_dataset
    train_frac : float
        Fraction for training (default 0.7)
    val_frac : float
        Fraction for validation (default 0.15)
    
    Returns:
    --------
    tuple
        (train_data, val_data, test_data, all_data_noshuffle)
    """
    train_size = int(train_frac * len(dataset))
    val_size = int(val_frac * len(dataset))
    all_data_noshuffle = dataset #unshuffled for post-training analysis
    if randomize==True:
        np.random.shuffle(dataset)

    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    return train_data, val_data, test_data, all_data_noshuffle

