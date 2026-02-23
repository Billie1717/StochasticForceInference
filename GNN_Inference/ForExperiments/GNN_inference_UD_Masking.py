"""
ForceGNN Inference Module
=========================
Function-based interface for training and applying ForceGNN to particle trajectory data.

Usage:
    from WorkingVersion.GNN_inference import *
    
    # Load and prepare data
    data = load_data(datadir, dataname)
    data_clean = calculate_derivatives(data, dt=0.1)
    dataset = prepare_dataset(data_clean, cutoff=3.0, max_frames=300)
    
    # Split data
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Train model
    model = create_model(hidden_dim=32)
    model, history = train_model(model, train_data, val_data, num_epochs=100)
    
    # Query force at distance
    force = get_force_at_distance(model, r=1.5)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from collections import defaultdict
from pathlib import Path
from sympy import symbols, lambdify, diff
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

__all__ = [
    'ForceGNN',
    'VoxelNeighborList',
    'load_data',
    'calculate_derivatives2',
    'build_graph',
    'prepare_dataset',
    'split_dataset',
    'create_model',
    'train_model',
    'evaluate_model',
    'evaluate_model_frames',
    'save_model',
    'load_model',
    'save_history',
    'load_history',
    'get_force_at_distance',
    'get_force_environment',
    'get_gamma',
    'estimate_noise_magnitude',
    'compute_weak_derivatives',
]

    # 'plot_force_function',
    # 'plot_training_history',
    # 'plot_predictions',
    # 'plot_force_components',
    # 'visualize_frame'
# ============================================================================
# Model Definition
# ============================================================================

class ForceGNN(tf.keras.Model):
    """
    Graph Neural Network for learning force decomposition.
    
    Forces are decomposed into:
    - F_env: Environmental/external forces (position-dependent, uses node_features[:, :2])
    - F_r: Pairwise interaction forces (distance-dependent)
    - γ * v: Drag/friction term (velocity-dependent, uses node_features[:, 2:4], with learnable scalar γ)
    
    Node features: [x, y, vx, vy, type] - positions, velocities, and particle type
    """
    
    def __init__(self, hidden_dim=32, gamma_init=0.0): #,gamma_init=0.0
        super(ForceGNN, self).__init__()
        self.r_mean = 0.0
        self.r_std  = 1.0
        self.y_mean = np.array([0.0, 0.0], dtype=np.float32)  # (2,)
        self.y_std  = np.array([1.0, 1.0], dtype=np.float32)  
        self.x_mean = np.array([0.0, 0.0], dtype=np.float32)  # mean of [x,y]
        self.x_std  = float(1.0)  # std of [x,y]
        alpha_init = 0.2
        init_logit = np.log(alpha_init / (1 - alpha_init))
        l2_reg = tf.keras.regularizers.l2(0.01)
        init   = tf.keras.initializers.HeNormal()
        self.env_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(2)
        ], name='F_env')
        ##tf.keras.layers.Dense(hidden_dim, activation='elu'),
        # tf.keras.layers.Dense(hidden_dim, activation=None,
        #                          kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        # self.interaction_net = tf.keras.Sequential([
        #     tf.keras.layers.Dense(hidden_dim, activation='elu', kernel_regularizer=l2_reg),
        #     tf.keras.layers.Dense(hidden_dim, activation='elu', kernel_regularizer=l2_reg),
        #     tf.keras.layers.Dense(hidden_dim, activation='elu', kernel_regularizer=l2_reg),
        #     tf.keras.layers.Dense(hidden_dim, activation='elu', kernel_regularizer=l2_reg),
        #     tf.keras.layers.Dense(1)
        # ], name='F_r')
        # tf.keras.layers.Dense(hidden_dim, activation=None,
        #                          kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        self.interaction_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='elu'),
            tf.keras.layers.Dense(hidden_dim, activation='elu'),
            tf.keras.layers.Dense(hidden_dim, activation='elu'),
            tf.keras.layers.Dense(1)
        ], name='F_r')
        # self.interaction_net_2 = tf.keras.Sequential([
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                           kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                          kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('elu'),
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                          kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('elu'),
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                          kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('elu'),
        #     tf.keras.layers.Dense(1)
        # ], name='F_r')
        
        # Learnable drag coefficient γ (initialized to gamma_init)
        self.gamma = tf.Variable(
            initial_value=tf.constant(gamma_init, dtype=tf.float32),
            trainable=True,
            name='gamma',
            dtype=tf.float32)
        
        #self.alpha_logit = tf.Variable(init_logit, dtype=tf.float32, trainable=True, name="alpha_logit")

        
        # self.env_net = tf.keras.Sequential([
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                         kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('tanh'),

        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                         kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('tanh'),

        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                         kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('tanh'),

        #     tf.keras.layers.Dense(2, kernel_initializer=init, kernel_regularizer=l2_reg),
        # ], name="F_env")

#### --------- THis one is the architecture from Ilya's paper --------- ####

        # self.interaction_net = tf.keras.Sequential([
        #     # i = 0  -> LeakyReLU
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        #     # i = 1  -> tanh
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('tanh'),

        #     # i = 2  -> LeakyReLU
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        #     # output
        #     tf.keras.layers.Dense(1, kernel_initializer=init, kernel_regularizer=l2_reg),
        # ], name='F_r')

       
        
        # self.env_net = tf.keras.Sequential([
        #     # i = 0  -> LeakyReLU
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        #     # i = 1  -> tanh
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.Activation('tanh'),

        #     # i = 2  -> LeakyReLU
        #     tf.keras.layers.Dense(hidden_dim, activation=None,
        #                     kernel_initializer=init, kernel_regularizer=l2_reg),
        #     tf.keras.layers.LeakyReLU(alpha=0.1),

        #     # output
        #     tf.keras.layers.Dense(2, kernel_initializer=init, kernel_regularizer=l2_reg),
        # ], name='F_env')   
        
    
    def call(self, inputs, return_components=False):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        inputs : tuple
            (node_features, edge_indices, edge_features)
            - node_features: (N, 5) [x, y, vx, vy, type] - positions, velocities, and type
            - edge_indices: (2, E) edge connections
            - edge_features: (E, 3) edge features [r, dx/r, dy/r]
        return_components : bool
            If True, return individual force components
        
        Returns:
        --------
        F_total or tuple of (F_total, F_env, F_r_aggregated, F_drag)
        """
        node_features, edge_indices, edge_features = inputs
        
        # Extract positions [x, y] for environmental force
        positions = node_features[:, :2]  # (N, 2)
        # Extract velocities [vx, vy] for drag term
        velocities = node_features[:, 2:4]  # (N, 2) - velocities at columns 2-3
        
        # Environmental force (only uses positions)
        F_env = self.env_net(positions)

        
        
        # Pairwise interaction force
        if tf.shape(edge_indices)[1] > 0:
            edge_distances = edge_features[:, 0:1]
            #alpha = tf.sigmoid(self.alpha_logit)
            #edge_type = edge_features[:, 3:4]
            #positions_node1 = tf.gather(positions, edge_indices[0, :])
            #positions_node2 = tf.gather(positions, edge_indices[1, :])
            #midpoint_edge_position = (positions_node1 + positions_node2) / 2.0
            #Make Fr_features = [r,x,y]
            #Fr_features = tf.concat([edge_distances], axis=1)
            Fr_features = tf.concat([edge_distances], axis=1)
            F_r_scalar = self.interaction_net(Fr_features)
            edge_unit_vectors = edge_features[:, 1:3]
            F_r_vector = F_r_scalar * edge_unit_vectors

            
            # Aggregate forces onto target nodes - Billie* Don't understand this step
            target_indices = edge_indices[1, :] #getting the nodes which are the targets of the edges
            indices = tf.expand_dims(target_indices, 1) 
            #F_r_aggregated has the shape of F_env (N,2) and we 
            F_r_aggregated = tf.tensor_scatter_nd_add( 
                tf.zeros_like(positions),
                indices,
                F_r_vector
            )
        else:
            F_r_aggregated = tf.zeros_like(positions)
        
        # Drag/friction term: γ * v
        F_drag = self.gamma * velocities
        
        # Total force
        F_total =   F_drag +F_r_aggregated + F_env #+ F_drag #F_env 
        
        if return_components:
            return F_total,F_r_aggregated,  F_drag #, F_env #, F_drag #F_env
        return F_total


# ============================================================================
# Neighbor List Utilities
# ============================================================================


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

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
t = symbols('t')

# Define the function w which is a sympy expression involving t
w_sym = (t**2-1)**2
def load_data(datadir, dataname):
    """
    Load particle trajectory data from CSV.
    
    Parameters:
    -----------
    datadir : str
        Directory containing the data file
    dataname : str
        Name of the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded data with columns including 'frame', 'x', 'y', etc.
    """
    filepath = Path(datadir) / dataname
    data = pd.read_csv(filepath)
    
    print(f"Data loaded: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Number of frames: {len(data['frame'].unique())}")
    
    return data


def calculate_derivatives2(data, dt=0.1,weak_form = False,naive_form = False,tau = 16):
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
    if 'tx' in data.columns and 'ay' in data.columns:
        print("Derivatives already in data, skipping calculation...")
        frames = sorted(data['frame'].unique())
        data_clean = data[(data['frame'] > frames[0]) & (data['frame'] < frames[-1])].copy()
        return data_clean
    
    print("Calculating derivatives...")
    print("with acceleration")
    print("got here in script")
    
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
    print("Calculating naïve derivatives...")
    if naive_form:
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
                data.loc[idx, 'vx'] = (data.loc[idx_next, 'x'] - data.loc[idx_prev, 'x']) / (2*dt)
                data.loc[idx, 'vy'] = (data.loc[idx_next, 'y'] - data.loc[idx_prev, 'y']) / (2*dt)
                #data.loc[idx, 'vx'] = (data.loc[idx_next, 'x'] - data.loc[idx, 'x']) / (dt)
                #data.loc[idx, 'vy'] = (data.loc[idx_next, 'y'] - data.loc[idx, 'y']) / (dt)
                
                # Acceleration
                data.loc[idx, 'ax'] = (data.loc[idx_next, 'x'] - 2*data.loc[idx, 'x'] + data.loc[idx_prev, 'x']) / (dt**2)
                data.loc[idx, 'ay'] = (data.loc[idx_next, 'y'] - 2*data.loc[idx, 'y'] + data.loc[idx_prev, 'y']) / (dt**2)
    print("Calculating weak form derivatives...")
    print("Pre-weak form data: ",data.head())
    if weak_form and not naive_form:
        #for pid in particles:
            #particle_mask = data['particle'] == pid
            #particle_data = data[particle_mask].sort_values('frame')
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
    print("Post-weak form data: ",data.head())
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
        # specifically get rid of the nodes (in only that frame) which have zero ax and ay
        #data = data[data['ax'] != 0]
        #I data = data[data['ay'] != 0]
    print("Post-weak and naive form data: ",data.head())
    
    # Remove boundary frames (first and last)
    frames = sorted(data['frame'].unique())
    #data_clean = data[(data['frame'] > frames[0]) & (data['frame'] < frames[-1])].copy()
    data_clean = data.copy()
    print(f"After removing boundary frames: {data_clean.shape}")
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
        (nodes, num_neighbors, edges, edge_features, mask, nn_r)
        - nodes: (N, 4) array of [x, y, vx, vy]
        - num_neighbors: (N,) array of number of neighbors per node
        - edges: (2, E) array of edge indices [source, target]
        - edge_features: (E, 3) array of [distance, dx/r, dy/r]
        - mask: (N,) array of mask values (1 = train, 0 = neighbor only), defaults to all 1s if not provided
        - nn_r: (N,) array of nearest neighbor distance per node (np.nan for nodes with no neighbors)
    """
    N = len(frame_data)
    nodes = frame_data[['x', 'y', 'vx', 'vy']].values #, 'vx', 'vy'
    edges = []
    edge_features = []
    
    # Initialize nearest neighbor distance per node
    nn_r_per_node = np.full(N, np.inf, dtype=np.float32)
    
    # Extract mask if present, otherwise default to all 1s (all nodes included in training)
    if 'mask' in frame_data.columns:
        mask = frame_data['mask'].values.astype(np.float32)
    else:
        mask = np.ones(N, dtype=np.float32)
    # now make mask a zero if ax, ay, vx, vy are zero
    mask = mask & (frame_data['ax'] != 0) & (frame_data['ay'] != 0) & (frame_data['vx'] != 0) & (frame_data['vy'] != 0)
    #also make all particles of type 2 have mask = 0
    
    #mask = mask & (frame_data['type'] != 2)
    # Use numpy array for counting neighbors (much faster than DataFrame.loc in loops)
    num_neighbors_arr = np.zeros(N, dtype=np.int32)
    
    positions = frame_data[['x', 'y']].to_numpy()
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
                #if NumNeighbors<3:
                #    frame_data.loc[row_idx, 'mask'] = 0
                num_neighbors_arr[row_idx] = NumNeighbors
                unit_dx = dx / r
                unit_dy = dy / r
                edges.append([row_idx, target_idx])
                edge_features.append([r, unit_dx, unit_dy])
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

            # Directed edge i -> j
            edges.append([i, j])
            edge_features.append([r, unit_dx, unit_dy])
            # Update nearest neighbor distance
            if r < nn_r_per_node[i]:
                nn_r_per_node[i] = r
            # Count neighbor for node i (use numpy array for speed)
            num_neighbors_arr[i] += 1

            # Directed edge j -> i (negative direction)
            edges.append([j, i])
            edge_features.append([r, -unit_dx, -unit_dy])
            # Update nearest neighbor distance
            if r < nn_r_per_node[j]:
                nn_r_per_node[j] = r
            # Count neighbor for node j (use numpy array for speed)
            num_neighbors_arr[j] += 1
    #mask = mask & (frame_data['num_neighbors'] > 2)
    #mask cells which are 20 away from the edge of the domain
    x_min = frame_data['x'].min()
    x_max = frame_data['x'].max()
    y_min = frame_data['y'].min()
    y_max = frame_data['y'].max()
    mask = mask & (frame_data['x'] > x_min + 20) & (frame_data['x'] < x_max - 20) & (frame_data['y'] > y_min + 20) & (frame_data['y'] < y_max - 20)
    
    #mask cells which have an acceleration magnitude higher than 5
    #mask = mask & (np.sqrt(frame_data['ax']**2 + frame_data['ay']**2) < 5)

    # Convert infinity to NaN for nodes with no neighbors
    nn_r_per_node[nn_r_per_node == np.inf] = np.nan
    
    return (
        nodes.astype(np.float32),
        num_neighbors_arr.astype(np.float32),
        np.array(edges, dtype=np.int32).T if edges else np.zeros((2, 0), dtype=np.int32),
        np.array(edge_features, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32),
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
        nodes, num_neighbors, edges, edge_feat, mask, nn_r = build_graph(frame_data, cutoff, neighbor_pairs=neighbor_pairs)
        
        # Targets (accelerations)
        targets = frame_data[['ax', 'ay']].values.astype(np.float32)
        
        # Note: nodes already contain [x, y, vx, vy] from build_graph
        # Mask: 1 = include in training loss, 0 = neighbor only

        dataset.append({
            'frame': frame_id,
            'nodes': nodes,
            'num_neighbors': num_neighbors,
            'edges': edges,
            'edge_feat': edge_feat,
            'targets': targets,
            'mask': mask,
            'particle_ids': frame_data['particle'].to_numpy(dtype=np.int32) if 'particle' in frame_data.columns else np.arange(len(frame_data)),
            'nn_r': nn_r  # Nearest neighbor distance per node (computed in build_graph)
        })
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
    #max_pairwise_distance = np.max(np.concatenate(all_pairwise_distances))
    # --- NEW: z-score parameters for r ---
    r_mean = float(np.mean(all_r))
    r_std  = float(np.std(all_r))
    if r_std < 1e-12:
        r_std = 1.0  # avoid divide-by-zero
    # --- NEW: target normalisation stats ---
    y_mean = np.mean(all_targets, axis=0).astype(np.float32)  # (2,)
    # single scalar std across both components after mean subtraction
    y_std = float(np.std((all_targets - y_mean).reshape(-1)))
    if y_std < 1e-12:
        y_std = 1.0
    print("y_mean:", y_mean, "y_std:", y_std)
    print("r_mean:", r_mean, "r_std:", r_std)
    
    for d in dataset:
        # --- CHANGED: was division by max_pairwise_distance ---
        d['edge_feat'][:, 0] = (d['edge_feat'][:, 0] - r_mean) / r_std
        # --- NEW: normalise targets ---
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
        # nn_r is already computed in build_graph (before normalization), so just store it
        # (it's already in the dataset dict from above, but we keep it here for consistency)
        #calculate local density


        #d['targets'] = d['targets'] / max_space

    print(f"Dataset prepared: {len(dataset)} frames")
    print(f"Example - Nodes: {dataset[0]['nodes'].shape}, Edges: {dataset[0]['edges'].shape}")
    mask_stats = np.sum([np.sum(d['mask']) for d in dataset])
    total_nodes = np.sum([len(d['mask']) for d in dataset])
    print(f"Mask statistics: {mask_stats:.0f}/{total_nodes} nodes included in training ({100*mask_stats/total_nodes:.1f}%)")
    
    return dataset


def split_dataset(dataset, train_frac=0.7, val_frac=0.15):
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
    all_data_noshuffle = dataset
    np.random.shuffle(dataset)
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    return train_data, val_data, test_data, all_data_noshuffle


# ============================================================================
# Model Training
# ============================================================================

def create_model(hidden_dim=32,gamma_init=0.0):
    """
    Create a ForceGNN model.
    
    Parameters:
    -----------
    hidden_dim : int
        Hidden dimension for neural network layers
    
    Returns:
    --------
    ForceGNN
        Initialized model
    """
    model = ForceGNN(hidden_dim=hidden_dim, gamma_init=gamma_init)
    print(f"ForceGNN model created with hidden_dim={hidden_dim}, gamma_init={gamma_init}")
    return model


def train_model(model, train_data, val_data,far_field_reg = 0,far_field_r = 100, num_epochs=100, learning_rate=0.001, verbose=True,max_r  = 100):
    """
    Train the ForceGNN model.
    
    Parameters:
    -----------
    model : ForceGNN
        Model to train
    train_data : list
        Training dataset (each entry should have 'mask' field if masking is used)
    val_data : list
        Validation dataset (each entry should have 'mask' field if masking is used)
    num_epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for Adam optimizer
    verbose : bool
        Print progress during training
    
    Returns:
    --------
    tuple
        (model, history) where history is dict with 'train_loss' and 'val_loss'
    """
    #r_far = 70.0          # physical units
    #lambda_far = 100      # tune
    Rthresh = far_field_r   # physical units
    Rmax    = max_r  # choose "definitely far"
    K_far   = 1000      # samples per step
    lambda_far = far_field_reg

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # --- NEW: set model normalisation from dataset if present ---
    if len(train_data) > 0 and isinstance(train_data[0], dict) and ('r_mean' in train_data[0]) and ('r_std' in train_data[0]):
        model.r_mean = float(train_data[0]['r_mean'])
        model.r_std  = float(train_data[0]['r_std'])
        if model.r_std == 0.0:
            model.r_std = 1.0
        if verbose:
            print(f"Using r normalisation: mean={model.r_mean:.6g}, std={model.r_std:.6g}")
    # ----------------------------------------------------------
    # --- NEW: set model target normalisation from dataset if present ---
    if len(train_data) > 0 and ('y_mean' in train_data[0]) and ('y_std' in train_data[0]):
        model.y_mean = np.array(train_data[0]['y_mean'], dtype=np.float32)
        model.y_std  = float(train_data[0]['y_std'])
        if model.y_std == 0.0:
            model.y_std = 1.0
        if verbose:
            print(f"Using target normalisation: y_mean={model.y_mean}, y_std={model.y_std:.6g}")
    if len(train_data) > 0 and ('x_mean' in train_data[0]) and ('x_std' in train_data[0]):
        model.x_mean = np.array(train_data[0]['x_mean'], dtype=np.float32)
        model.x_std  = float(train_data[0]['x_std'])
        if model.x_std == 0.0:
            model.x_std = 1.0
        if verbose:
            print(f"Using x normalisation: mean={model.x_mean}, std={model.x_std}")
    
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for data_point in train_data:
            # Skip frames with no masked nodes (to avoid gradient issues)
            if 'mask' in data_point:
                mask_np = data_point['mask']
                if np.sum(mask_np) == 0:
                    continue  # Skip this frame entirely
            
            with tf.GradientTape() as tape:
                inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
                predictions = model(inputs)
                targets = tf.cast(data_point['targets'], tf.float32)
                
                # Apply masking: only compute loss on nodes where mask=1
                if 'mask' in data_point:
                    mask = tf.cast(data_point['mask'], tf.float32)
                    # Expand mask to (N, 2) to match predictions/targets shape
                    mask_expanded = tf.expand_dims(mask, 1)  # (N, 1)
                    squared_errors = (predictions - targets)**2  # (N, 2)
                    masked_errors = squared_errors * mask_expanded  # (N, 2)
                    # Compute mean only over masked nodes
                    num_masked_nodes = tf.reduce_sum(mask)
                    loss = tf.reduce_sum(masked_errors) / (num_masked_nodes * 2.0)  # divide by num_nodes * 2 dims
                    # --- NEW: far-field regularisation f_int(r_far) ~ 0 ---
                    # r_mean = float(getattr(model, "r_mean", 0.0))
                    # r_std  = float(getattr(model, "r_std", 1.0)) or 1.0
                    # r_far_norm = (far_field_r - r_mean) / r_std

                    # F_far = model.interaction_net(tf.constant([[r_far_norm]], dtype=tf.float32))  # shape (1,1)
                    # # add loss to all distances r>r_far

                    # loss += far_field_reg * tf.reduce_mean(tf.square(F_far))
                    # --- Far-range regularisation: enforce f_int(r)=0 for r>Rthresh ---
                    r_mean = float(getattr(model, "r_mean", 0.0))
                    r_std  = float(getattr(model, "r_std", 1.0))
                    if r_std == 0.0:
                        r_std = 1.0

                    # sample physical r in [Rthresh, Rmax]
                    r_far_phys = tf.random.uniform([K_far, 1], minval=Rthresh, maxval=Rmax, dtype=tf.float32)

                    # convert to the normalised r used by interaction_net
                    r_far_norm = (r_far_phys - r_mean) / r_std

                    F_far = model.interaction_net(r_far_norm)  # shape (K_far,1), in *normalised output units*
                    F_far_phys = F_far
                    loss += lambda_far * tf.reduce_mean(tf.square(F_far_phys))
                    #print("Far field loss: ",far_field_reg * tf.reduce_mean(tf.square(F_far)))
                else:
                    # No masking: compute loss on all nodes
                    loss = tf.reduce_mean((predictions - targets)**2)
                    # --- NEW: far-field regularisation f_int(r_far) ~ 0 ---
                    # r_mean = float(getattr(model, "r_mean", 0.0))
                    # r_std  = float(getattr(model, "r_std", 1.0)) or 1.0
                    # r_far_norm = (far_field_r - r_mean) / r_std

                    # F_far = model.interaction_net(tf.constant([[r_far_norm]], dtype=tf.float32))  # shape (1,1)
                    # loss += far_field_reg * tf.reduce_mean(tf.square(F_far))
                    r_mean = float(getattr(model, "r_mean", 0.0))
                    r_std  = float(getattr(model, "r_std", 1.0))
                    if r_std == 0.0:
                        r_std = 1.0

                    # sample physical r in [Rthresh, Rmax]
                    r_far_phys = tf.random.uniform([K_far, 1], minval=Rthresh, maxval=Rmax, dtype=tf.float32)

                    # convert to the normalised r used by interaction_net
                    r_far_norm = (r_far_phys - r_mean) / r_std

                    F_far = model.interaction_net(r_far_norm)  # shape (K_far,1), in *normalised output units*
                    F_far_phys = F_far
                    loss += lambda_far * tf.reduce_mean(tf.square(F_far_phys))
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_losses.append(loss.numpy())
        
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation
        val_epoch_losses = []
        for data_point in val_data:
            # Skip frames with no masked nodes (to avoid loss computation issues)
            if 'mask' in data_point:
                mask_np = data_point['mask']
                if np.sum(mask_np) == 0:
                    continue  # Skip this frame entirely
            
            inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
            predictions = model(inputs)
            targets = tf.cast(data_point['targets'], tf.float32)
            
            # Apply masking for validation as well
            if 'mask' in data_point:
                mask = tf.cast(data_point['mask'], tf.float32)
                mask_expanded = tf.expand_dims(mask, 1)  # (N, 1)
                squared_errors = (predictions - targets)**2  # (N, 2)
                masked_errors = squared_errors * mask_expanded  # (N, 2)
                num_masked_nodes = tf.reduce_sum(mask)
                loss = tf.reduce_sum(masked_errors) / (num_masked_nodes * 2.0)
                # r_mean = float(getattr(model, "r_mean", 0.0))
                # r_std  = float(getattr(model, "r_std", 1.0)) or 1.0
                # r_far_norm = (far_field_r - r_mean) / r_std

                # F_far = model.interaction_net(tf.constant([[r_far_norm]], dtype=tf.float32))  # shape (1,1)
                # loss += far_field_reg * tf.reduce_mean(tf.square(F_far))

                r_mean = float(getattr(model, "r_mean", 0.0))
                r_std  = float(getattr(model, "r_std", 1.0))
                if r_std == 0.0:
                    r_std = 1.0

                # sample physical r in [Rthresh, Rmax]
                r_far_phys = tf.random.uniform([K_far, 1], minval=Rthresh, maxval=Rmax, dtype=tf.float32)

                # convert to the normalised r used by interaction_net
                r_far_norm = (r_far_phys - r_mean) / r_std

                F_far = model.interaction_net(r_far_norm)  # shape (K_far,1), in *normalised output units*
                loss += lambda_far * tf.reduce_mean(tf.square(F_far))
                #F_far_phys =  F_far
                #loss += lambda_far * tf.reduce_mean(tf.square(F_far_phys))
            else:
                loss = tf.reduce_mean((predictions - targets)**2)
                # r_mean = float(getattr(model, "r_mean", 0.0))
                # r_std  = float(getattr(model, "r_std", 1.0)) or 1.0
                # r_far_norm = (far_field_r - r_mean) / r_std

                # F_far = model.interaction_net(tf.constant([[r_far_norm]], dtype=tf.float32))  # shape (1,1)
                # loss += far_field_reg * tf.reduce_mean(tf.square(F_far))
                r_mean = float(getattr(model, "r_mean", 0.0))
                r_std  = float(getattr(model, "r_std", 1.0))
                if r_std == 0.0:
                    r_std = 1.0

                # sample physical r in [Rthresh, Rmax]
                r_far_phys = tf.random.uniform([K_far, 1], minval=Rthresh, maxval=Rmax, dtype=tf.float32)

                # convert to the normalised r used by interaction_net
                r_far_norm = (r_far_phys - r_mean) / r_std

                F_far = model.interaction_net(r_far_norm)  # shape (K_far,1), in *normalised output units*
                loss += lambda_far * tf.reduce_mean(tf.square(F_far))
                #F_far_phys = model.y_std * F_far
                #loss += lambda_far * tf.reduce_mean(tf.square(F_far_phys))
            val_epoch_losses.append(loss.numpy())
        
        val_loss = np.mean(val_epoch_losses)
        val_losses.append(val_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    if verbose:
        print("\nTraining complete!")
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history

def evaluate_model(model, test_data, verbose=True):
    test_predictions_masked = []
    test_targets_masked = []
    index_map = []  # (frame_idx, node_idx) for each masked row

    total_nodes = 0
    masked_nodes = 0

    for frame_idx, data_point in enumerate(test_data):
        inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
        predictions = model(inputs).numpy()
        targets = data_point['targets']

        if 'mask' in data_point:
            mask = data_point['mask']
        else:
            mask = np.ones(len(predictions), dtype=np.float32)

        mask_indices = mask == 1.0

        masked_pred = predictions[mask_indices]
        masked_targ = targets[mask_indices]

        test_predictions_masked.append(masked_pred)
        test_targets_masked.append(masked_targ)

        # record original indices
        node_ids = np.where(mask_indices)[0]
        index_map.extend([(frame_idx, int(n)) for n in node_ids])

        total_nodes += len(predictions)
        masked_nodes += np.sum(mask_indices)

    if len(test_predictions_masked) > 0:
        test_predictions_flat = np.vstack(test_predictions_masked)
        test_targets_flat = np.vstack(test_targets_masked)
        num_masked_nodes = len(test_predictions_flat)

        if num_masked_nodes > 0:
            squared_errors = (test_predictions_flat - test_targets_flat) ** 2
            test_mse_x = np.mean(squared_errors[:, 0])
            test_mse_y = np.mean(squared_errors[:, 1])
            test_mse = np.mean(squared_errors)
            test_mae = np.mean(np.abs(test_predictions_flat - test_targets_flat))
        else:
            test_mse_x = test_mse_y = test_mse = test_mae = 0.0
    else:
        test_predictions_flat = np.zeros((0, 2), dtype=np.float32)
        test_targets_flat = np.zeros((0, 2), dtype=np.float32)
        index_map = []
        num_masked_nodes = 0
        test_mse_x = test_mse_y = test_mse = test_mae = 0.0

    if verbose:
        print(f"\nTest Set Evaluation (masked nodes only):")
        print(f"  Total nodes in test set: {total_nodes}")
        print(f"  Masked nodes (mask=1): {masked_nodes} ({100*masked_nodes/total_nodes:.1f}%)")
        print(f"  MSE (x-direction): {test_mse_x:.6f} (on {num_masked_nodes:.0f} masked nodes)")
        print(f"  MSE (y-direction): {test_mse_y:.6f} (on {num_masked_nodes:.0f} masked nodes)")
        print(f"  MSE (overall): {test_mse:.6f} (on {num_masked_nodes:.0f} masked nodes)")
        print(f"  MAE: {test_mae:.6f} (on {num_masked_nodes:.0f} masked nodes)")

    return {
        'mse': test_mse,
        'mse_x': test_mse_x,
        'mse_y': test_mse_y,
        'mae': test_mae,
        'predictions': test_predictions_flat,
        'targets': test_targets_flat,
        'index_map': index_map,              # <-- add this
        'num_masked_nodes': num_masked_nodes,
        'total_nodes': total_nodes
    }


def evaluate_model_frames(model, test_data, verbose=True):
    """
    Evaluate model on test data and return a DataFrame with error and frame information.
    
    This function works with datasets that keep frame and node_ids consistent across frames.
    It adds columns for frame, node_id, and error to track how error changes across 
    consecutive frames for a given node.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    test_data : list
        Test dataset (each entry should have 'frame', 'nodes', 'edges', 'edge_feat', 'targets',
        and optionally 'mask' and 'particle_ids' fields)
    verbose : bool
        Print summary statistics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'frame': Frame index
        - 'node_id': Node identifier (from 'particle_ids' if available, otherwise index)
        - 'x': x-position from nodes array
        - 'y': y-position from nodes array
        - 'n_neighbors': Number of nearest neighbors
        - 'nn_r': nearest neighbour distance
        - 'error': Overall error (L2 norm of prediction - target)
        - 'error_x': Error in x-direction
        - 'error_y': Error in y-direction
        - 'pred_x': Predicted x-component
        - 'pred_y': Predicted y-component
        - 'target_x': Target x-component
        - 'target_y': Target y-component
        Only includes nodes where mask=1 (if mask is present)
    """
    rows = []
    
    for frame_idx, data_point in enumerate(test_data):
        frame_id = data_point.get('frame', None)
        if frame_id is None:
            # If no frame field, use index
            frame_id = frame_idx
        
        inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
        predictions = model(inputs).numpy()
        targets = data_point['targets']
        nodes = data_point['nodes']  # Get nodes array for positions
        
        # Get mask if present, otherwise default to all 1s
        if 'mask' in data_point:
            mask = data_point['mask']
        else:
            mask = np.ones(len(predictions), dtype=np.float32)
        
        # Get particle_ids if available, otherwise use indices
        if 'particle_ids' in data_point:
            node_ids = data_point['particle_ids']
        else:
            node_ids = np.arange(len(predictions))
        
        mask_indices = mask == 1.0
        
        # Calculate errors for all nodes (we'll filter by mask when creating DataFrame)
        theta_pred = np.arctan2(predictions[:, 1],predictions[:, 0])
        theta_true = np.arctan2(targets[:, 1],targets[:, 0])

        errors_theta = np.abs(theta_pred - theta_true)
        errors_x = (predictions[:, 0] - targets[:, 0])
        errors_y = (predictions[:, 1] - targets[:, 1])
        errors = np.sqrt(errors_x**2 + errors_y**2)
        num_neighbors_arr = data_point['num_neighbors']
        nn_r_arr = data_point['nn_r']
        # Create rows for all nodes (masked or not, but we can filter later if needed)
        for i in range(len(predictions)):
            if mask_indices[i]:  # Only include masked nodes
                rows.append({
                    'frame': frame_id,
                    'node_id': int(node_ids[i]) if hasattr(node_ids[i], '__int__') else node_ids[i],
                    'x': float(nodes[i, 0]),  # x-position
                    'y': float(nodes[i, 1]),  # y-position
                    'num_neighbors': float(num_neighbors_arr[i]),
                    'nn_r': float(nn_r_arr[i]),
                    'error': float(errors[i]),
                    'error_theta': float(errors_theta[i]),
                    'error_x': float(errors_x[i]),
                    'error_y': float(errors_y[i]),
                    'pred_x': float(predictions[i, 0]),
                    'pred_y': float(predictions[i, 1]),
                    'target_x': float(targets[i, 0]),
                    'target_y': float(targets[i, 1])
                })
    
    df = pd.DataFrame(rows)
    
    if verbose:
        if len(df) > 0:
            print(f"\nFrame-based Evaluation:")
            print(f"  Total rows: {len(df)}")
            print(f"  Unique frames: {df['frame'].nunique()}")
            print(f"  Unique nodes: {df['node_id'].nunique()}")
            print(f"  Mean error: {df['error'].mean():.6f}")
            print(f"  Median error: {df['error'].median():.6f}")
            print(f"  Std error: {df['error'].std():.6f}")
        else:
            print("\nFrame-based Evaluation: No data (all nodes may have been masked out)")
    
    return df

# def evaluate_model(model, test_data, verbose=True):
#     """
#     Evaluate model on test data.
    
#     Only evaluates on nodes where mask=1. Nodes with mask=0 are completely excluded
#     from predictions, targets, and metrics.
    
#     Parameters:
#     -----------
#     model : ForceGNN
#         Trained model
#     test_data : list
#         Test dataset (each entry should have 'mask' field if masking is used)
#     verbose : bool
#         Print results
    
#     Returns:
#     --------
#     dict
#         Dictionary with 'mse', 'mse_x', 'mse_y', 'mae', 'predictions', 'targets'
#         - 'mse': Overall MSE across both dimensions (for backward compatibility)
#         - 'mse_x': MSE for x-direction predictions
#         - 'mse_y': MSE for y-direction predictions
#         - 'mae': Mean absolute error
#         All arrays contain ONLY nodes where mask=1
#     """
#     test_predictions_masked = []
#     test_targets_masked = []
    
#     total_nodes = 0
#     masked_nodes = 0
    
#     for data_point in test_data:
#         inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
#         predictions = model(inputs).numpy()
#         targets = data_point['targets']
        
#         # Get mask if present, otherwise default to all 1s
#         if 'mask' in data_point:
#             mask = data_point['mask']
#         else:
#             mask = np.ones(len(predictions), dtype=np.float32)
        
#         # Filter to only include nodes where mask=1
#         mask_indices = mask == 1.0
#         masked_pred = predictions[mask_indices]
#         masked_targ = targets[mask_indices]
        
#         test_predictions_masked.append(masked_pred)
#         test_targets_masked.append(masked_targ)
        
#         total_nodes += len(predictions)
#         masked_nodes += np.sum(mask_indices)
    
#     # Stack all masked predictions and targets
#     if len(test_predictions_masked) > 0:
#         test_predictions_flat = np.vstack(test_predictions_masked)
#         test_targets_flat = np.vstack(test_targets_masked)
        
#         num_masked_nodes = len(test_predictions_flat)
        
#         if num_masked_nodes > 0:
#             # Compute metrics on masked nodes only
#             squared_errors = (test_predictions_flat - test_targets_flat)**2
#             # Separate MSE for x and y directions
#             test_mse_x = np.mean(squared_errors[:, 0])  # x-direction (column 0)
#             test_mse_y = np.mean(squared_errors[:, 1])  # y-direction (column 1)
#             test_mse = np.mean(squared_errors)  # Overall MSE (for backward compatibility)
#             test_mae = np.mean(np.abs(test_predictions_flat - test_targets_flat))
#         else:
#             test_mse_x = 0.0
#             test_mse_y = 0.0
#             test_mse = 0.0
#             test_mae = 0.0
#     else:
#         test_predictions_flat = np.zeros((0, 2), dtype=np.float32)
#         test_targets_flat = np.zeros((0, 2), dtype=np.float32)
#         num_masked_nodes = 0
#         test_mse_x = 0.0
#         test_mse_y = 0.0
#         test_mse = 0.0
#         test_mae = 0.0
    
#     if verbose:
#         print(f"\nTest Set Evaluation (masked nodes only):")
#         print(f"  Total nodes in test set: {total_nodes}")
#         print(f"  Masked nodes (mask=1): {masked_nodes} ({100*masked_nodes/total_nodes:.1f}%)")
#         print(f"  MSE (x-direction): {test_mse_x:.6f} (on {num_masked_nodes:.0f} masked nodes)")
#         print(f"  MSE (y-direction): {test_mse_y:.6f} (on {num_masked_nodes:.0f} masked nodes)")
#         print(f"  MSE (overall): {test_mse:.6f} (on {num_masked_nodes:.0f} masked nodes)")
#         print(f"  MAE: {test_mae:.6f} (on {num_masked_nodes:.0f} masked nodes)")
    
#     return {
#         'mse': test_mse,  # Overall MSE (for backward compatibility)
#         'mse_x': test_mse_x,  # MSE for x-direction predictions
#         'mse_y': test_mse_y,  # MSE for y-direction predictions
#         'mae': test_mae,
#         'predictions': test_predictions_flat,  # Only masked nodes
#         'targets': test_targets_flat,  # Only masked nodes
#         'num_masked_nodes': num_masked_nodes,
#         'total_nodes': total_nodes
#     }


# ============================================================================
# Force Query Functions
# ============================================================================

# def get_force_at_distance(model, r):
#     """
#     Get the learned pairwise force magnitude at a given distance.
    
#     Parameters:
#     -----------
#     model : ForceGNN
#         Trained model
#     r : float
#         Distance value
#     type1 : int
#         Type of the first particle
#     type2 : int
#         Type of the second particle
#     Returns:
#     --------
#     float
#         Force magnitude F_r(r)
#     """
#     return model.interaction_net(tf.constant([[r]], dtype=tf.float32)).numpy()[0, 0]

def get_force_at_distance(model, r):
    """
    Get the learned pairwise force magnitude at a given distance.
    Uses the same r normalisation as training if present on the model.
    """
    r_mean = float(getattr(model, "r_mean", 0.0))
    r_std  = float(getattr(model, "r_std", 1.0))
    if r_std == 0.0:
        r_std = 1.0
    y_std = float(getattr(model, "y_std", 1.0))
    if y_std == 0.0:
        y_std = 1.0

    r_norm = (float(r) - r_mean) / r_std
    F_norm = model.interaction_net(tf.constant([[r_norm]], dtype=tf.float32)).numpy()[0, 0]

    # de-normalise scalar magnitude
    return y_std * F_norm

# def get_force_function(model, r_min=0.5, r_max=3.5, num_points=100):
#     """
#     Get the learned force function over a range of distances.
    
#     Parameters:
#     -----------
#     model : ForceGNN
#         Trained model
#     r_min : float
#         Minimum distance
#     r_max : float
#         Maximum distance
#     num_points : int
#         Number of points to sample
    
#     Returns:
#     --------
#     tuple
#         (r_values, F_r_values) arrays
#     """
#     r_values = np.linspace(r_min, r_max, num_points).astype(np.float32)[:, None]
#     F_r_values = model.interaction_net(r_values).numpy()
#     return r_values.flatten(), F_r_values.flatten()


# def get_force_environment(model, x, y):
#     """
#     Get the learned environmental force at a given position.
    
#     Parameters:
#     -----------
#     model : ForceGNN
#         Trained model
#     x : float
#         x-coordinate
#     y : float
#         y-coordinate
    
#     Returns:
#     --------
#     np.ndarray
#         Environmental force vector [F_env_x, F_env_y]
    
#     Raises:
#     -------
#     AttributeError
#         If the model does not have an env_net (F_env was not used in training)
#     """
#     if not hasattr(model, 'env_net') or model.env_net is None:
#         raise AttributeError(
#             "This model does not have an env_net (F_env component). "
#             "The model was trained without environmental forces."
#         )
#     return model.env_net(tf.constant([[x, y]], dtype=tf.float32)).numpy()[0, :]

def get_force_environment(model, x, y):
    if not hasattr(model, 'env_net') or model.env_net is None:
        raise AttributeError(
            "This model does not have an env_net (F_env component). "
            "The model was trained without environmental forces."
        )

    x_mean = np.array(getattr(model, "x_mean", [0.0, 0.0]), dtype=np.float32)
    x_std  = float(getattr(model, "x_std",  1.0))
    if x_std == 0.0:
        x_std = 1.0

    # normalise query position
    x_norm = (float(x) - float(x_mean[0])) / x_std
    y_norm = (float(y) - float(x_mean[1])) / float(x_std)

    F_env_norm = model.env_net(tf.constant([[x_norm, y_norm]], dtype=tf.float32)).numpy()[0, :]

    # de-normalise output to physical units (consistent with targets / y_std scaling)
    y_std = float(getattr(model, "y_std", 1.0))
    if y_std == 0.0:
        y_std = 1.0

    return y_std * F_env_norm



def get_gamma(model):
    """
    Get the learned drag coefficient γ.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    
    Returns:
    --------
    float
        The learned drag coefficient γ
    """
    return float(model.gamma.numpy())


def save_model(model, filepath, save_format='weights', history=None):
    """
    Save a trained ForceGNN model.
    
    For subclassed models, saving weights is the most reliable method.
    The model structure must be recreated using create_model() before loading.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model to save
    filepath : str
        Path where to save the model
    save_format : str, default 'weights'
        Format to save in:
        - 'weights': Save only weights (recommended for subclassed models)
        - 'savedmodel': Save as TensorFlow SavedModel format
        - 'tf': Alias for 'savedmodel'
    history : dict, optional
        Training history dictionary (e.g., from train_model). If provided,
        will be saved to a separate JSON file with '_history.json' suffix.
    
    Returns:
    --------
    None
    """
    if save_format in ['weights', 'h5']:
        # Save weights only (most reliable for subclassed models)
        if not filepath.endswith('.h5') and not filepath.endswith('.weights.h5'):
            filepath = filepath + '.h5'
        model.save_weights(filepath)
        print(f"Model weights saved to: {filepath}")
        print("Note: Use load_model() to recreate the model structure and load weights.")
    elif save_format in ['savedmodel', 'tf']:
        # Save as SavedModel format
        model.save(filepath, save_format='tf')
        print(f"Model saved as SavedModel to: {filepath}")
    else:
        raise ValueError(f"Unknown save_format: {save_format}. Use 'weights' or 'savedmodel'.")
    # Save normalisation stats if present on model
    norm = {
        "r_mean": float(getattr(model, "r_mean", 0.0)),
        "r_std":  float(getattr(model, "r_std", 1.0)),
        "y_std":  float(getattr(model, "y_std", 1.0)),
        "x_mean": [float(getattr(model, "x_mean", [0.0, 0.0])[0]),
                float(getattr(model, "x_mean", [0.0, 0.0])[1])],
        "x_std":  float(getattr(model, "x_std",  1.0)),
    }
    norm_filepath = filepath.rsplit('.', 1)[0] + '_norm.json'
    with open(norm_filepath, 'w') as f:
        json.dump(norm, f, indent=2)
        
    print(f"Normalisation stats saved to: {norm_filepath}")
    # Save history if provided
    if history is not None:
        history_filepath = filepath.rsplit('.', 1)[0] + '_history.json'
        save_history(history, history_filepath)


def load_model(filepath, hidden_dim=None, gamma_init=None, save_format='weights'):
    """
    Load a saved ForceGNN model.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    hidden_dim : int, optional
        Hidden dimension used when creating the model. Required if save_format='weights'.
        If None, will try to infer from saved model (only works for SavedModel format).
    gamma_init : float, optional
        Initial gamma value used when creating the model. Required if save_format='weights'.
        If None, will try to infer from saved model (only works for SavedModel format).
    save_format : str, default 'weights'
        Format the model was saved in:
        - 'weights': Load from weights file (requires hidden_dim and gamma_init)
        - 'savedmodel': Load from SavedModel format
        - 'tf': Alias for 'savedmodel'
    
    Returns:
    --------
    ForceGNN
        Loaded model
    """
    if save_format in ['weights', 'h5']:
        # Load from weights - need to recreate model structure first
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided when loading from weights. "
                           "Use the same value as when you created the model.")
        if gamma_init is None:
            # Default to 0.0 if not provided (will be overwritten by loaded weights)
            gamma_init = 0.0
            print("Warning: gamma_init not provided, using 0.0. The loaded weights will override this.")
        
        # Recreate model structure
        model = create_model(hidden_dim=hidden_dim, gamma_init=gamma_init)
        
        # For subclassed models, we need to call the model first to create variables
        # Explicitly build the sub-networks first to ensure all layers are initialized
        # Build env_net (takes 2D positions) - only if it exists
        if hasattr(model, 'env_net') and model.env_net is not None:
            model.env_net.build(input_shape=(None, 2))
        # Build interaction_net (takes 1D distance features)
        model.interaction_net.build(input_shape=(None, 1))
        
        # Also call the model with dummy inputs to ensure everything is connected
        # Need at least 2 nodes and 1 edge to initialize interaction_net
        dummy_nodes = tf.zeros((2, 4), dtype=tf.float32)  # (N, 4) for [x, y, vx, vy] - need at least 2 nodes
        dummy_edges = tf.constant([[0], [1]], dtype=tf.int32)  # (2, E) - one edge from node 0 to node 1
        dummy_edge_feat = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)  # (E, 3) - [r, dx/r, dy/r]
        dummy_inputs = (dummy_nodes, dummy_edges, dummy_edge_feat)
        
        # Call model to initialize variables (this ensures all connections are built)
        _ = model(dummy_inputs, training=False)
        
        # Now load weights
        model.load_weights(filepath)
        # Try to restore normalisation stats saved alongside weights
        norm_filepath = filepath.rsplit('.', 1)[0] + '_norm.json'
        try:
            with open(norm_filepath, 'r') as f:
                norm = json.load(f)
            model.r_mean = float(norm.get("r_mean", 0.0))
            model.r_std  = float(norm.get("r_std", 1.0))
            model.y_mean = np.array([float(norm.get("y_mean", [0.0, 0.0])[0]), float(norm.get("y_mean", [0.0, 0.0])[1])], dtype=np.float32)
            model.y_std = float(norm.get("y_std", 1.0))
            model.x_mean = np.array(norm.get("x_mean", [0.0, 0.0]), dtype=np.float32)
            model.x_std  = float(norm.get("x_std",  1.0))
            if model.x_std == 0.0:
                model.x_std = 1.0
            if model.r_std == 0.0:
                model.r_std = 1.0
            print(f"Restored r normalisation: mean={model.r_mean:.6g}, std={model.r_std:.6g}")
        except FileNotFoundError:
            print(f"Normalisation file not found ({norm_filepath}); using defaults mean=0, std=1.")
        print(f"Model loaded from weights: {filepath}")
        print(f"  hidden_dim={hidden_dim}, gamma_init={gamma_init}")
        print(f"  Learned gamma value: {get_gamma(model):.6f}")
        return model
    elif save_format in ['savedmodel', 'tf']:
        # Load from SavedModel format
        model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from SavedModel: {filepath}")
        if hasattr(model, 'gamma'):
            print(f"  Learned gamma value: {get_gamma(model):.6f}")
        return model
    else:
        raise ValueError(f"Unknown save_format: {save_format}. Use 'weights' or 'savedmodel'.")


def save_history(history, filepath):
    """
    Save training history to a JSON file.
    
    Parameters:
    -----------
    history : dict
        Training history dictionary (e.g., from train_model).
        Should contain keys like 'train_loss', 'val_loss', etc.
    filepath : str
        Path where to save the history JSON file
    
    Returns:
    --------
    None
    """
    # Convert numpy arrays to lists for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, (np.ndarray, list)):
            # Convert numpy arrays and lists to regular Python lists
            history_serializable[key] = [float(x) for x in value]
        else:
            history_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"Training history saved to: {filepath}")


def load_history(filepath):
    """
    Load training history from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the history JSON file
    
    Returns:
    --------
    dict
        Training history dictionary with keys like 'train_loss', 'val_loss', etc.
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    print(f"Training history loaded from: {filepath}")
    if 'train_loss' in history:
        print(f"  Number of epochs: {len(history['train_loss'])}")
        if len(history['train_loss']) > 0:
            print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    if 'val_loss' in history:
        if len(history['val_loss']) > 0:
            print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    
    return history


def estimate_noise_magnitude(model, test_data):
    """
    Estimate noise magnitude based on model residuals.
    
    Computes residuals between predicted and true accelerations across all test frames,
    then calculates per-dimension variance/std and overall RMS of the residual.
    Only computes statistics on nodes where mask=1.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    test_data : list of dict
        Test dataset, where each dict contains:
        - 'nodes': (N, 4) array of node features [x, y, vx, vy]
        - 'edges': (2, E) array of edge indices
        - 'edge_feat': (E, 3) array of edge features
        - 'targets': (N, 2) array of true accelerations
        - 'mask': (N,) array of mask values (1 = include, 0 = exclude), optional
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'per_dim_var': (2,) array of variance per dimension (ax, ay)
        - 'per_dim_std': (2,) array of std per dimension (ax, ay)
        - 'overall_rms': float, overall RMS of residuals across all dimensions
        - 'num_samples': int, total number of masked particles across all frames
        - 'num_frames': int, number of frames processed
    """
    all_residuals = []
    all_masks = []
    
    # Collect residuals from all frames
    for data_point in test_data:
        # Get predictions
        inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
        Y_pred = model(inputs, training=False)  # (N, 2)
        
        # Get targets
        Y_true = tf.cast(data_point['targets'], tf.float32)  # (N, 2)
        
        # Compute residuals
        resid = Y_true - Y_pred  # (N, 2)
        all_residuals.append(resid.numpy())
        
        # Get mask if present
        if 'mask' in data_point:
            all_masks.append(data_point['mask'])
        else:
            all_masks.append(np.ones(len(resid), dtype=np.float32))
    
    # Stack all residuals and masks (flatten across frames)
    all_residuals = np.vstack(all_residuals)  # (total_particles, 2)
    all_masks = np.concatenate(all_masks)  # (total_particles,)
    
    # Apply masking: only compute statistics on nodes where mask=1
    mask_indices = all_masks == 1.0
    masked_residuals = all_residuals[mask_indices]  # (num_masked_particles, 2)
    
    if len(masked_residuals) == 0:
        print("Warning: No masked nodes found in test data for noise estimation")
        return {
            "per_dim_var": np.array([0.0, 0.0]),
            "per_dim_std": np.array([0.0, 0.0]),
            "overall_rms": 0.0,
            "num_samples": 0,
            "num_frames": len(test_data),
            "mean_residual": np.array([0.0, 0.0]),
        }
    
    # Compute statistics on masked residuals
    num_samples = len(masked_residuals)
    mean_r = np.mean(masked_residuals, axis=0)  # (2,)
    
    # Unbiased variance (using Bessel's correction)
    if num_samples > 1:
        var_r = np.sum((masked_residuals - mean_r)**2, axis=0) / (num_samples - 1)  # (2,)
    else:
        var_r = np.zeros(2)
    std_r = np.sqrt(var_r)  # (2,)
    
    # Overall RMS (across all dimensions and samples)
    overall_rms = np.sqrt(np.mean(np.sum(masked_residuals**2, axis=1)))  # scalar
    sum_residuals_sq_x = np.sum(masked_residuals[:,0]**2)
    sum_residuals_sq_y = np.sum(masked_residuals[:,1]**2)
    
    return {
        "per_dim_var": var_r,
        "per_dim_std": std_r,
        "overall_rms": float(overall_rms),
        "num_samples": num_samples,
        "num_frames": len(test_data),
        "mean_residual": mean_r,  # Bonus: mean residual (should be ~0 if unbiased)
        "sum_residuals_sq_x": sum_residuals_sq_x,
        "sum_residuals_sq_y": sum_residuals_sq_y,
    }

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

    # Prepare columns
    for c in [vx_col, vy_col]:
        if c not in data.columns:
            data[c] = np.nan

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

    # Work per particle
    for pid, dfp in data.sort_values([particle_col, frame_col]).groupby(particle_col, sort=False):
        idx = dfp.index.to_numpy()
        n = len(idx)
        half = tau // 2
        if n < tau + 1:
            continue  # not enough points for any center

        # We’ll slide a centered window; centers from half .. n-1-half
        centers = np.arange(half, n - half, dtype=int)

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
    print("Out of Compute Weak Derivatives: ",data.head())
    return data #if not inplace else None

# ============================================================================
# Convenience Workflow Function
# ============================================================================

def run_full_workflow(datadir, dataname, cutoff=3.0, max_frames=300, 
                      hidden_dim=32, num_epochs=100, learning_rate=0.001,
                      verbose=True):
    """
    Run the complete ForceGNN workflow from data loading to training.
    
    Parameters:
    -----------
    datadir : str
        Directory containing data
    dataname : str
        CSV filename
    cutoff : float
        Graph connectivity cutoff
    max_frames : int
        Maximum frames to use
    hidden_dim : int
        Model hidden dimension
    num_epochs : int
        Training epochs
    learning_rate : float
        Learning rate
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict
        Dictionary containing: model, history, evaluation, train_data, val_data, test_data, data_clean
    """
    if verbose:
        print("=" * 60)
        print("ForceGNN Complete Workflow")
        print("=" * 60)
    
    # Load and preprocess
    data = load_data(datadir, dataname)
    data_clean = calculate_derivatives2(data, dt=0.1)
    dataset = prepare_dataset(data_clean, cutoff=cutoff, max_frames=max_frames)
    train_data, val_data, test_data = split_dataset(dataset)
    
    # Train
    model = create_model(hidden_dim=hidden_dim, gamma_init=gamma_init)
    model, history = train_model(model, train_data, val_data,
                                  num_epochs=num_epochs, 
                                  learning_rate=learning_rate,
                                  verbose=verbose)
    
    # Evaluate
    evaluation = evaluate_model(model, test_data, verbose=verbose)
    
    return {
        'model': model,
        'history': history,
        'evaluation': evaluation,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'data_clean': data_clean
    }


if __name__ == "__main__":
    print(__doc__)
