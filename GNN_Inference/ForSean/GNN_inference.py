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
from pathlib import Path
from sympy import symbols, lambdify, diff

__all__ = [
    'ForceGNN',
    'load_data',
    'calculate_derivatives',
    'build_graph',
    'prepare_dataset',
    'split_dataset',
    'create_model',
    'train_model',
    'evaluate_model',
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
    - γ * v: Drag/friction term (velocity-dependent, uses node_features[:, 2:], with learnable scalar γ)
    
    Node features: [x, y, vx, vy] - positions and velocities combined
    """
    
    def __init__(self, hidden_dim=32, gamma_init=0.0):
        super(ForceGNN, self).__init__()
        
        self.env_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(2)
        ], name='F_env')
        
        self.interaction_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(hidden_dim, activation='tanh'),
            tf.keras.layers.Dense(1)
        ], name='F_r')
        
        # Learnable drag coefficient γ (initialized to gamma_init)
        self.gamma = tf.Variable(
            initial_value=tf.constant(gamma_init, dtype=tf.float32),
            trainable=True,
            name='gamma',
            dtype=tf.float32
        )
    
    def call(self, inputs, return_components=False):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        inputs : tuple
            (node_features, edge_indices, edge_features)
            - node_features: (N, 4) [x, y, vx, vy] - positions and velocities
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
        velocities = node_features[:, 2:]  # (N, 2)
        
        # Environmental force (only uses positions)
        F_env = self.env_net(positions)
        
        # Pairwise interaction force
        if tf.shape(edge_indices)[1] > 0:
            edge_distances = edge_features[:, 0:1]
            F_r_scalar = self.interaction_net(edge_distances)
            edge_unit_vectors = edge_features[:, 1:3]
            F_r_vector = F_r_scalar * edge_unit_vectors
            
            # Aggregate forces onto target nodes - Billie* Don't understand this step
            target_indices = edge_indices[1, :] #getting the nodes which are the targets of the edges
            indices = tf.expand_dims(target_indices, 1) 
            #F_r_aggregated has the shape of F_env (N,2) and we 
            F_r_aggregated = tf.tensor_scatter_nd_add( 
                tf.zeros_like(F_env),
                indices,
                F_r_vector
            )
        else:
            F_r_aggregated = tf.zeros_like(F_env)
        
        # Drag/friction term: γ * v
        F_drag = self.gamma * velocities
        
        # Total force
        F_total = F_env + F_r_aggregated + F_drag
        
        if return_components:
            return F_total, F_env, F_r_aggregated, F_drag
        return F_total


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
    if 'tx' in data.columns and 'ay' in data.columns:
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
                
                # Acceleration
                data.loc[idx, 'ax'] = (data.loc[idx_next, 'x'] - 2*data.loc[idx, 'x'] + data.loc[idx_prev, 'x']) / (dt**2)
                data.loc[idx, 'ay'] = (data.loc[idx_next, 'y'] - 2*data.loc[idx, 'y'] + data.loc[idx_prev, 'y']) / (dt**2)
    #print("Calculating weak form derivatives...")
    #print("Pre-weak form data: ",data.head())
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
    #print("Post-weak form data: ",data.head())
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
        #data = data[data['ay'] != 0]
    #print("Post-weak and naive form data: ",data.head())
    
    # Remove boundary frames (first and last)
    frames = sorted(data['frame'].unique())
    data_clean = data[(data['frame'] > frames[0]) & (data['frame'] < frames[-1])].copy()
    data_clean = data.copy()
    print("length of data_clean: ", len(data_clean))
    print("how much of the data has zero ax and ay: ", len(data_clean[data_clean['ax'] == 0])/len(data_clean))
    print(f"After removing boundary frames: {data_clean.shape}")
    print(f"Acceleration statistics:")
    print(f"  ax: mean={data_clean['ax'].mean():.3f}, std={data_clean['ax'].std():.3f}")
    print(f"  ay: mean={data_clean['ay'].mean():.3f}, std={data_clean['ay'].std():.3f}")
    
    return data_clean


def build_graph(frame_data, cutoff=3.0):
    """
    Build graph structure from particle data for one frame.
    
    Parameters:
    -----------
    frame_data : pd.DataFrame
        Single frame data with columns 'x', 'y', 'vx', 'vy'
    cutoff : float
        Maximum distance for edge connections
    
    Returns:
    --------
    tuple
        (nodes, edges, edge_features)
        - nodes: (N, 4) array of [x, y, vx, vy]
        - edges: (2, E) array of edge indices [source, target]
        - edge_features: (E, 3) array of [distance, dx/r, dy/r]
    """
    N = len(frame_data)
    # Node features: positions and velocities [x, y, vx, vy]
    nodes = frame_data[['x', 'y', 'vx', 'vy']].values
    edges = []
    edge_features = []
    
    # Build edges based on positions only (for distance calculations)
    for i in range(N):
        xi, yi = frame_data.iloc[i][['x', 'y']]
        for j in range(N):
            if i == j:
                continue
            xj, yj = frame_data.iloc[j][['x', 'y']]
            dx, dy = xj - xi, yj - yi
            r = np.sqrt(dx**2 + dy**2)
            if r < cutoff:
                edges.append([i, j])
                edge_features.append([r, dx/r, dy/r])
    
    return (
        nodes.astype(np.float32),
        np.array(edges, dtype=np.int32).T if edges else np.zeros((2, 0), dtype=np.int32),
        np.array(edge_features, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32)
    )


def prepare_dataset(data_clean, cutoff=3.0, max_frames=None):
    """
    Prepare training dataset from all frames.
    
    Parameters:
    -----------
    data_clean : pd.DataFrame
        Cleaned data with derivatives (must have columns: x, y, vx, vy, ax, ay)
    cutoff : float
        Graph connectivity cutoff distance
    max_frames : int, optional
        Maximum number of frames to use (None = use all)
    
    Returns:
    --------
    list of dict
        Each dict contains: frame, nodes (N,4) [x,y,vx,vy], edges, edge_feat, targets
    """
    frames = sorted(data_clean['frame'].unique())
    if max_frames:
        frames = frames[:max_frames]
    
    dataset = []
    print(f"Preparing dataset from {len(frames)} frames...")
    
    for frame_id in frames:
        print("processing frame ", frame_id)
        frame_data = data_clean[data_clean['frame'] == frame_id].reset_index(drop=True)
        
        # Build graph
        nodes, edges, edge_feat = build_graph(frame_data, cutoff)
        
        # Targets (accelerations)
        targets = frame_data[['ax', 'ay']].values.astype(np.float32)
        
        # Billie: this will be where I will have fluctuating particles
        # Note: nodes already contain [x, y, vx, vy] from build_graph
        dataset.append({
            'frame': frame_id,
            'nodes': nodes,
            'edges': edges,
            'edge_feat': edge_feat,
            'targets': targets
        })
    
    print(f"Dataset prepared: {len(dataset)} frames")
    print(f"Example - Nodes: {dataset[0]['nodes'].shape}, Edges: {dataset[0]['edges'].shape}")
    
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
        (train_data, val_data, test_data)
    """
    train_size = int(train_frac * len(dataset))
    val_size = int(val_frac * len(dataset))

    # I should maybe scramble the dataset before splitting it
    np.random.shuffle(dataset)
    
    #train_data = dataset[:train_size]
    #val_data = dataset[train_size:train_size+val_size]
    val_data = dataset[:val_size]
    train_data = dataset[val_size:val_size+train_size]
    test_data = dataset[train_size+val_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    return train_data, val_data, test_data


# ============================================================================
# Model Training
# ============================================================================

def create_model(hidden_dim=32, gamma_init=0.0):
    """
    Create a ForceGNN model.
    
    Parameters:
    -----------
    hidden_dim : int
        Hidden dimension for neural network layers
    gamma_init : float
        Initial value for drag coefficient γ (default 0.0)
    
    Returns:
    --------
    ForceGNN
        Initialized model
    """
    model = ForceGNN(hidden_dim=hidden_dim, gamma_init=gamma_init)
    print(f"ForceGNN model created with hidden_dim={hidden_dim}, gamma_init={gamma_init}")
    return model


def train_model(model, train_data, val_data, num_epochs=100, learning_rate=0.001, verbose=True):
    """
    Train the ForceGNN model.
    
    Parameters:
    -----------
    model : ForceGNN
        Model to train
    train_data : list
        Training dataset
    val_data : list
        Validation dataset
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    train_losses = []
    val_losses = []
    
    if verbose:
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []
        for data_point in train_data:
            with tf.GradientTape() as tape:
                inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
                predictions = model(inputs)
                loss = tf.reduce_mean((predictions - data_point['targets'])**2)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_losses.append(loss.numpy())
        
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation
        val_epoch_losses = []
        for data_point in val_data:
            inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
            predictions = model(inputs)
            loss = tf.reduce_mean((predictions - data_point['targets'])**2)
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
    """
    Evaluate model on test data.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    test_data : list
        Test dataset
    verbose : bool
        Print results
    
    Returns:
    --------
    dict
        Dictionary with 'mse', 'mae', 'predictions', 'targets'
    """
    test_predictions = []
    test_targets = []
    
    for data_point in test_data:
        inputs = (data_point['nodes'], data_point['edges'], data_point['edge_feat'])
        predictions = model(inputs).numpy()
        targets = data_point['targets']
        
        test_predictions.append(predictions)
        test_targets.append(targets)
    
    # Flatten
    test_predictions_flat = np.vstack(test_predictions)
    test_targets_flat = np.vstack(test_targets)
    
    test_mse = np.mean((test_predictions_flat - test_targets_flat)**2)
    test_mae = np.mean(np.abs(test_predictions_flat - test_targets_flat))
    
    if verbose:
        print(f"\nTest Set Evaluation:")
        print(f"  MSE: {test_mse:.6f}")
        print(f"  MAE: {test_mae:.6f}")
    
    return {
        'mse': test_mse,
        'mae': test_mae,
        'predictions': test_predictions_flat,
        'targets': test_targets_flat
    }


# ============================================================================
# Force Query Functions
# ============================================================================

def get_force_at_distance(model, r):
    """
    Get the learned pairwise force magnitude at a given distance.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    r : float
        Distance value
    
    Returns:
    --------
    float
        Force magnitude F_r(r)
    """
    return model.interaction_net(tf.constant([[r]], dtype=tf.float32)).numpy()[0, 0]


def get_force_function(model, r_min=0.5, r_max=3.0, num_points=100):
    """
    Get the learned force function over a range of distances.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    r_min : float
        Minimum distance
    r_max : float
        Maximum distance
    num_points : int
        Number of points to sample
    
    Returns:
    --------
    tuple
        (r_values, F_r_values) arrays
    """
    r_values = np.linspace(r_min, r_max, num_points).astype(np.float32)[:, None]
    F_r_values = model.interaction_net(r_values).numpy()
    return r_values.flatten(), F_r_values.flatten()

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


def get_force_environment(model, x, y):
    """
    Get the learned environmental force at a given position.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    x : float
        x-coordinate
    y : float
        y-coordinate
    
    Returns:
    --------
    np.ndarray
        Environmental force vector [F_env_x, F_env_y]
    """
    return model.env_net(tf.constant([[x, y]], dtype=tf.float32)).numpy()[0, :]


def estimate_noise_magnitude(model, test_data):
    """
    Estimate noise magnitude based on model residuals.
    
    Computes residuals between predicted and true accelerations across all test frames,
    then calculates per-dimension variance/std and overall RMS of the residual.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    test_data : list of dict
        Test dataset, where each dict contains:
        - 'nodes': (N, 2) array of positions
        - 'edges': (2, E) array of edge indices
        - 'edge_feat': (E, 3) array of edge features
        - 'targets': (N, 2) array of true accelerations
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'per_dim_var': (2,) array of variance per dimension (ax, ay)
        - 'per_dim_std': (2,) array of std per dimension (ax, ay)
        - 'overall_rms': float, overall RMS of residuals across all dimensions
        - 'num_samples': int, total number of particles across all frames
        - 'num_frames': int, number of frames processed
    """
    all_residuals = []
    
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
    
    # Stack all residuals (flatten across frames)
    all_residuals = np.vstack(all_residuals)  # (total_particles, 2)
    
    # Compute statistics
    num_samples = all_residuals.shape[0]
    mean_r = np.mean(all_residuals, axis=0)  # (2,)
    
    # Unbiased variance (using Bessel's correction)
    var_r = np.sum((all_residuals - mean_r)**2, axis=0) / (num_samples - 1)  # (2,)
    std_r = np.sqrt(var_r)  # (2,)
    
    # Overall RMS (across all dimensions and samples)
    overall_rms = np.sqrt(np.mean(np.sum(all_residuals**2, axis=1)))  # scalar
    
    return {
        "per_dim_var": var_r,
        "per_dim_std": std_r,
        "overall_rms": float(overall_rms),
        "num_samples": num_samples,
        "num_frames": len(test_data),
        "mean_residual": mean_r,  # Bonus: mean residual (should be ~0 if unbiased)
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

# ============================================================================
# Convenience Workflow Function
# ============================================================================

def run_full_workflow(datadir, dataname, cutoff=3.0, max_frames=300, 
                      hidden_dim=32, num_epochs=100, learning_rate=0.001,
                      gamma_init=0.0, verbose=True):
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
    gamma_init : float
        Initial value for drag coefficient γ (default 0.0)
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
    data_clean = calculate_derivatives(data, dt=0.1)
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
