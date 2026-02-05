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
]

# ============================================================================
# Model Definition
# ============================================================================

class ForceGNN(tf.keras.Model):
    """
    Graph Neural Network for learning force decomposition.
    
    Forces are decomposed into:
    - F_env: Environmental/external forces (position-dependent)
    - F_r: Pairwise interaction forces (distance-dependent)
    """
    
    def __init__(self, hidden_dim=32):
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
    
    def call(self, inputs, return_components=False):
        node_features, edge_indices, edge_features = inputs
        
        # Environmental force
        F_env = self.env_net(node_features)
        
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
        
        F_total = F_env + F_r_aggregated
        
        if return_components:
            return F_total, F_env, F_r_aggregated
        return F_total


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

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


def calculate_derivatives(data, dt=0.1):
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
    if 'id' in data.columns:
        particles = data['id'].unique()
    else:
        # Create IDs if not present
        num_frames = len(data['frame'].unique())
        num_particles = len(data) // num_frames
        particles = range(num_particles)
        data['id'] = np.tile(particles, num_frames)
    
    print(f"Processing {len(particles)} particles...")
    
    # Calculate derivatives for each particle using central differences
    for pid in particles:
        particle_mask = data['id'] == pid
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
    
    # Remove boundary frames (first and last)
    frames = sorted(data['frame'].unique())
    data_clean = data[(data['frame'] > frames[0]) & (data['frame'] < frames[-1])].copy()
    
    print(f"After removing boundary frames: {data_clean.shape}")
    print(f"Acceleration statistics:")
    print(f"  ax: mean={data_clean['ax'].mean():.3f}, std={data_clean['ax'].std():.3f}")
    print(f"  ay: mean={data_clean['ay'].mean():.3f}, std={data_clean['ay'].std():.3f}")
    
    return data_clean


def build_graph(frame_data, cutoff=3.0):
    """
    Build graph structure from particle positions for one frame.
    
    Parameters:
    -----------
    frame_data : pd.DataFrame
        Single frame data with columns 'x', 'y'
    cutoff : float
        Maximum distance for edge connections
    
    Returns:
    --------
    tuple
        (nodes, edges, edge_features)
        - nodes: (N, 2) array of positions
        - edges: (2, E) array of edge indices [source, target]
        - edge_features: (E, 3) array of [distance, dx/r, dy/r]
    """
    N = len(frame_data)
    nodes = frame_data[['x', 'y']].values
    edges = []
    edge_features = []
    
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
        Cleaned data with derivatives
    cutoff : float
        Graph connectivity cutoff distance
    max_frames : int, optional
        Maximum number of frames to use (None = use all)
    
    Returns:
    --------
    list of dict
        Each dict contains: frame, nodes, edges, edge_feat, targets
    """
    frames = sorted(data_clean['frame'].unique())
    if max_frames:
        frames = frames[:max_frames]
    
    dataset = []
    print(f"Preparing dataset from {len(frames)} frames...")
    
    for frame_id in frames:
        frame_data = data_clean[data_clean['frame'] == frame_id].reset_index(drop=True)
        
        # Build graph
        nodes, edges, edge_feat = build_graph(frame_data, cutoff)
        
        # Targets (accelerations)
        targets = frame_data[['ax', 'ay']].values.astype(np.float32)
        # Billie: this will be where I will have fluctuating particles
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
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size+val_size]
    test_data = dataset[train_size+val_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")
    
    return train_data, val_data, test_data


# ============================================================================
# Model Training
# ============================================================================

def create_model(hidden_dim=32):
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
    model = ForceGNN(hidden_dim=hidden_dim)
    print(f"ForceGNN model created with hidden_dim={hidden_dim}")
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

def get_force_environment(model, x, y):
    """
    Get the learned environmental force at a given position.
    
    Parameters:
    -----------
    model : ForceGNN
        Trained model
    x : float
    """
    return model.env_net(tf.constant([[x, y]], dtype=tf.float32)).numpy()[0, :]



if __name__ == "__main__":
    print(__doc__)
