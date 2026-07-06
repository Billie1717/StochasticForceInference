# Force Inference with Graph Neural Networks

This repo infers pairwise and environmental forces from particle/cell trajectory data using a graph neural network (GNN), implemented in PyTorch. It's an adaptation of Billie's original TensorFlow force-inference code.

There are two examples, going from simple to complex:

- **`UD_Inference/`** — underdamped simulation data. Ground-truth forces are known, so this is the place to learn the pipeline and sanity-check that the model recovers the correct forces.
- **`Experiment_Inference/`** — real experimental trajectories (neutrophils), which are overdamped/persistent rather than underdamped, noisier, and have variable numbers of cells per frame. Ground truth is unknown here — the goal is to infer forces, not recover known ones.

Start with `UD_Inference`, then move to `Experiment_Inference` once the pipeline makes sense.

## Setup

```
pip install torch numpy pandas sympy tqdm matplotlib seaborn scipy functorch
```

Both notebooks use `importlib.reload` on the local module, so you can edit the `.py` file and re-run cells without restarting the kernel.

## Repo structure

```
UD_Inference/
  torch_inference.py            module: data loading, derivatives, graph building, dataset prep
  torch_notebook_example.ipynb  full pipeline: load -> derivatives -> graph -> train -> evaluate -> query forces
  Data_CSVs/                    simulated underdamped trajectories

Experiment_Inference/
  torch_inference_FintDot.py    module (extended version, see "Two modules" below)
  Fish_Vis.ipynb                Step 1: visualise raw tracks, build one-csv-per-experiment dataframes
  torch_notebook_Fish.ipynb     Step 2: run inference on the cleaned csv from Fish_Vis
  Data_CSVs/                    experimental trajectory csvs (per-condition master track files)
```

## UD_Inference walkthrough

Run `torch_notebook_example.ipynb` top to bottom:

1. **Load and prepare data** — CSV columns are `frame, particle, x, y, z, mass`; simulation has 300 frames, with either 15 or 20 particles (see the two files in `Data_CSVs/`).
2. **Calculate derivatives** — finite-difference velocities/accelerations per particle (`calculate_derivatives`).
3. **Visualize trajectories** — sanity check the tracks look sensible before building graphs.
4. **Build dataset as graph** (`prepare_dataset` / `build_graph`) — each frame becomes a graph: nodes are `[x, y, vx, vy]`, edges connect particles within `cutoff`, edge features are `[distance, unit dx, unit dy]`.
5. **Split dataset** — train/val/test split over frames.
6. **Create and train model** — a small GNN (`ForceGNN`) predicts acceleration from node + edge features.
7. **Compare to true forces** — because this is simulated data, the true pairwise (Lennard-Jones-like) and environmental forces are known and plotted against the model's predictions.
8. **Evaluate model, query forces** — R² on held-out data, then query the learned pairwise force curve and environmental force field directly from the trained model.

## Experiment_Inference walkthrough

Two notebooks, run in order:

1. **`Fish_Vis.ipynb`** — loads the raw `master_tracks_*.csv`, visualises trajectories, and reshapes the data into the per-experiment CSV format the inference pipeline expects (one file per condition/experiment, standardised columns).
2. **`torch_notebook_Fish.ipynb`** — loads that cleaned CSV and runs the same load → derivatives → graph → train → evaluate → query-forces pipeline as `UD_Inference`, adapted for overdamped/persistent dynamics (no ground-truth forces to compare against, and it additionally uses the pairwise **velocity difference** as an edge feature, since persistence/alignment matters more here than in the underdamped case).

### Gotchas specific to this dataset

- **Derivative method**: use the naive (finite-difference) derivative for now (`naive_form=True`). The weak-form derivative is under active development and does not yet recover the correct timescale (`tau`) — don't trust `tau` from weak-form results yet.
- **Neighbour lists**: `prepare_dataset` builds a voxel-based neighbour list (`VoxelNeighborList`) to avoid O(N²) distance checks across frames. Double check `n_boxes`, `neighbor_rebuild_interval`, and `cutoff` are sensible for your particle density — a cutoff much larger than the voxel size defeats the point of the neighbour list.
- **Masking**: nodes are masked out of training/loss (but still appear as neighbours to other nodes) if they're missing derivative information (e.g. edge-of-trajectory frames), or — in `torch_inference_FintDot.py` only — if they're within 20 units of the domain boundary, since forces there may be driven by things outside the field of view.
- **Variable particle counts**: unlike the fixed-N underdamped simulation, the number of cells per frame varies, so check the "particles per frame" plot in `Fish_Vis.ipynb` before training — very sparse frames can skew normalisation statistics.

## Two modules: `torch_inference.py` vs `torch_inference_FintDot.py`

These started as the same file and have since diverged — they are **not** interchangeable, and `Experiment_Inference` should only import `torch_inference_FintDot`. Differences worth knowing about:

- `torch_inference_FintDot.py` adds a velocity-difference edge feature (`vel_edge_feat`: `[|dv|, dvx/|dv|, dvy/|dv|]`) alongside the position edge feature — `build_graph` and everything downstream of it therefore returns/expects one extra array.
- `torch_inference_FintDot.py` adds `compute_backward_weak_velocity` (a backward-windowed weak-form velocity, used to break time-symmetry issues in the original centred weak-form estimator) — not present in `torch_inference.py`.
- Boundary masking (excluding nodes within 20 units of the domain edge) is only applied in `torch_inference_FintDot.py`'s `build_graph`.
- Edge-distance normalisation differs slightly: `torch_inference.py` centres distances (`(r - r_mean) / r_std`) before scaling; `torch_inference_FintDot.py` only scales (`r / r_std`), no centring.

If you find yourself wanting a fix in one module, check whether it should go in the other too.

## Notation

- `frame`: timestep index; `particle`: track/particle ID.
- `x, y`: positions; `vx, vy`: velocities; `ax, ay`: accelerations (model targets).
- `mask`: 1 = node included in the training loss, 0 = node present as a neighbour only.
- `nn_r`: nearest-neighbour distance per node (useful for checking density/crowding).
