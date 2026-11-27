# Code overview

A map of the files that make up the current pipeline. See [usage.md](usage.md) for how to run them.

## Models

All three models share the same structure: a nominal MJX model of the arm without the fluid, plus a learned residual on the joint accelerations, integrated over time with Diffrax. They differ only in what the residual is and how it is fit.

| File | Model |
| --- | --- |
| `Core/KNODE_Modules_Old.py` | **KNODE** for the fixed-base Panda. `Residual_Network` is an Equinox MLP (GELU) with fixed input/output normalization; `KNODE_Dynamics` adds `res_mixing_matrix @ f_θ(q, q̇, τ)` to the MJX acceleration; `KNODE` integrates it with `diffrax.diffeqsolve` (Euler, same step as the data). Passing `with_learned=0` turns the residual off, which gives the nominal model. |
| `Core/KNODE_Modules_UVMS.py` | The same model for the vehicle-manipulator system. Position derivatives include a quaternion derivative for the vehicle's orientation, and integration uses a small `ManifoldEuler` solver wrapper. |
| `Core/Fourier_SINDy_Modules.py` | **Fourier-basis SINDy** baseline. The library is a fixed set of random Fourier features, `[sin(Wx), cos(Wx)]`, of the normalized state and input; the sparse coefficients are fit with sequentially thresholded least squares (`stlsq`) or `sr3`. Wrapped in `KNODE_Fourier_SINDy` so it can be rolled out exactly like KNODE. (`_UVMS` version for the vehicle-manipulator.) |
| `Core/GP_Modules.py` | **Gaussian process** baseline: an independent GP with an ARD RBF kernel per degree of freedom, hyperparameters fit by maximizing the marginal likelihood, wrapped in `KNODE_GP` for roll-outs. |

## Training

| File | What it does |
| --- | --- |
| `Core/Train_Model_Old.py` | Trains KNODE end to end through the ODE solver. Loads batches of trajectories from Zarr with Grain, rolls out the model from each initial state, and minimizes `l2·MSE(q) + l4·MSE(q̇)` with Adam. Only the MLP and the mixing matrix are trainable (selected with an Equinox filter spec); the MJX model stays fixed. |
| `Core/Train_Model_UVMS.py` | Same, for the vehicle-manipulator model. |
| `Core/Train_Fourier_SINDy.py`, `Core/Train_GP.py` | Fit the baselines by regression on a residual dataset (target: ground-truth acceleration minus nominal acceleration). |
| `Utils/Generate_Residual_Dataset.py` | Builds that residual dataset from a trajectory dataset by evaluating the nominal and fluid MJX models at every sample. |

## Data and simulation

| File | What it does |
| --- | --- |
| `Core/Dataset_Generation.py` | Random initial states and constant torques, rolled out with `mjx.step` under `jax.vmap`, saved to Zarr. (`_UVMS` version for the vehicle-manipulator.) |
| `Utils/Mujoco_Utils.py` | Builds a MuJoCo XML from a model template plus the gravity/fluid/time-step options in a spec, creates MuJoCo and MJX models, and has viewer helpers. Contacts and joint limits are disabled in the MJX model. |
| `Utils/MJX_Diff.py` | A copy of MJX's forward pass that can be differentiated in reverse mode. With contacts disabled it skips the constraint solver and returns the unconstrained acceleration (`qacc_smooth`). |
| `Utils/MJX_Solver_Diff.py` | A copy of MJX's constraint solver with its `while_loop`s replaced by fixed-length scans, for reverse-mode gradients when contacts are enabled (currently switched off in `MJX_Diff.forward`). |
| `Utils/Data_Utils.py` | Conversions between Python dicts and Zarr attributes, and the batch collate function. |
| `Library/XML_Models/` | MuJoCo models: the Panda (`mjx_panda*`, including the 5-arm comparison scenes), the BlueROV2 vehicle-manipulator (`bluerov2*`), and the XML snippets used to switch gravity and the fluid on or off. |

## Evaluation

| File | What it does |
| --- | --- |
| `Core/Test_Suite_C5.py` | Loads any saved model type and runs the jobs listed in a test spec (statistics, per-trajectory plots, phase portraits, error distributions, frequency analysis, acceleration landscapes, loss surfaces). `C3` and `C4` are earlier versions. |
| `Core/Visualize_Comparison.py` | Plays back KNODE, GP, SINDy, nominal and ground truth together in the MuJoCo viewer. `_Last_State` holds the final frame. |

## Specification files

Every run is described by a JSON file in `Library/`, and the options used are copied next to the saved output, so each model in `Models/` and each result in `Test_Outputs/` records how it was made. Some spec files contain extra keys beginning with `_` or with free-text names; these are notes and are ignored by the code.

| Folder | Used by |
| --- | --- |
| `Dataset_Specifications/` | `Dataset_Generation*` |
| `Model_Specifications/` | `Train_Model*` (`-m`) |
| `Training_Specifications/` | `Train_Model*` (`-t`) |
| `SINDy_Specifications/` | `Train_Fourier_SINDy*` |
| `GP_Specifications/` | `Train_GP` |
| `Test_Specifications/` | `Test_Suite_C5`, `Visualize_Comparison*` |
