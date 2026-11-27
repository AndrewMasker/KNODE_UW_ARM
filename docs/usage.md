# Usage

Every script reads a JSON specification from `Library/` and writes its output to `Datasets/`, `Models/` or `Test_Outputs/`. Run everything as a module from the repository root, since all paths are relative to the working directory. Spec names are given without the `.json` extension and relative to the matching `Library/` subfolder.

The pipeline used for the results in the README is:

1. Generate ground-truth trajectories with the fluid on
2. Train the KNODE model against them with the fluid off in the nominal model
3. Build a residual dataset and train the SINDy and GP baselines on it
4. Evaluate each model with the test suite
5. View all of them side by side in MuJoCo

## 1. Generate a dataset

```bash
python3 -m Core.Dataset_Generation dataset_3_fluid_no_coll_specs
python3 -m Core.Dataset_Generation dataset_3_fluid_no_coll_valid_specs   # validation set
```

The spec name is a positional argument, read from `Library/Dataset_Specifications/`. Each sample starts from a random joint configuration, joint velocity and constant torque (drawn uniformly within the joint, `qvel_max` and actuator ranges) and is rolled out with `mjx.step`, vectorized over a batch on the GPU. The result is saved to `Datasets/<name>.zarr`, with one group per trajectory holding `qpos`, `qvel` and `ctrl`, and the spec stored in the root attributes.

Key fields of a dataset spec:

| Field | Meaning |
| --- | --- |
| `model_name` | MuJoCo model in `Library/XML_Models/` (e.g. `mjx_panda`) |
| `num_batches`, `batch_size` | Number of trajectories = `num_batches * batch_size` |
| `step_size`, `num_steps` | Simulation time step and trajectory length |
| `is_gravity`, `is_fluid` | Toggle gravity and MuJoCo's fluid model |
| `wind`, `density`, `viscosity` | Fluid parameters (water: density `1000`) |
| `qvel_max` | Range for the random initial joint velocities; also used to normalize network inputs |

The XML templates contain placeholder comments (`<!-- FLUID_FLAG -->`, `<!-- GRAVITY_FLAG -->`, `<!-- TIMESTEP_FLAG -->`) that `Utils/Mujoco_Utils.py` fills in from the spec, so the same arm can be built with or without the fluid.

## 2. Train the KNODE model

```bash
python3 -m Core.Train_Model_Old -m model_fluid_old -t training_fluid_old
```

- `-m` picks a network spec in `Library/Model_Specifications/` (`width` and `depth` of the MLP).
- `-t` picks a training spec in `Library/Training_Specifications/`.

The training spec names the dataset to train on and describes the *nominal* model. For the fluid experiments it sets `is_fluid: false`, so the network has to learn the difference between the arm in air and the arm in water. Other important fields:

| Field | Meaning |
| --- | --- |
| `dataset_name`, `subset_size` | Dataset to train on, and how many trajectories of it to use |
| `batch_size` | Trajectories per gradient step |
| `lr_strategy`, `num_batches_strategy`, `steps_strategy` | Lists that define training stages: learning rate, number of batches, and the fraction of each trajectory used in each stage |
| `l2`, `l4` | Loss weights on joint position and joint velocity error |
| `should_load`, `load_name` | Continue training from a saved model |
| `should_save`, `should_plot` | Save to `Models/<name>/` and/or show plots at the end |

A trained model is saved to `Models/<name>/` (a numeric suffix is added if the name is taken) as `model.eqx`, `loss_list.json`, `model_options.json` and `training_options.json`.

`Core.Train_Model_UVMS` is the equivalent script for the underwater vehicle-manipulator model (default spec `training_fluid_uvms`).

## 3. Train the baselines

The baselines are trained on the residual between the fluid and no-fluid accelerations, rather than through the ODE solver. First build that residual dataset from an existing trajectory dataset (it is saved as `Datasets/<dataset>_res.zarr`):

```bash
python3 -m Utils.Generate_Residual_Dataset -m model_fluid_old -t training_fluid_old -d dataset_3_fluid_no_coll
```

Then fit the baselines:

```bash
python3 -m Core.Train_Fourier_SINDy -t training_fourier_sindy   # spec in Library/SINDy_Specifications/
python3 -m Core.Train_GP -t gp_fluid                            # spec in Library/GP_Specifications/
```

Check that the spec's dataset field (`sindy_dataset_name` or `residual_dataset_name`) matches the name of the residual dataset you generated. The Fourier SINDy spec includes notes on its sparsity and feature-count settings. Example runs at several dataset sizes are in the `Generalization_Runs/` folders of each spec directory.

## 4. Evaluate with the test suite

```bash
python3 -m Core.Test_Suite_C5 -t test_old
```

A test spec in `Library/Test_Specifications/` names the saved model (`load_name`), its type (`model_type`: `old` for the KNODE model above, `gp`, `fsindy`, `uvms`, `fsindy_uvms`, ...) and a set of jobs to run. Each job writes figures and the underlying data to `Test_Outputs/<save_name>/<job>/`:

| Job `type` | Output |
| --- | --- |
| `statistics` | RMSE and R² per joint and over time, learned model vs. nominal, over a whole dataset |
| `visualize` | Per-trajectory joint comparisons, error over time, error heatmaps and the residual's contribution |
| `phase_portrait` | Joint position vs. velocity, predicted vs. ground truth |
| `error_distribution` | Histograms of prediction error |
| `frequency_analysis` | Frequency content of the prediction error |
| `acc_landscape` | Learned acceleration correction over slices of the state space |
| `loss_surface` | Loss along two directions in parameter space |

`Test_Suite_C3` and `Test_Suite_C4` are earlier versions kept for reference.

## 5. Side-by-side comparison in MuJoCo

```bash
python3 -m Core.Visualize_Comparison -s visualization_spec
```

This loads a KNODE, a GP and a Fourier SINDy model (paths in `Library/Test_Specifications/visualization_spec.json`), runs each one from the same initial state of a validation trajectory, and plays them back in the MuJoCo viewer together with the ground truth and the nominal model, using a model file with five copies of the Panda (`mjx_panda_5_solid`). `Core.Visualize_Comparison_Last_State` holds the final frame instead of looping the playback, which is what the end-point figures in the README show.

## Other utilities

- `python3 -m Tests.Visualize_Mujoco_Model <model_name>`: open one of the models in `Library/XML_Models/` in the viewer.
- `Utils/Show_Loss_Trace.py` and `Utils/Loss_Processing.py`: plot a saved `loss_list.json` (the path is set at the top of `Show_Loss_Trace.py`).

## Known rough edges

- The `run.sh` / `test.sh` helper scripts in `Library/` were written for earlier module names (`Core.Train_Model`, `Core.Test_Suite`). The current equivalents are `Core.Train_Model_Old` and `Core.Test_Suite_C5`; the newer `Train_Model` variant is kept in `Core/could_revisit/`.
- Loading a saved model reads `qvel_max` from the dataset it was trained on, so the dataset must exist locally.
- `Core/old/` and `Core/could_revisit/` hold superseded experiments (plain SINDy, Fourier-feature networks, a power-residual model, etc.) and are not part of the current pipeline.
