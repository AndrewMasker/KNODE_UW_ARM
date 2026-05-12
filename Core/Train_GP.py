"""
Train_GP.py
===========
Fitting script for the Gaussian Process residual model.
Mirrors Train_Fourier_SINDy.py: load residual dataset → sub-sample → fit GPs.
"""

import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import zarr

from Core.GP_Modules import (
    GP_Residual,
    KNODE_GP,
    fit_GP,
)
import Utils.Mujoco_Utils as mj_utils


# ============================================================================
# ARGUMENT / SPEC PARSING
# ============================================================================

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_specs", type=str,
                        default="training_gp")
    args = parser.parse_args()
    return "Library/GP_Specifications/" + args.training_specs + ".json"


def parse_Specs(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================================
# DATASET LOADING  (same structure as Train_Fourier_SINDy.py)
# ============================================================================

def load_Residual_Dataset(training_options: dict):
    dataset_path = "Datasets/" + training_options["sindy_dataset_name"] + ".zarr"
    root = zarr.open(dataset_path, mode="r")

    keys = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))

    subset_size = training_options.get("subset_size", None)
    if subset_size is None:
        subset_size = len(keys)
    elif subset_size > len(keys):
        raise ValueError("subset_size cannot exceed dataset size.")

    qpos_list, qvel_list, ctrl_list, res_list = [], [], [], []
    for i in range(subset_size):
        d = root[keys[i]]
        qpos_traj = d["qpos"][:]
        qvel_traj = d["qvel"][:]
        ctrl_raw  = d["ctrl"][:]
        ctrl_traj = (np.tile(ctrl_raw[None, :], (qpos_traj.shape[0], 1))
                     if ctrl_raw.ndim == 1 else ctrl_raw)
        res_traj  = d["res_qacc"][:]

        qpos_list.append(qpos_traj)
        qvel_list.append(qvel_traj)
        ctrl_list.append(ctrl_traj)
        res_list.append(res_traj)

        if i % 10 == 0:
            print(f"  Trajectory {i}/{subset_size} loaded.")

    qpos_arr = np.concatenate(qpos_list, axis=0).astype(np.float64)
    qvel_arr = np.concatenate(qvel_list, axis=0).astype(np.float64)
    ctrl_arr = np.concatenate(ctrl_list, axis=0).astype(np.float64)
    res_arr  = np.concatenate(res_list,  axis=0).astype(np.float64)

    print(f"Loaded {qpos_arr.shape[0]} samples from {subset_size} trajectories.")

    max_n = training_options.get("max_samples", None)
    if max_n is not None and max_n < qpos_arr.shape[0]:
        rng = np.random.default_rng(seed=42)
        idx = np.sort(rng.choice(qpos_arr.shape[0], size=int(max_n), replace=False))
        qpos_arr, qvel_arr, ctrl_arr, res_arr = (
            qpos_arr[idx], qvel_arr[idx], ctrl_arr[idx], res_arr[idx]
        )
        print(f"  Sub-sampled to {qpos_arr.shape[0]} samples.")

    return qpos_arr, qvel_arr, ctrl_arr, res_arr


def get_qvel_max(training_options: dict) -> jax.Array:
    dataset_path = "Datasets/" + training_options["sindy_dataset_name"] + ".zarr"
    root = zarr.open(dataset_path, mode="r")
    return jnp.array(root.attrs["dataset_options"]["qvel_max"])


# ============================================================================
# SAVING / LOADING
# ============================================================================

def _unique_save_dir(base_name: str) -> str:
    path = os.path.join(os.getcwd(), "Models", base_name)
    if not os.path.exists(path):
        return path
    counter = 1
    while os.path.exists(f"{path}_{counter}"):
        counter += 1
    return f"{path}_{counter}"


def save_Model(residual: GP_Residual,
               full_model: KNODE_GP,
               training_options: dict) -> str:
    save_path = _unique_save_dir(training_options.get("name", "gp_model"))
    os.makedirs(save_path)

    eqx.tree_serialise_leaves(os.path.join(save_path, "gp_residual.eqx"), residual)
    eqx.tree_serialise_leaves(os.path.join(save_path, "model.eqx"), full_model)

    with open(os.path.join(save_path, "training_options.json"), "w") as f:
        json.dump(training_options, f, indent=2)

    # Save GP data as numpy for easy inspection
    np.save(os.path.join(save_path, "X_train_norm.npy"),
            np.array(residual.X_train_norm))
    np.save(os.path.join(save_path, "alpha.npy"),
            np.array(residual.alpha))
    np.save(os.path.join(save_path, "log_lengthscales.npy"),
            np.array(residual.log_lengthscales))
    np.save(os.path.join(save_path, "log_signal_var.npy"),
            np.array(residual.log_signal_var))
    np.save(os.path.join(save_path, "log_noise_var.npy"),
            np.array(residual.log_noise_var))

    print(f"\nModel saved to: {save_path}")
    return save_path


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(residual: GP_Residual,
             qpos_arr, qvel_arr, ctrl_arr, res_arr) -> dict:
    X    = jnp.concatenate([jnp.array(qpos_arr),
                             jnp.array(qvel_arr),
                             jnp.array(ctrl_arr)], axis=-1)
    pred = np.array(jax.vmap(residual)(X))
    tgt  = np.array(res_arr)
    mse  = np.mean((pred - tgt) ** 2)
    r2   = 1.0 - np.sum((pred - tgt) ** 2) / (np.sum((tgt - tgt.mean(0)) ** 2) + 1e-12)
    per_dof_rmse = np.sqrt(np.mean((pred - tgt) ** 2, axis=0))

    print(f"\n--- Evaluation ---")
    print(f"  MSE  = {mse:.6e}")
    print(f"  RMSE = {np.sqrt(mse):.6e}")
    print(f"  R²   = {r2:.4f}")
    print(f"  Per-DOF RMSE: {np.round(per_dof_rmse, 6)}")
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)),
            "r2": float(r2), "per_dof_rmse": per_dof_rmse.tolist()}


# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_Trained_Model(model: KNODE_GP, training_options: dict):
    root = zarr.open(
        os.getcwd() + "/Datasets/" + training_options["sindy_dataset_name"] + ".zarr",
        mode="r"
    )
    dt         = root.attrs["dataset_options"]["step_size"]
    xml        = root.attrs["xml"]
    _, mjx_data = mj_utils.init_mjx_model(xml)

    keys      = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    d         = root[keys[training_options["which_to_show_at_end"]]]
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts        = jnp.arange(0, step_size * num_steps, step=step_size)

    init_states = {"qpos": jnp.array(d["qpos"][0, :]),
                   "qvel": jnp.array(d["qvel"][0, :])}
    ctrl        = jnp.array(d["ctrl"][0, :])

    learned_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
    nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))

    l2 = training_options.get("l2", 1.0)
    l4 = training_options.get("l4", 1.0)
    print(f"Learned loss : {l2*jnp.mean((learned_traj['qpos'] - d['qpos'][:])**2) + l4*jnp.mean((learned_traj['qvel'] - d['qvel'][:])**2):.6e}")
    print(f"Nominal loss : {l2*jnp.mean((nominal_traj['qpos'] - d['qpos'][:])**2) + l4*jnp.mean((nominal_traj['qvel'] - d['qvel'][:])**2):.6e}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(learned_traj["qpos"], color="green")
    axes[0].plot(nominal_traj["qpos"], color="red")
    axes[0].plot(d["qpos"][:],         color="black")
    axes[0].set_title("qpos")

    axes[1].plot(learned_traj["qvel"], color="green")
    axes[1].plot(nominal_traj["qvel"], color="red")
    axes[1].plot(d["qvel"][:],         color="black")
    axes[1].set_title("qvel")

    learned_acc = (learned_traj["qvel"][1:] - learned_traj["qvel"][:-1]) / dt
    nominal_acc = (nominal_traj["qvel"][1:] - nominal_traj["qvel"][:-1]) / dt
    gt_acc      = (d["qvel"][1:] - d["qvel"][:-1]) / dt

    axes[2].plot(learned_acc, color="green")
    axes[2].plot(nominal_acc, color="red")
    axes[2].plot(gt_acc,      color="black")
    axes[2].set_title("qacc")

    axes[3].plot(learned_acc - nominal_acc, color="green")
    axes[3].plot(gt_acc - nominal_acc,      color="black")
    axes[3].set_title("residual qacc")

    plt.tight_layout()
    plt.show(block=False)

    gt_qpos      = np.array(d["qpos"][:])
    nominal_qpos = np.array(nominal_traj["qpos"])
    learned_qpos = np.array(learned_traj["qpos"])
    mj_utils.visualize_GT_vs_Nominal_vs_KNODE_Trajectory(
        training_options["model_name"], gt_qpos, nominal_qpos, learned_qpos
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    training_path    = return_Arguments()
    training_options = parse_Specs(training_path)

    model_xml           = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data = mj_utils.init_mjx_model(model_xml)

    # ---- Load residual dataset ----
    t0 = time.time()
    qpos_arr, qvel_arr, ctrl_arr, res_arr = load_Residual_Dataset(training_options)
    print(f"  Dataset loaded in {time.time()-t0:.2f}s")

    qvel_max = get_qvel_max(training_options)

    # ---- Fit GPs ----
    key = jr.key(int(time.time()))
    t0 = time.time()
    gp_residual = fit_GP(
        mjx_model          = mjx_model,
        qvel_max           = qvel_max,
        qpos_arr           = qpos_arr,
        qvel_arr           = qvel_arr,
        ctrl_arr           = ctrl_arr,
        residual_qacc      = res_arr,
        num_training_points = training_options.get("num_training_points", 2000),
        lr                 = training_options.get("gp_lr",        0.01),
        num_iters          = training_options.get("gp_num_iters", 500),
        jitter             = training_options.get("gp_jitter",    1e-6),
        key                = key,
        verbose            = True,
    )
    print(f"  Fitting completed in {time.time()-t0:.2f}s")

    report = gp_residual.sparsity_report()
    print(f"\nGP model: {report['n_training_points']} training points, "
          f"{report['output_num']} outputs, {report['input_dim']}D input")

    # ---- Evaluate on full dataset ----
    if training_options.get("should_show", False):
        evaluate(gp_residual, qpos_arr, qvel_arr, ctrl_arr, res_arr)

    # ---- Build full KNODE model ----
    full_model = KNODE_GP(mjx_model, gp_residual)

    # ---- Save ----
    if training_options.get("should_save", False):
        save_Model(gp_residual, full_model, training_options)

    # ---- Visualise trajectory rollout ----
    if training_options.get("should_plot", False):
        visualize_Trained_Model(full_model, training_options)
        plt.show()
