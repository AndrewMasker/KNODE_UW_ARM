"""
Train_SINDy.py
==============
Fitting script for the JAX/Equinox SINDy model.
Mirrors the structure of Train_Model_Old.py so the two are easy to compare.

Expected zarr dataset layout
-----------------------------
Datasets/<sindy_dataset_name>.zarr
  attrs:
    dataset_options:
      step_size   : float
      num_steps   : int
      qvel_max    : List[float]   # per-joint velocity normalisation
    xml           : str           # MuJoCo XML string
  trajectory_0/
    qpos          : (T, nq)
    qvel          : (T, nv)
    ctrl          : (T, nu)  OR  (nu,) broadcast
    residual_qacc : (T, nq)   ← actual_qacc − nominal_mjx_qacc

Training JSON spec example (Library/Training_Specifications/training_sindy.json)
----------------------------------------------------------------------------------
{
  "name"             : "sindy_fluid",
  "model_name"       : "your_robot",
  "sindy_dataset_name": "residual_dataset",
  "poly_degree"      : 2,
  "n_frequencies"    : 1,
  "include_bias"     : true,
  "include_sin"      : true,
  "include_cos"      : true,
  "sindy_optimizer"  : "stlsq",
  "sindy_threshold"  : 0.05,
  "sindy_alpha"      : 1e-5,
  "sindy_nu"         : 1.0,
  "sindy_max_iter"   : 20,
  "max_samples"      : null,
  "should_save"      : true,
  "which_to_show_at_end": 0
}
"""

import argparse
import json
import os
import time
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import zarr

from Core.SINDy_Modules import (
    SINDy_Library,
    SINDy_Residual,
    KNODE_SINDy,
    fit_SINDy,
    build_library_matrix,
)
import Utils.Mujoco_Utils as mj_utils


# ============================================================================
# ARGUMENT / SPEC PARSING  (mirrors Train_Model_Old.py)
# ============================================================================

def return_Arguments():
    parser = argparse.ArgumentParser()
#    parser.add_argument("-m", "--model_specs",    type=str, default="model_fluid_sindy")
    parser.add_argument("-t", "--training_specs", type=str, default="training_sindy")
    args = parser.parse_args()
#    model_path    = "Library/Model_Specifications/"    + args.model_specs    + ".json"
    training_path = "Library/SINDy_Specifications/" + args.training_specs + ".json"
    return training_path


def parse_Specs(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_Residual_Dataset(training_options: dict):
    """
    Load (qpos, qvel, ctrl, residual_qacc) from a zarr dataset and return
    flat (n_total_samples, *) arrays.

    If training_options["max_samples"] is set, sub-samples randomly.
    """
    dataset_path = "Datasets/" + training_options["sindy_dataset_name"] + ".zarr"
    root = zarr.open(dataset_path, mode="r")

    qpos_list, qvel_list, ctrl_list, res_list = [], [], [], []

    keys = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    
    if training_options["subset_size"] == None:
        subset_size = len(keys)
    elif training_options["subset_size"]>len(keys):
        raise ValueError("Subset Size Cannot Exceed Dataset Size")
    else:
        subset_size = training_options["subset_size"]
    
    for traj in range(subset_size):
        k = keys[traj]
        d = root[k]
        qpos_traj = d["qpos"][:]   # (T, nq)
        qvel_traj = d["qvel"][:]   # (T, nv)
        ctrl_raw  = d["ctrl"][:]

        # ctrl may be stored as (nu,) constant or (T, nu)
        if ctrl_raw.ndim == 1:
            ctrl_traj = np.tile(ctrl_raw[None, :], (qpos_traj.shape[0], 1))
        else:
            ctrl_traj = ctrl_raw

        res_traj  = d["res_qacc"][:]   # (T, nq)

        qpos_list.append(qpos_traj)
        qvel_list.append(qvel_traj)
        ctrl_list.append(ctrl_traj)
        res_list.append(res_traj)
        
        if traj%10 == 0:
            print(f"Trajectory {traj} of {subset_size} Appended.")

    qpos_arr = np.concatenate(qpos_list, axis=0).astype(np.float64)
    qvel_arr = np.concatenate(qvel_list, axis=0).astype(np.float64)
    ctrl_arr = np.concatenate(ctrl_list, axis=0).astype(np.float64)
    res_arr  = np.concatenate(res_list,  axis=0).astype(np.float64)

    print(f"Loaded {qpos_arr.shape[0]} samples from {len(keys)} trajectory/ies.")
    print(f"  qpos shape : {qpos_arr.shape}")
    print(f"  qvel shape : {qvel_arr.shape}")
    print(f"  ctrl shape : {ctrl_arr.shape}")
    print(f"  res  shape : {res_arr.shape}")

    # Optional sub-sampling for speed / memory
    max_n = training_options.get("max_samples", None)
    if max_n is not None and max_n < qpos_arr.shape[0]:
        rng  = np.random.default_rng(seed=42)
        idx  = rng.choice(qpos_arr.shape[0], size=int(max_n), replace=False)
        idx.sort()
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
# LIBRARY CONSTRUCTION
# ============================================================================

def build_SINDy_Library(training_options: dict, mjx_model) -> SINDy_Library:
    """
    Construct the SINDy feature library from the training spec and MJX model.
    Prints the full feature name list for inspection.
    """
    qvel_max    = get_qvel_max(training_options)
    qpos_ranges = mjx_model.jnt_range
    ctrl_ranges = mjx_model.actuator_ctrlrange

    library = SINDy_Library(
        nq            = mjx_model.nq,
        nv            = mjx_model.nv,
        nu            = mjx_model.nu,
        qpos_ranges   = qpos_ranges,
        qvel_max      = qvel_max,
        ctrl_ranges   = ctrl_ranges,
        poly_degree   = training_options.get("poly_degree",    2),
        n_frequencies = training_options.get("n_frequencies",  1),
        include_bias  = training_options.get("include_bias",   True),
        include_sin   = training_options.get("include_sin",    True),
        include_cos   = training_options.get("include_cos",    True),
        include_sum_fourier = training_options.get("include_sum_fourier", True),
        sum_fourier_degree  = training_options.get("sum_fourier_degree", 2),
    )

    n_poly    = library.n_poly_terms
    n_fourier = library.n_fourier_terms
    n_total   = library.n_library_features

    print(f"\nLibrary:  {n_poly} polynomial  +  {n_fourier} Fourier  =  {n_total} features")
    print(f"  poly_degree   = {training_options.get('poly_degree', 2)}")
    print(f"  n_frequencies = {training_options.get('n_frequencies', 1)}")
    print(f"  Features ({n_total}):")
    for i, name in enumerate(library.feature_names):
        print(f"    [{i:3d}]  {name}")

    return library


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


def save_SINDy_Model(
    sindy_residual:   SINDy_Residual,
    training_options: dict,
) -> str:
    name      = training_options.get("name", "sindy_model")
    save_path = _unique_save_dir(name)
    os.makedirs(save_path)

    # Serialise the eqx.Module (same pattern as save_Model in Train_Model_Old.py)
    eqx.tree_serialise_leaves(os.path.join(save_path, "sindy_residual.eqx"), sindy_residual)

    with open(os.path.join(save_path, "training_options.json"), "w") as f:
        json.dump(training_options, f, indent=2)
#    if model_options:
#        with open(os.path.join(save_path, "model_options.json"), "w") as f:
#            json.dump(model_options, f, indent=2)

    # Also save coefficient matrix as numpy for easy external inspection / plotting
    np.save(os.path.join(save_path, "coefficients.npy"),
            np.array(sindy_residual.coefficients))
    np.save(os.path.join(save_path, "feature_names.npy"),
            np.array(sindy_residual.library.feature_names, dtype=object))

    print(f"\nModel saved to: {save_path}")
    return save_path
    
def save_Model(save_path , model):
    
    eqx.tree_serialise_leaves(save_path + "/model.eqx" , model)



def load_SINDy_Model(
    save_path:      str,
    mjx_model,
    training_options: dict,
) -> SINDy_Residual:
    """Deserialise a previously saved SINDy_Residual."""
    saved_opts = parse_Specs(os.path.join(save_path, "training_options.json"))
    qvel_max   = get_qvel_max(saved_opts)

    library = SINDy_Library(
        nq            = mjx_model.nq,
        nv            = mjx_model.nv,
        nu            = mjx_model.nu,
        qpos_ranges   = mjx_model.jnt_range,
        qvel_max      = qvel_max,
        ctrl_ranges   = mjx_model.actuator_ctrlrange,
        poly_degree   = saved_opts.get("poly_degree",    2),
        n_frequencies = saved_opts.get("n_frequencies",  1),
        include_bias  = saved_opts.get("include_bias",   True),
        include_sin   = saved_opts.get("include_sin",    True),
        include_cos   = saved_opts.get("include_cos",    True),
    )
    # Build a zero-coefficient shell then deserialise
    dummy_model = SINDy_Residual(library, mjx_model.nq)
    return eqx.tree_deserialise_leaves(
        os.path.join(save_path, "model.eqx"), dummy_model
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_SINDy(
    sindy_residual: SINDy_Residual,
    qpos_arr:       np.ndarray,
    qvel_arr:       np.ndarray,
    ctrl_arr:       np.ndarray,
    res_arr:        np.ndarray,
    n_show:         int = 5,
) -> dict:
    """Compute MSE / R² on a labelled dataset and print a quick report."""
    X   = jnp.concatenate(
        [jnp.array(qpos_arr), jnp.array(qvel_arr), jnp.array(ctrl_arr)],
        axis=-1
    )
    pred   = np.array(jax.vmap(sindy_residual)(X))   # (n_samples, nq)
    target = np.array(res_arr)

    mse    = np.mean((pred - target) ** 2)
    rmse   = np.sqrt(mse)
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)

    per_dof_rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))

    print("\n--- Evaluation ---")
    print(f"  MSE   = {mse:.6e}")
    print(f"  RMSE  = {rmse:.6e}")
    print(f"  R²    = {r2:.4f}")
    print(f"  Per-DOF RMSE: {np.round(per_dof_rmse, 6)}")

    if n_show > 0:
        print(f"\n  First {n_show} sample predictions vs targets:")
        for i in range(min(n_show, len(pred))):
            print(f"    [{i}]  pred ={np.round(pred[i], 4)}  "
                  f"target={np.round(target[i], 4)}")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "per_dof_rmse": per_dof_rmse.tolist(),
    }


# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_SINDy_Fit(
    sindy_residual: SINDy_Residual,
    qpos_arr:       np.ndarray,
    qvel_arr:       np.ndarray,
    ctrl_arr:       np.ndarray,
    res_arr:        np.ndarray,
    metrics:        dict,
    training_options: dict,
) -> None:
    """
    Plot:
      • coefficient heat-map (library_features × DOFs)
      • per-DOF residual: pred vs target scatter
      • per-DOF time-series for one trajectory
    """
    xi = np.array(sindy_residual.coefficients)   # (n_lib, nq)
    nq = sindy_residual.output_num
    names = sindy_residual.library.feature_names

    n_axes = 3
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, max(4, nq * 0.4 + 2)))
    fig.suptitle(
        f"SINDy Fit  |  R²={metrics['r2']:.4f}  "
        f"RMSE={metrics['rmse']:.2e}  "
        f"optimizer={training_options.get('sindy_optimizer','stlsq').upper()}  "
        f"threshold={training_options.get('sindy_threshold', 0.05)}",
        fontsize=11,
    )

    # ---- 1. Coefficient heat-map ----
    ax = axes[0]
    im = ax.imshow(xi, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(xi)) - 1e-8,
                   vmax= np.max(np.abs(xi)) + 1e-8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=max(4, 9 - len(names) // 10))
    ax.set_xticks(range(nq))
    ax.set_xticklabels([f"qacc[{i}]" for i in range(nq)], rotation=45, ha="right")
    ax.set_title("Coefficient matrix Ξ\n(non-zero = active term)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    # ---- 2. Pred vs target scatter (all DOFs combined) ----
    ax = axes[1]
    X   = jnp.concatenate(
        [jnp.array(qpos_arr), jnp.array(qvel_arr), jnp.array(ctrl_arr)],
        axis=-1
    )
    pred = np.array(jax.vmap(sindy_residual)(X))
    flat_pred   = pred.ravel()
    flat_target = res_arr.ravel()
    max_abs = max(np.abs(flat_target).max(), np.abs(flat_pred).max()) + 1e-8
    ax.scatter(flat_target, flat_pred, alpha=0.15, s=2, rasterized=True)
    ax.plot([-max_abs, max_abs], [-max_abs, max_abs], "r--", lw=1.5, label="perfect")
    ax.set_xlabel("Ground-truth residual qacc")
    ax.set_ylabel("SINDy prediction")
    ax.set_title("Pred vs Target (all DOFs)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", "box")

    # ---- 3. Time-series for first trajectory, first DOF ----
    ax = axes[2]
    T_show = min(pred.shape[0], 200)
    for dof in range(min(nq, 4)):
        ax.plot(res_arr[:T_show, dof],  lw=1.5, label=f"GT DOF {dof}",   ls="-")
        ax.plot(pred[:T_show, dof],     lw=1.0, label=f"SINDy DOF {dof}", ls="--")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Residual qacc")
    ax.set_title(f"Time-series (first {T_show} samples, first {min(nq,4)} DOFs)")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.show(block=False)


def visualize_Coefficient_Sparsity(sindy_residual: SINDy_Residual) -> None:
    """
    Bar chart of coefficient magnitudes sorted by absolute value —
    handy for choosing the threshold hyperparameter.
    """
    xi   = np.abs(np.array(sindy_residual.coefficients))   # (n_lib, nq)
    mags = xi.max(axis=1)   # largest magnitude per feature across all DOFs
    order = np.argsort(mags)[::-1]
    names = np.array(sindy_residual.library.feature_names)

    fig, ax = plt.subplots(figsize=(max(10, len(mags) // 4), 4))
    ax.bar(range(len(mags)), mags[order], color="steelblue")
    ax.set_xticks(range(len(mags)))
    ax.set_xticklabels(names[order],
                       rotation=90, fontsize=max(5, 10 - len(mags) // 20))
    ax.set_ylabel("max |ξ| across DOFs")
    ax.set_title("Feature coefficient magnitudes (sorted)  —  use to tune threshold")
    ax.set_yscale("symlog", linthresh=1e-4)
    plt.tight_layout()
    plt.show(block=False)
    
    
def visualize_Trained_Model(model , training_options):
    root = zarr.open(os.getcwd() + "/Datasets/" + training_options["sindy_dataset_name"] + ".zarr" , mode = 'r')
    dt = root.attrs.get("dataset_options")["step_size"]
    xml = root.attrs.get("xml")
    mj_model , data = mj_utils.init_mujoco_model(xml)
    
    _ , mjx_data = mj_utils.init_mjx_model(xml)
    
    trajectory_number = training_options["which_to_show_at_end"]
    keys = sorted(root.keys(), key=lambda k: int(k.split('_')[-1]))
    t = keys[trajectory_number]
    print(t)
    d = root[t]#"trajectory_" + str(trajectory_number)
    gt_qpos_traj = d["qpos"][:]
#    print(d["qpos"])
    
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0 , step_size*(num_steps) , step = step_size)
    
    init_states = {"qpos": jnp.array(d["qpos"][0 , :]) , "qvel": jnp.array(d["qvel"][0 , :])}
    ctrl = jnp.array(d["ctrl"][0 , :])
    
    model.knode_dynamics(0 , init_states , (mjx_data , ctrl , jnp.array(1.0)))
    
    KNODE_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(1.0))
    KNODE_qpos_traj = np.array(KNODE_traj["qpos"])
    
    nominal_traj = model(ts , mjx_data , init_states , ctrl , with_learned = jnp.array(0.0))
    nominal_qpos_traj = np.array(nominal_traj["qpos"])
    
    
    

    l2 = training_options["l2"]
    l4 = training_options["l4"]
    
    
    
             
    print(l2*jnp.mean((KNODE_traj["qpos"] - gt_qpos_traj)**2) + l4*jnp.mean((KNODE_traj["qvel"] - d["qvel"][:])**2))
    

    
    fig , (ax1 , ax2 , ax3 , ax4 , ax5 , ax6) = plt.subplots(1 , 6)
    
#    loss_inds = np.arange(1 , len(loss_list) + 1)
#    ax1.plot(loss_inds , loss_list)
#    ax2.loglog(loss_inds , loss_list)
    
    
    ax3.plot(KNODE_traj["qpos"] , color = 'green')
    ax3.plot(nominal_traj["qpos"] , color = 'red')
    ax3.plot(d["qpos"] , color = 'black')
    
    ax4.plot(KNODE_traj["qvel"] , color = 'green')
    ax4.plot(nominal_traj["qvel"] , color = 'red')
    ax4.plot(d["qvel"] , color = 'black')
    
    KNODE_acc = (KNODE_traj["qvel"][1:] - KNODE_traj["qvel"][:-1])/dt
    nominal_acc = (nominal_traj["qvel"][1:] - nominal_traj["qvel"][:-1])/dt
    gt_acc = (d["qvel"][1:] - d["qvel"][:-1])/dt
    
    ax5.plot(KNODE_acc , color = 'green')
    ax5.plot(nominal_acc , color = 'red')
    ax5.plot(gt_acc , color = 'black')
    
    ax6.plot(KNODE_acc - nominal_acc , color = 'green')
#    ax5.plot(nominal_acc , color = 'red')
    ax6.plot((gt_acc - nominal_acc) , color = 'black')
    
   
    plt.show(block = False)
    
    
    plt.show()
    
    mj_utils.visualize_GT_vs_Nominal_vs_KNODE_Trajectory(training_options["model_name"] , gt_qpos_traj , nominal_qpos_traj , KNODE_qpos_traj)



# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # ---- Parse args & specs ----
    training_path = return_Arguments()
#    model_options    = parse_Specs(model_path)
    training_options = parse_Specs(training_path)

    # ---- Build MJX model ----
    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    # ---- Load residual dataset ----
    t0 = time.time()
    qpos_arr, qvel_arr, ctrl_arr, res_arr = load_Residual_Dataset(training_options)
    print(f"  Dataset loaded in {time.time() - t0:.2f}s")

    # ---- Build feature library ----
    library = build_SINDy_Library(training_options, mjx_model)

    # ---- Fit SINDy ----
    t0 = time.time()
    sindy_residual = fit_SINDy(
        library        = library,
        qpos_arr       = qpos_arr,
        qvel_arr       = qvel_arr,
        ctrl_arr       = ctrl_arr,
        residual_qacc  = res_arr,
        output_num     = mjx_model.nq,
        optimizer      = training_options.get("sindy_optimizer",  "stlsq"),
        threshold      = training_options.get("sindy_threshold",  0.05),
        alpha          = training_options.get("sindy_alpha",       1e-5),
        nu             = training_options.get("sindy_nu",          1.0),
        max_iter       = training_options.get("sindy_max_iter",    20),
        verbose        = True,
    )
    print(f"  Fitting completed in {time.time() - t0:.2f}s")

    # ---- Sparsity report ----
    report = sindy_residual.sparsity_report()
    print(f"\nSparsity : {report['sparsity_%']:.1f}%  "
          f"({report['n_nonzero']} active / {report['n_total']} total coefficients)")

    # ---- Print equations ----
    sindy_residual.print_equations()

    if training_options["should_show"]:
        # ---- Evaluate ----
        metrics = evaluate_SINDy(
            sindy_residual,
            qpos_arr, qvel_arr, ctrl_arr, res_arr,
            n_show=training_options.get("which_to_show_at_end", 5),
        )


        # ---- Visualise ----
        visualize_SINDy_Fit(
            sindy_residual, qpos_arr, qvel_arr, ctrl_arr, res_arr,
            metrics, training_options,
        )
        visualize_Coefficient_Sparsity(sindy_residual)
    
    full_model = KNODE_SINDy(mjx_model , sindy_residual)
    
    if training_options["should_save"]:
        save_path = save_SINDy_Model(sindy_residual, training_options)
        save_Model(save_path , full_model)
    
    if training_options["should_plot"]:
        visualize_Trained_Model(full_model , training_options)
        plt.show()
    
    
