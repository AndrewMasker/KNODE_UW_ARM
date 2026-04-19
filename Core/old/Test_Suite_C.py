import os
import argparse
import equinox as eqx
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – swap to "TkAgg" / "Qt5Agg" if you want live windows
import matplotlib.pyplot as plt
import mujoco.mjx as mjx
import numpy as np
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils
import zarr


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT AND SPECIFICATION PARSING
# ══════════════════════════════════════════════════════════════════════════════

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_specs", type=str, default="test_new")
    args = parser.parse_args()
    test_path = "Library/Test_Specifications/" + args.test_specs + ".json"
    return test_path


def parse_Specs(path_to_test_specs):
    with open(path_to_test_specs) as f:
        d = json.load(f)
    return d


def get_New_Name(name, *args):
    if len(args) == 0:
        if os.path.exists(os.getcwd() + "/Test_Outputs/" + name):
            return get_New_Name(name, 1)
        else:
            return name
    else:
        if os.path.exists(os.getcwd() + "/Test_Outputs/" + name + "_" + str(args[0])):
            return get_New_Name(name, args[0] + 1)
        else:
            return name + "_" + str(args[0])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_Training_Options(test_options):
    load_name = test_options["load_name"]
    path = "Models/" + load_name + "/training_options.json"
    with open(path) as f:
        return json.load(f)


def get_Model_Options(test_options):
    load_name = test_options["load_name"]
    path = "Models/" + load_name + "/model_options.json"
    with open(path) as f:
        return json.load(f)


def load_Old_Model(test_options):
    from Core.KNODE_Modules_Old import KNODE

    training_options = get_Training_Options(test_options)
    model_options    = get_Model_Options(test_options)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root         = zarr.open(dataset_path)
    qvel_max     = jnp.array(root.attrs["dataset_options"]["qvel_max"])

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    key   = jr.key(int(time.time()))
    model = KNODE(key, mjx_model, model_options["width"], model_options["depth"], qvel_max)

    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data, mjx_model


def load_New_Model(test_options):
    from Core.KNODE_Modules import KNODE

    training_options = get_Training_Options(test_options)
    model_options    = get_Model_Options(test_options)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root         = zarr.open(dataset_path)
    qvel_max     = jnp.array(root.attrs["dataset_options"]["qvel_max"])

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    key   = jr.key(int(time.time()))
    model = KNODE(key, mjx_model, model_options["width"], model_options["depth"], qvel_max)

    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data, mjx_model


def load_SINDy_Model(test_options):
    from Core.SINDy_Modules    import SINDy_Residual, KNODE_SINDy
    from Core.Visualize_SINDy  import build_SINDy_Library

    training_options = get_Training_Options(test_options)

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    library        = build_SINDy_Library(training_options, mjx_model)
    sindy_residual = SINDy_Residual(
        library, mjx_model.nq,
        jnp.zeros((library.n_library_features, mjx_model.nq))
    )
    model = KNODE_SINDy(mjx_model, sindy_residual)

    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data, mjx_model


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY RUNNING  (JIT-compiled, vmapped)
# ══════════════════════════════════════════════════════════════════════════════

def run_Single_Trajectory(model, mjx_data, gt_qpos_traj, gt_qvel_traj, ctrl, ts):
    init_states  = {"qpos": gt_qpos_traj[0, :], "qvel": gt_qvel_traj[0, :]}
    learned_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
    nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))
    return (
        learned_traj["qpos"], learned_traj["qvel"],
        nominal_traj["qpos"], nominal_traj["qvel"],
    )


@eqx.filter_jit
def run_Trajectories(model, mjx_data, ts, gt_qpos_trajs, gt_qvel_trajs, ctrls):
    lqp, lqv, nqp, nqv = jax.vmap(
        run_Single_Trajectory, in_axes=(None, None, 0, 0, 0, None)
    )(model, mjx_data, gt_qpos_trajs, gt_qvel_trajs, ctrls, ts)
    return lqp, lqv, nqp, nqv, gt_qpos_trajs, gt_qvel_trajs


# ══════════════════════════════════════════════════════════════════════════════
# SAVING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def make_dir(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def save_Figure(fig, path_no_ext):
    """Save figure as PNG (150 dpi)."""
    fig.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")


def save_Plot_Data(data_dict, path_no_ext):
    """Save the raw arrays behind a figure so plots can be rebuilt later."""
    np.savez(path_no_ext + "_data.npz", **data_dict)


def json_safe(v):
    """Convert numpy scalars/arrays to JSON-serialisable Python types."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ══════════════════════════════════════════════════════════════════════════════
# ERROR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_Metrics(pred, gt, label):
    """
    Compute a suite of error metrics.

    Parameters
    ----------
    pred, gt : array_like  (N, T, D)
    label    : str  prefix for dict keys, e.g. "learned_qpos"

    Returns
    -------
    scalars : dict of float   – overall summary numbers
    arrays  : dict of ndarray – per-joint and per-timestep breakdowns
    """
    err    = np.asarray(pred) - np.asarray(gt)
    sq_err = err ** 2
    abs_err = np.abs(err)

    rmse_overall = float(np.sqrt(np.mean(sq_err)))
    mae_overall  = float(np.mean(abs_err))
    max_err      = float(np.max(abs_err))
    rmse_final   = float(np.sqrt(np.mean(sq_err[:, -1, :])))   # drift at end of rollout

    # Per-joint RMSE: average over trajectories and time  →  (D,)
    rmse_per_joint = np.sqrt(np.mean(sq_err, axis=(0, 1)))

    # Per-timestep RMSE: average over trajectories and joints  →  (T,)
    rmse_per_timestep = np.sqrt(np.mean(sq_err, axis=(0, 2)))

    scalars = {
        f"{label}_rmse":          rmse_overall,
        f"{label}_mae":           mae_overall,
        f"{label}_max_err":       max_err,
        f"{label}_rmse_final_t":  rmse_final,
    }
    arrays = {
        f"{label}_rmse_per_joint":     rmse_per_joint,
        f"{label}_rmse_per_timestep":  rmse_per_timestep,
    }
    return scalars, arrays


def improvement_pct(learned_rmse, nominal_rmse):
    """Percentage RMSE reduction: positive = learned is better."""
    return float((nominal_rmse - learned_rmse) / (nominal_rmse + 1e-12) * 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS JOB
# ══════════════════════════════════════════════════════════════════════════════

def run_Statistics_Job(model, mjx_data, mjx_model, job_options, job_name, save_path):
    job_dir = make_dir(save_path, job_name)

    # ── Load dataset info ──────────────────────────────────────────────────
    root      = zarr.open(os.getcwd() + "/Datasets/" + job_options["dataset_name"] + ".zarr", mode="r")
    keys      = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts        = jnp.arange(0, step_size * num_steps, step=step_size)

    if job_options["subset_size"] == "None":
        subset_size = len(keys)
    elif job_options["subset_size"] > len(keys):
        raise ValueError("Subset size cannot exceed dataset size.")
    else:
        subset_size = job_options["subset_size"]

    batch_size = job_options["batch_size"]
    if subset_size % batch_size != 0:
        raise ValueError("Subset size must be divisible by batch size.")
    num_batches = subset_size // batch_size

    # ── Collect trajectory predictions ────────────────────────────────────
    all_learned_qpos, all_learned_qvel = [], []
    all_nominal_qpos, all_nominal_qvel = [], []
    all_gt_qpos,      all_gt_qvel      = [], []

    for batch in range(num_batches):
        gt_qpos_list, gt_qvel_list, ctrl_list = [], [], []
        for idx in range(batch * batch_size, (batch + 1) * batch_size):
            d = root[keys[idx]]
            gt_qpos_list.append(np.array(d["qpos"][:]))
            gt_qvel_list.append(np.array(d["qvel"][:]))
            ctrl_list.append(np.array(d["ctrl"][0, :]))

        gt_qpos_arr = jnp.array(np.stack(gt_qpos_list, axis=0))
        gt_qvel_arr = jnp.array(np.stack(gt_qvel_list, axis=0))
        ctrl_arr    = jnp.array(np.stack(ctrl_list,    axis=0))

        lqp, lqv, nqp, nqv, gqp, gqv = run_Trajectories(
            model, mjx_data, ts, gt_qpos_arr, gt_qvel_arr, ctrl_arr
        )
        all_learned_qpos.append(np.array(lqp));  all_learned_qvel.append(np.array(lqv))
        all_nominal_qpos.append(np.array(nqp));  all_nominal_qvel.append(np.array(nqv))
        all_gt_qpos.append(np.array(gqp));       all_gt_qvel.append(np.array(gqv))
        print(f"  Batch {batch + 1}/{num_batches} complete.")

    learned_qpos = np.concatenate(all_learned_qpos, axis=0)
    learned_qvel = np.concatenate(all_learned_qvel, axis=0)
    nominal_qpos = np.concatenate(all_nominal_qpos, axis=0)
    nominal_qvel = np.concatenate(all_nominal_qvel, axis=0)
    gt_qpos      = np.concatenate(all_gt_qpos,      axis=0)
    gt_qvel      = np.concatenate(all_gt_qvel,      axis=0)

    # ── Compute metrics ───────────────────────────────────────────────────
    lqp_sc, lqp_ar = compute_Metrics(learned_qpos, gt_qpos, "learned_qpos")
    nqp_sc, nqp_ar = compute_Metrics(nominal_qpos, gt_qpos, "nominal_qpos")
    lqv_sc, lqv_ar = compute_Metrics(learned_qvel, gt_qvel, "learned_qvel")
    nqv_sc, nqv_ar = compute_Metrics(nominal_qvel, gt_qvel, "nominal_qvel")

    qpos_impr = improvement_pct(lqp_sc["learned_qpos_rmse"], nqp_sc["nominal_qpos_rmse"])
    qvel_impr = improvement_pct(lqv_sc["learned_qvel_rmse"], nqv_sc["nominal_qvel_rmse"])

    # ── Numerical acceleration metrics ────────────────────────────────────
    dt = float(step_size)
    learned_qacc = (learned_qvel[:, 1:, :] - learned_qvel[:, :-1, :]) / dt
    nominal_qacc = (nominal_qvel[:, 1:, :] - nominal_qvel[:, :-1, :]) / dt
    gt_qacc      = (gt_qvel[:, 1:, :]      - gt_qvel[:, :-1, :])      / dt

    lqa_sc, lqa_ar = compute_Metrics(learned_qacc, gt_qacc, "learned_qacc")
    nqa_sc, nqa_ar = compute_Metrics(nominal_qacc, gt_qacc, "nominal_qacc")
    qacc_impr = improvement_pct(lqa_sc["learned_qacc_rmse"], nqa_sc["nominal_qacc_rmse"])

    # ── Body-space (xpos) metrics  –  optional ────────────────────────────
    xpos_scalars, xpos_arrays = {}, {}
    if mjx_model is not None:
        try:
            def _get_xpos(qpos_vec, qvel_vec):
                d = mjx_data.replace(qpos=qpos_vec, qvel=qvel_vec)
                d = mjx.com_pos(mjx_model, mjx.kinematics(mjx_model, d))
                return d.xpos  # (nbody, 3)

            _get_xpos_traj = jax.vmap(
                jax.vmap(_get_xpos, in_axes=(0, 0)), in_axes=(0, 0)
            )
            lxp = np.array(_get_xpos_traj(learned_qpos, learned_qvel)).reshape(learned_qpos.shape[0], learned_qpos.shape[1], -1)
            nxp = np.array(_get_xpos_traj(nominal_qpos, nominal_qvel)).reshape(nominal_qpos.shape[0], nominal_qpos.shape[1], -1)
            gxp = np.array(_get_xpos_traj(gt_qpos, gt_qvel)).reshape(gt_qpos.shape[0], gt_qpos.shape[1], -1)

            lx_sc, lx_ar = compute_Metrics(lxp, gxp, "learned_xpos")
            nx_sc, nx_ar = compute_Metrics(nxp, gxp, "nominal_xpos")
            xpos_impr = improvement_pct(lx_sc["learned_xpos_rmse"], nx_sc["nominal_xpos_rmse"])

            xpos_scalars = {**lx_sc, **nx_sc, "xpos_improvement_pct": xpos_impr}
            xpos_arrays  = {**lx_ar, **nx_ar}
            np.savez(os.path.join(job_dir, "xpos_errors.npz"), **xpos_arrays)
        except Exception as e:
            print(f"  Warning: could not compute body-space metrics: {e}")

    # ── Print summary ──────────────────────────────────────────────────────
    N, T, nq = learned_qpos.shape
    print("\n" + "═" * 54)
    print("  STATISTICS SUMMARY")
    print("═" * 54)
    print(f"  Trajectories:  {N}   |   Timesteps: {T}   |   DOF: {nq}")
    print("─" * 54)
    _print_metric_row("qpos", lqp_sc["learned_qpos_rmse"], nqp_sc["nominal_qpos_rmse"],
                      lqp_sc["learned_qpos_rmse_final_t"], nqp_sc["nominal_qpos_rmse_final_t"], qpos_impr)
    _print_metric_row("qvel", lqv_sc["learned_qvel_rmse"], nqv_sc["nominal_qvel_rmse"],
                      lqv_sc["learned_qvel_rmse_final_t"], nqv_sc["nominal_qvel_rmse_final_t"], qvel_impr)
    _print_metric_row("qacc", lqa_sc["learned_qacc_rmse"], nqa_sc["nominal_qacc_rmse"],
                      lqa_sc["learned_qacc_rmse_final_t"], nqa_sc["nominal_qacc_rmse_final_t"], qacc_impr)
    if xpos_scalars:
        xi = xpos_scalars
        _print_metric_row("xpos",
                          xi["learned_xpos_rmse"], xi["nominal_xpos_rmse"],
                          xi["learned_xpos_rmse_final_t"], xi["nominal_xpos_rmse_final_t"],
                          xi["xpos_improvement_pct"])
    print("═" * 54 + "\n")

    # ── Save summary JSON ──────────────────────────────────────────────────
    summary = {
        **lqp_sc, **nqp_sc,
        **lqv_sc, **nqv_sc,
        **lqa_sc, **nqa_sc,
        **xpos_scalars,
        "qpos_improvement_pct": qpos_impr,
        "qvel_improvement_pct": qvel_impr,
        "qacc_improvement_pct": qacc_impr,
        "num_trajectories": N,
        "num_timesteps":    T,
        "dof":              nq,
    }
    with open(os.path.join(job_dir, "summary.json"), "w") as f:
        json.dump({k: json_safe(v) for k, v in summary.items()}, f, indent=2)

    # ── Save error arrays ──────────────────────────────────────────────────
    np.savez(os.path.join(job_dir, "qpos_errors.npz"), ts=np.array(ts), **lqp_ar, **nqp_ar)
    np.savez(os.path.join(job_dir, "qvel_errors.npz"), ts=np.array(ts), **lqv_ar, **nqv_ar)
    np.savez(os.path.join(job_dir, "qacc_errors.npz"), ts=np.array(ts[:-1]), **lqa_ar, **nqa_ar)

    # ── Plots ──────────────────────────────────────────────────────────────
    _plot_Statistics(
        lqp_ar, nqp_ar,
        lqv_ar, nqv_ar,
        lqa_ar, nqa_ar,
        np.array(ts), dt,
        job_dir
    )

    print(f"  Results saved to: {job_dir}\n")


def _print_metric_row(name, l_rmse, n_rmse, l_final, n_final, impr):
    print(f"  {name:<6}  RMSE  learned={l_rmse:.5f}  nominal={n_rmse:.5f}  Δ={impr:+.1f}%")
    print(f"         final RMSE  learned={l_final:.5f}  nominal={n_final:.5f}")


def _plot_Statistics(lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar, ts, dt, job_dir):
    plots_dir = make_dir(job_dir, "plots")
    ts_acc    = ts[:-1]

    # ── Figure 1: per-timestep RMSE growth ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RMSE vs Time (averaged over all trajectories & joints)", fontsize=12)

    for ax, (l_key, n_key, title, ylabel) in zip(axes, [
        ("learned_qpos_rmse_per_timestep", "nominal_qpos_rmse_per_timestep", "Joint Position (qpos)", "RMSE"),
        ("learned_qvel_rmse_per_timestep", "nominal_qvel_rmse_per_timestep", "Joint Velocity (qvel)", "RMSE"),
        ("learned_qacc_rmse_per_timestep", "nominal_qacc_rmse_per_timestep", "Joint Accel. (qacc)", "RMSE"),
    ]):
        l_ts = ts_acc if "qacc" in l_key else ts
        ax.plot(l_ts, lqp_ar[l_key] if "qpos" in l_key else (lqv_ar[l_key] if "qvel" in l_key else lqa_ar[l_key]),
                color="green", label="Learned", linewidth=1.5)
        ax.plot(l_ts, nqp_ar[n_key] if "qpos" in n_key else (nqv_ar[n_key] if "qvel" in n_key else nqa_ar[n_key]),
                color="red",   label="Nominal", linewidth=1.5, linestyle="--")
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = os.path.join(plots_dir, "rmse_vs_time")
    save_Figure(fig, stem)
    save_Plot_Data({
        "ts": ts, "ts_acc": ts_acc,
        **lqp_ar, **nqp_ar, **lqv_ar, **nqv_ar, **lqa_ar, **nqa_ar,
    }, stem)
    plt.close(fig)

    # ── Figure 2: per-joint RMSE bar chart ──────────────────────────────────
    nq = lqp_ar["learned_qpos_rmse_per_joint"].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RMSE per Joint (averaged over all trajectories & timesteps)", fontsize=12)
    joint_idx = np.arange(nq)
    w = 0.35

    for ax, (l_vals, n_vals, title) in zip(axes, [
        (lqp_ar["learned_qpos_rmse_per_joint"], nqp_ar["nominal_qpos_rmse_per_joint"], "Joint Position (qpos)"),
        (lqv_ar["learned_qvel_rmse_per_joint"], nqv_ar["nominal_qvel_rmse_per_joint"], "Joint Velocity (qvel)"),
        (lqa_ar["learned_qacc_rmse_per_joint"], nqa_ar["nominal_qacc_rmse_per_joint"], "Joint Accel. (qacc)"),
    ]):
        ax.bar(joint_idx - w/2, l_vals, w, label="Learned", color="green", alpha=0.75)
        ax.bar(joint_idx + w/2, n_vals, w, label="Nominal", color="red",   alpha=0.75)
        ax.set_title(title); ax.set_xlabel("Joint Index"); ax.set_ylabel("RMSE")
        ax.set_xticks(joint_idx)
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    stem = os.path.join(plots_dir, "rmse_per_joint")
    save_Figure(fig, stem)
    save_Plot_Data({
        "joint_idx": joint_idx,
        "learned_qpos_per_joint": lqp_ar["learned_qpos_rmse_per_joint"],
        "nominal_qpos_per_joint": nqp_ar["nominal_qpos_rmse_per_joint"],
        "learned_qvel_per_joint": lqv_ar["learned_qvel_rmse_per_joint"],
        "nominal_qvel_per_joint": nqv_ar["nominal_qvel_rmse_per_joint"],
        "learned_qacc_per_joint": lqa_ar["learned_qacc_rmse_per_joint"],
        "nominal_qacc_per_joint": nqa_ar["nominal_qacc_rmse_per_joint"],
    }, stem)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE JOB
# ══════════════════════════════════════════════════════════════════════════════

def run_Visualize_Job(model, mjx_data, mjx_model, job_options, job_name, save_path):
    job_dir = make_dir(save_path, job_name)

    root      = zarr.open(os.getcwd() + "/Datasets/" + job_options["dataset_name"] + ".zarr", mode="r")
    keys      = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts        = jnp.arange(0, step_size * num_steps, step=step_size)
    ts_np     = np.array(ts)
    dt        = float(step_size)

    for traj_idx in job_options["which_to_visualize"]:
        print(f"  Visualizing trajectory {traj_idx} ...")
        d = root[keys[traj_idx]]

        gt_qpos = jnp.array(d["qpos"][:])    # (T, nq)
        gt_qvel = jnp.array(d["qvel"][:])    # (T, nv)
        ctrl    = jnp.array(d["ctrl"][0, :]) # (nu,)

        init_states  = {"qpos": gt_qpos[0], "qvel": gt_qvel[0]}
        learned_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
        nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))

        lqp = np.array(learned_traj["qpos"])
        lqv = np.array(learned_traj["qvel"])
        nqp = np.array(nominal_traj["qpos"])
        nqv = np.array(nominal_traj["qvel"])
        gqp = np.array(gt_qpos)
        gqv = np.array(gt_qvel)

        # Numerical acceleration
        lqa   = (lqv[1:] - lqv[:-1]) / dt
        nqa   = (nqv[1:] - nqv[:-1]) / dt
        gqa   = (gqv[1:] - gqv[:-1]) / dt
        ts_acc = ts_np[:-1]

        residual_acc = lqa - nqa   # contribution of the learned residual

        traj_dir = make_dir(job_dir, f"trajectory_{traj_idx}")
        nq = gqp.shape[1]

        # ── Plot 1: per-joint state comparison ──────────────────────────────
        fig, axes = plt.subplots(3, nq, figsize=(max(3 * nq, 10), 10), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Per-Joint State Comparison\n"
                     f"(black=GT, green=Learned, red=Nominal)", fontsize=11)

        for j in range(nq):
            for row, (gt_y, l_y, n_y, t_arr, row_label) in enumerate([
                (gqp[:, j], lqp[:, j], nqp[:, j], ts_np,  "qpos"),
                (gqv[:, j], lqv[:, j], nqv[:, j], ts_np,  "qvel"),
                (gqa[:, j], lqa[:, j], nqa[:, j], ts_acc, "qacc"),
            ]):
                ax = axes[row, j]
                ax.plot(t_arr, gt_y, color="black", linewidth=1.5, label="GT")
                ax.plot(t_arr, l_y,  color="green", linewidth=1.0, linestyle="--", label="Learned")
                ax.plot(t_arr, n_y,  color="red",   linewidth=1.0, linestyle=":",  label="Nominal")
                ax.set_title(f"J{j} {row_label}", fontsize=9)
                ax.set_xlabel("t (s)", fontsize=7)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel(row_label)
                    if row == 0:
                        ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        stem = os.path.join(traj_dir, "joint_comparison")
        save_Figure(fig, stem)
        plt.close(fig)

        # ── Plot 2: scalar error-over-time ───────────────────────────────────
        l_qpos_err = np.sqrt(np.mean((lqp - gqp) ** 2, axis=1))
        n_qpos_err = np.sqrt(np.mean((nqp - gqp) ** 2, axis=1))
        l_qvel_err = np.sqrt(np.mean((lqv - gqv) ** 2, axis=1))
        n_qvel_err = np.sqrt(np.mean((nqv - gqv) ** 2, axis=1))
        l_qacc_err = np.sqrt(np.mean((lqa - gqa) ** 2, axis=1))
        n_qacc_err = np.sqrt(np.mean((nqa - gqa) ** 2, axis=1))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Trajectory {traj_idx}  —  RMSE Over Time", fontsize=11)

        for ax, (l_err, n_err, t_arr, title) in zip(axes, [
            (l_qpos_err, n_qpos_err, ts_np,  "qpos RMSE"),
            (l_qvel_err, n_qvel_err, ts_np,  "qvel RMSE"),
            (l_qacc_err, n_qacc_err, ts_acc, "qacc RMSE"),
        ]):
            ax.plot(t_arr, l_err, color="green", label="Learned", linewidth=1.5)
            ax.plot(t_arr, n_err, color="red",   label="Nominal", linewidth=1.5, linestyle="--")
            ax.set_title(title); ax.set_xlabel("t (s)"); ax.set_ylabel("RMSE")
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        stem = os.path.join(traj_dir, "error_over_time")
        save_Figure(fig, stem)
        save_Plot_Data({
            "ts": ts_np, "ts_acc": ts_acc,
            "learned_qpos_err": l_qpos_err, "nominal_qpos_err": n_qpos_err,
            "learned_qvel_err": l_qvel_err, "nominal_qvel_err": n_qvel_err,
            "learned_qacc_err": l_qacc_err, "nominal_qacc_err": n_qacc_err,
        }, stem)
        plt.close(fig)

        # ── Plot 3: residual contribution per joint ──────────────────────────
        fig, axes = plt.subplots(1, nq, figsize=(max(3 * nq, 10), 4), squeeze=False)
        fig.suptitle(
            f"Trajectory {traj_idx}  —  Residual Acceleration Contribution\n"
            f"(purple = learned − nominal,  dashed = GT − nominal)", fontsize=11
        )
        for j in range(nq):
            ax = axes[0, j]
            ax.plot(ts_acc, residual_acc[:, j],       color="purple", linewidth=1.5, label="L − N")
            ax.plot(ts_acc, gqa[:, j] - nqa[:, j],    color="black",  linewidth=1.0, linestyle="--", label="GT − N")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
            ax.set_title(f"Joint {j}", fontsize=9)
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("Δ qacc"); ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        stem = os.path.join(traj_dir, "residual_contribution")
        save_Figure(fig, stem)
        save_Plot_Data({
            "ts_acc": ts_acc,
            **{f"residual_j{j}":      residual_acc[:, j]          for j in range(nq)},
            **{f"gt_minus_nom_j{j}":  gqa[:, j] - nqa[:, j]       for j in range(nq)},
        }, stem)
        plt.close(fig)

        # ── Plot 4: per-joint absolute error heatmaps ────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle(f"Trajectory {traj_idx}  —  Absolute Error  |pred − GT|", fontsize=11)

        for ax, (pred, gt_y, title) in zip(axes.flat, [
            (lqp,  gqp,  "Learned qpos"),
            (nqp,  gqp,  "Nominal qpos"),
            (lqv,  gqv,  "Learned qvel"),
            (nqv,  gqv,  "Nominal qvel"),
        ]):
            err_mat = np.abs(pred - gt_y).T        # (nq, T)
            im = ax.imshow(err_mat, aspect="auto", origin="lower",
                           extent=[ts_np[0], ts_np[-1], -0.5, nq - 0.5],
                           cmap="hot_r")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t (s)"); ax.set_ylabel("Joint Index")
            ax.set_yticks(range(nq))
            plt.colorbar(im, ax=ax, shrink=0.85)

        plt.tight_layout()
        stem = os.path.join(traj_dir, "error_heatmap")
        save_Figure(fig, stem)
        save_Plot_Data({
            "ts": ts_np,
            "learned_qpos_abs_err": np.abs(lqp - gqp).T,
            "nominal_qpos_abs_err": np.abs(nqp - gqp).T,
            "learned_qvel_abs_err": np.abs(lqv - gqv).T,
            "nominal_qvel_abs_err": np.abs(nqv - gqv).T,
        }, stem)
        plt.close(fig)

        # ── Save raw trajectory arrays ───────────────────────────────────────
        np.savez(
            os.path.join(traj_dir, "trajectory_data.npz"),
            ts=ts_np, ts_acc=ts_acc,
            gt_qpos=gqp, gt_qvel=gqv, gt_qacc=gqa,
            learned_qpos=lqp, learned_qvel=lqv, learned_qacc=lqa,
            nominal_qpos=nqp, nominal_qvel=nqv, nominal_qacc=nqa,
            residual_acc=residual_acc,
            ctrl=np.array(ctrl),
        )

        print(f"    Saved to: {traj_dir}")

    print(f"  Visualize job saved to: {job_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    test_path    = return_Arguments()
    test_options = parse_Specs(test_path)

    # ── Load model ────────────────────────────────────────────────────────
    model_type = test_options["model_type"]
    if model_type == "old":
        model, mjx_data, mjx_model = load_Old_Model(test_options)
    elif model_type == "new":
        model, mjx_data, mjx_model = load_New_Model(test_options)
    elif model_type == "sindy":
        model, mjx_data, mjx_model = load_SINDy_Model(test_options)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    # ── Build output directory ────────────────────────────────────────────
    save_name = get_New_Name(test_options["save_name"])
    save_path = os.path.join(os.getcwd(), "Test_Outputs", save_name)
    os.makedirs(save_path, exist_ok=True)

    # Archive the spec used for this run so results are self-contained
    with open(os.path.join(save_path, "test_options.json"), "w") as f:
        json.dump(test_options, f, indent=2)

    # ── Run jobs ──────────────────────────────────────────────────────────
    for job_name, job_options in test_options["jobs"].items():
        job_type = job_options["type"]
        print(f"\n{'━' * 54}")
        print(f"  Job: {job_name}  (type: {job_type})")
        print(f"{'━' * 54}")

        if job_type == "statistics":
            run_Statistics_Job(model, mjx_data, mjx_model, job_options, job_name, save_path)

        elif job_type == "visualize":
            run_Visualize_Job(model, mjx_data, mjx_model, job_options, job_name, save_path)

        else:
            print(f"  Unknown job type '{job_type}' – skipping.")

    print(f"\nAll jobs complete.")
    print(f"Model loaded : {test_options['load_name']}")
    print(f"Results saved: Test_Outputs/{save_name}\n")


if __name__ == "__main__":
    main()
