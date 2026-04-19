import os
import argparse
import equinox as eqx
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import matplotlib
matplotlib.use("Agg")   # non-interactive — swap to "TkAgg" / "Qt5Agg" for live windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mujoco.mjx as mjx
import numpy as np
import time
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils_C2 as mj_utils
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
    
    
def load_Fourier_Model(test_options):
    from Core.Fourier_Modules import KNODE

    training_options = get_Training_Options(test_options)
    model_options    = get_Model_Options(test_options)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root         = zarr.open(dataset_path)
    qvel_max     = jnp.array(root.attrs["dataset_options"]["qvel_max"])

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    key   = jr.key(int(time.time()))
    model = KNODE(key, mjx_model, model_options["num_fourier_features"], model_options["coeff_std"], qvel_max)

    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data, mjx_model
    
    
def load_FSINDy_Model(test_options):
    from Core.Fourier_SINDy_Modules    import Fourier_SINDy_Residual, KNODE_Fourier_SINDy
    from Core.Train_Fourier_SINDy  import build_Fourier_SINDy_Library

    training_options = get_Training_Options(test_options)

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    library        = build_Fourier_SINDy_Library(training_options, mjx_model)
    sindy_residual = Fourier_SINDy_Residual(
        library, mjx_model.nq,
        jnp.zeros((library.n_library_features, mjx_model.nq))
    )
    model = KNODE_Fourier_SINDy(mjx_model, sindy_residual)

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
    fig.savefig(path_no_ext + ".png", dpi=150, bbox_inches="tight")


def save_Plot_Data(data_dict, path_no_ext):
    """Save the raw arrays behind a figure so the plot can be rebuilt later."""
    np.savez(path_no_ext + "_data.npz", **data_dict)


def json_safe(v):
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
    Parameters
    ----------
    pred, gt : array_like  (N, T, D)
    label    : str   prefix for all dict keys

    Returns
    -------
    scalars : dict of float
    arrays  : dict of ndarray   — per-joint and per-timestep breakdowns
    """
    pred  = np.asarray(pred)
    gt    = np.asarray(gt)
    err   = pred - gt
    sq    = err ** 2
    ab    = np.abs(err)

    scalars = {
        f"{label}_rmse":         float(np.sqrt(np.mean(sq))),
        f"{label}_mae":          float(np.mean(ab)),
        f"{label}_max_err":      float(np.max(ab)),
        f"{label}_rmse_final_t": float(np.sqrt(np.mean(sq[:, -1, :]))),
    }
    arrays = {
        f"{label}_rmse_per_joint":     np.sqrt(np.mean(sq, axis=(0, 1))),  # (D,)
        f"{label}_rmse_per_timestep":  np.sqrt(np.mean(sq, axis=(0, 2))),  # (T,)
        # per-trajectory per-joint RMSE kept for distribution jobs
        f"{label}_rmse_per_traj_joint": np.sqrt(np.mean(sq, axis=1)),      # (N, D)
    }
    return scalars, arrays


def improvement_pct(learned_rmse, nominal_rmse):
    return float((nominal_rmse - learned_rmse) / (nominal_rmse + 1e-12) * 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH COLLECTION HELPER   (shared by several jobs)
# ══════════════════════════════════════════════════════════════════════════════

def collect_Batch_Predictions(model, mjx_data, root, keys, ts, subset_size, batch_size):
    """
    Run learned and nominal predictions over `subset_size` trajectories in batches.

    Returns six np arrays, each (N, T, D):
        learned_qpos, learned_qvel, nominal_qpos, nominal_qvel, gt_qpos, gt_qvel
    """
    if subset_size % batch_size != 0:
        raise ValueError("Subset size must be divisible by batch size.")

    lqp_list, lqv_list, nqp_list, nqv_list, gqp_list, gqv_list = [], [], [], [], [], []

    for batch in range(subset_size // batch_size):
        gqp_b, gqv_b, ctrl_b = [], [], []
        for idx in range(batch * batch_size, (batch + 1) * batch_size):
            d = root[keys[idx]]
            gqp_b.append(np.array(d["qpos"][:]))
            gqv_b.append(np.array(d["qvel"][:]))
            ctrl_b.append(np.array(d["ctrl"][0, :]))

        lqp, lqv, nqp, nqv, gqp, gqv = run_Trajectories(
            model, mjx_data, ts,
            jnp.array(np.stack(gqp_b, 0)),
            jnp.array(np.stack(gqv_b, 0)),
            jnp.array(np.stack(ctrl_b, 0)),
        )
        lqp_list.append(np.array(lqp)); lqv_list.append(np.array(lqv))
        nqp_list.append(np.array(nqp)); nqv_list.append(np.array(nqv))
        gqp_list.append(np.array(gqp)); gqv_list.append(np.array(gqv))
        print(f"    Batch {batch + 1}/{subset_size // batch_size} complete.")

    return (
        np.concatenate(lqp_list, 0), np.concatenate(lqv_list, 0),
        np.concatenate(nqp_list, 0), np.concatenate(nqv_list, 0),
        np.concatenate(gqp_list, 0), np.concatenate(gqv_list, 0),
    )


def open_Dataset(job_options):
    root      = zarr.open(os.getcwd() + "/Datasets/" + job_options["dataset_name"] + ".zarr", mode="r")
    keys      = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts        = jnp.arange(0, step_size * num_steps, step=step_size)
    return root, keys, ts, float(step_size)


def resolve_Subset(job_options, keys):
    raw = job_options.get("subset_size", "None")
    if raw == "None" or raw is None:
        return len(keys)
    if raw > len(keys):
        raise ValueError("Subset size cannot exceed dataset size.")
    return int(raw)


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS JOB
# ══════════════════════════════════════════════════════════════════════════════

def run_Statistics_Job(model, mjx_data, mjx_model, job_options, job_name, save_path):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, step_size = open_Dataset(job_options)
    subset_size = resolve_Subset(job_options, keys)
    batch_size  = job_options["batch_size"]

    lqp, lqv, nqp, nqv, gqp, gqv = collect_Batch_Predictions(
        model, mjx_data, root, keys, ts, subset_size, batch_size
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    lqp_sc, lqp_ar = compute_Metrics(lqp, gqp, "learned_qpos")
    nqp_sc, nqp_ar = compute_Metrics(nqp, gqp, "nominal_qpos")
    lqv_sc, lqv_ar = compute_Metrics(lqv, gqv, "learned_qvel")
    nqv_sc, nqv_ar = compute_Metrics(nqv, gqv, "nominal_qvel")

    dt = step_size
    lqa = (lqv[:, 1:] - lqv[:, :-1]) / dt
    nqa = (nqv[:, 1:] - nqv[:, :-1]) / dt
    gqa = (gqv[:, 1:] - gqv[:, :-1]) / dt
    lqa_sc, lqa_ar = compute_Metrics(lqa, gqa, "learned_qacc")
    nqa_sc, nqa_ar = compute_Metrics(nqa, gqa, "nominal_qacc")

    # ── Body-space metrics (optional) ─────────────────────────────────────
    xpos_scalars = {}
    if mjx_model is not None:
        try:
            def _get_xpos(qp, qv):
                d = mjx_data.replace(qpos=qp, qvel=qv)
                return mjx.com_pos(mjx_model, mjx.kinematics(mjx_model, d)).xpos

            _batch = jax.vmap(jax.vmap(_get_xpos, (0, 0)), (0, 0))
            lxp = np.array(_batch(lqp, lqv)).reshape(lqp.shape[0], lqp.shape[1], -1)
            nxp = np.array(_batch(nqp, nqv)).reshape(nqp.shape[0], nqp.shape[1], -1)
            gxp = np.array(_batch(gqp, gqv)).reshape(gqp.shape[0], gqp.shape[1], -1)

            lx_sc, _ = compute_Metrics(lxp, gxp, "learned_xpos")
            nx_sc, _ = compute_Metrics(nxp, gxp, "nominal_xpos")
            xpos_scalars = {
                **lx_sc, **nx_sc,
                "xpos_improvement_pct": improvement_pct(lx_sc["learned_xpos_rmse"],
                                                        nx_sc["nominal_xpos_rmse"]),
            }
        except Exception as e:
            print(f"    Warning: body-space metrics failed: {e}")

    # ── Console summary ───────────────────────────────────────────────────
    N, T, nq = lqp.shape
    print("\n" + "═"*54)
    print("  STATISTICS SUMMARY")
    print("═"*54)
    print(f"  Trajectories: {N}   Timesteps: {T}   DOF: {nq}")
    print("─"*54)
    for name, lsc, nsc in [
        ("qpos", lqp_sc, nqp_sc), ("qvel", lqv_sc, nqv_sc), ("qacc", lqa_sc, nqa_sc),
    ]:
        impr = improvement_pct(lsc[f"learned_{name}_rmse"], nsc[f"nominal_{name}_rmse"])
        print(f"  {name:<6}  RMSE  learned={lsc[f'learned_{name}_rmse']:.5f}  "
              f"nominal={nsc[f'nominal_{name}_rmse']:.5f}  Δ={impr:+.1f}%")
        print(f"         final  learned={lsc[f'learned_{name}_rmse_final_t']:.5f}  "
              f"nominal={nsc[f'nominal_{name}_rmse_final_t']:.5f}")
    if xpos_scalars:
        print(f"  xpos   RMSE  learned={xpos_scalars['learned_xpos_rmse']:.5f}  "
              f"nominal={xpos_scalars['nominal_xpos_rmse']:.5f}  "
              f"Δ={xpos_scalars['xpos_improvement_pct']:+.1f}%")
    print("═"*54 + "\n")

    if not should_save:
        return

    # ── Save ──────────────────────────────────────────────────────────────
    summary = {
        **lqp_sc, **nqp_sc, **lqv_sc, **nqv_sc, **lqa_sc, **nqa_sc,
        **xpos_scalars,
        "qpos_improvement_pct": improvement_pct(lqp_sc["learned_qpos_rmse"], nqp_sc["nominal_qpos_rmse"]),
        "qvel_improvement_pct": improvement_pct(lqv_sc["learned_qvel_rmse"], nqv_sc["nominal_qvel_rmse"]),
        "qacc_improvement_pct": improvement_pct(lqa_sc["learned_qacc_rmse"], nqa_sc["nominal_qacc_rmse"]),
        "num_trajectories": N, "num_timesteps": T, "dof": nq,
    }
    with open(os.path.join(job_dir, "summary.json"), "w") as f:
        json.dump({k: json_safe(v) for k, v in summary.items()}, f, indent=2)

    ts_np = np.array(ts)
    np.savez(os.path.join(job_dir, "qpos_errors.npz"), ts=ts_np, **lqp_ar, **nqp_ar)
    np.savez(os.path.join(job_dir, "qvel_errors.npz"), ts=ts_np, **lqv_ar, **nqv_ar)
    np.savez(os.path.join(job_dir, "qacc_errors.npz"), ts=ts_np[:-1], **lqa_ar, **nqa_ar)

    _plot_Statistics(lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar, ts_np, job_dir)
    print(f"  Saved to: {job_dir}\n")


def _plot_Statistics(lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar, ts, job_dir):
    plots_dir = make_dir(job_dir, "plots")
    ts_acc    = ts[:-1]

    # ── RMSE vs time ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RMSE vs Time", fontsize=12)
    for ax, (lk, nk, t_arr, title) in zip(axes, [
        ("learned_qpos_rmse_per_timestep", "nominal_qpos_rmse_per_timestep", ts,     "qpos"),
        ("learned_qvel_rmse_per_timestep", "nominal_qvel_rmse_per_timestep", ts,     "qvel"),
        ("learned_qacc_rmse_per_timestep", "nominal_qacc_rmse_per_timestep", ts_acc, "qacc"),
    ]):
        l_ar = lqp_ar if "qpos" in lk else (lqv_ar if "qvel" in lk else lqa_ar)
        n_ar = nqp_ar if "qpos" in nk else (nqv_ar if "qvel" in nk else nqa_ar)
        ax.plot(t_arr, l_ar[lk], color="green", label="Learned", linewidth=1.5)
        ax.plot(t_arr, n_ar[nk], color="red",   label="Nominal", linewidth=1.5, linestyle="--")
        ax.set_title(title); ax.set_xlabel("t (s)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    stem = os.path.join(plots_dir, "rmse_vs_time")
    save_Figure(fig, stem)
    save_Plot_Data({"ts": ts, "ts_acc": ts_acc, **lqp_ar, **nqp_ar, **lqv_ar, **nqv_ar, **lqa_ar, **nqa_ar}, stem)
    plt.close(fig)

    # ── RMSE per joint ────────────────────────────────────────────────────
    nq  = lqp_ar["learned_qpos_rmse_per_joint"].shape[0]
    idx = np.arange(nq)
    w   = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RMSE per Joint", fontsize=12)
    for ax, (lv, nv, title) in zip(axes, [
        (lqp_ar["learned_qpos_rmse_per_joint"], nqp_ar["nominal_qpos_rmse_per_joint"], "qpos"),
        (lqv_ar["learned_qvel_rmse_per_joint"], nqv_ar["nominal_qvel_rmse_per_joint"], "qvel"),
        (lqa_ar["learned_qacc_rmse_per_joint"], nqa_ar["nominal_qacc_rmse_per_joint"], "qacc"),
    ]):
        ax.bar(idx - w/2, lv, w, label="Learned", color="green", alpha=0.75)
        ax.bar(idx + w/2, nv, w, label="Nominal", color="red",   alpha=0.75)
        ax.set_title(title); ax.set_xlabel("Joint"); ax.set_ylabel("RMSE")
        ax.set_xticks(idx); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    stem = os.path.join(plots_dir, "rmse_per_joint")
    save_Figure(fig, stem)
    save_Plot_Data({
        "joint_idx": idx,
        **{k: v for d in [lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar]
           for k, v in d.items() if "per_joint" in k and "traj" not in k},
    }, stem)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZE JOB
# ══════════════════════════════════════════════════════════════════════════════

def run_Visualize_Job(model, mjx_data, mjx_model, job_options, job_name, save_path, test_options):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, step_size = open_Dataset(job_options)
    ts_np = np.array(ts)
    dt    = step_size

    for traj_idx in job_options["which_to_visualize"]:
        print(f"  Trajectory {traj_idx} ...")
        d = root[keys[traj_idx]]

        gt_qpos = jnp.array(d["qpos"][:])
        gt_qvel = jnp.array(d["qvel"][:])
        ctrl    = jnp.array(d["ctrl"][0, :])

        init_states  = {"qpos": gt_qpos[0], "qvel": gt_qvel[0]}
        learned_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
        nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))

        lqp = np.array(learned_traj["qpos"]);  lqv = np.array(learned_traj["qvel"])
        nqp = np.array(nominal_traj["qpos"]);  nqv = np.array(nominal_traj["qvel"])
        gqp = np.array(gt_qpos);               gqv = np.array(gt_qvel)

        lqa   = (lqv[1:] - lqv[:-1]) / dt
        nqa   = (nqv[1:] - nqv[:-1]) / dt
        gqa   = (gqv[1:] - gqv[:-1]) / dt
        ts_acc = ts_np[:-1]
        res_acc = lqa - nqa
        nq = gqp.shape[1]

        if not should_save:
            continue

        traj_dir = make_dir(job_dir, f"trajectory_{traj_idx}")

        # ── Plot 1: per-joint state comparison ───────────────────────────────
        fig, axes = plt.subplots(3, nq, figsize=(max(3*nq, 10), 10), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Per-Joint State\n"
                     "(black=GT  green=Learned  red=Nominal)", fontsize=11)
        for j in range(nq):
            for row, (gy, ly, ny, t_arr, yl) in enumerate([
                (gqp[:, j], lqp[:, j], nqp[:, j], ts_np,  "qpos"),
                (gqv[:, j], lqv[:, j], nqv[:, j], ts_np,  "qvel"),
                (gqa[:, j], lqa[:, j], nqa[:, j], ts_acc, "qacc"),
            ]):
                ax = axes[row, j]
                ax.plot(t_arr, gy, "k",  linewidth=1.5)
                ax.plot(t_arr, ly, "g--",linewidth=1.0)
                ax.plot(t_arr, ny, "r:", linewidth=1.0)
                ax.set_title(f"J{j} {yl}", fontsize=9)
                ax.set_xlabel("t", fontsize=7); ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel(yl)
                    if row == 0:
                        ax.legend(["GT","Learned","Nominal"], fontsize=6)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_Figure(fig, os.path.join(traj_dir, "joint_comparison"))
        plt.close(fig)

        # ── Plot 2: scalar error over time ───────────────────────────────────
        le_p = np.sqrt(np.mean((lqp-gqp)**2, 1));  ne_p = np.sqrt(np.mean((nqp-gqp)**2, 1))
        le_v = np.sqrt(np.mean((lqv-gqv)**2, 1));  ne_v = np.sqrt(np.mean((nqv-gqv)**2, 1))
        le_a = np.sqrt(np.mean((lqa-gqa)**2, 1));  ne_a = np.sqrt(np.mean((nqa-gqa)**2, 1))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Trajectory {traj_idx}  —  RMSE Over Time", fontsize=11)
        for ax, (le, ne, t_arr, title) in zip(axes, [
            (le_p, ne_p, ts_np,  "qpos"), (le_v, ne_v, ts_np, "qvel"),
            (le_a, ne_a, ts_acc, "qacc"),
        ]):
            ax.plot(t_arr, le, "g", label="Learned", linewidth=1.5)
            ax.plot(t_arr, ne, "r--", label="Nominal", linewidth=1.5)
            ax.set_title(title); ax.set_xlabel("t"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        stem = os.path.join(traj_dir, "error_over_time")
        save_Figure(fig, stem)
        save_Plot_Data({"ts": ts_np, "ts_acc": ts_acc,
                        "le_p": le_p, "ne_p": ne_p, "le_v": le_v,
                        "ne_v": ne_v, "le_a": le_a, "ne_a": ne_a}, stem)
        plt.close(fig)

        # ── Plot 3: residual contribution ────────────────────────────────────
        fig, axes = plt.subplots(1, nq, figsize=(max(3*nq, 10), 4), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Residual Acceleration  "
                     "(purple=L−N  dashed=GT−N)", fontsize=11)
        for j in range(nq):
            ax = axes[0, j]
            ax.plot(ts_acc, res_acc[:, j],          color="purple", linewidth=1.5)
            ax.plot(ts_acc, gqa[:, j]-nqa[:, j],    "k--", linewidth=1.0)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(f"J{j}", fontsize=9); ax.set_xlabel("t", fontsize=7)
            if j == 0: ax.set_ylabel("Δacc"); ax.legend(["L−N","GT−N"], fontsize=6)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        stem = os.path.join(traj_dir, "residual_contribution")
        save_Figure(fig, stem)
        save_Plot_Data({"ts_acc": ts_acc, "res_acc": res_acc,
                        "gt_minus_nom": gqa - nqa}, stem)
        plt.close(fig)

        # ── Plot 4: absolute error heatmaps ──────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle(f"Trajectory {traj_idx}  —  |pred − GT|  heatmap", fontsize=11)
        for ax, (pred, gt_y, title) in zip(axes.flat, [
            (lqp, gqp, "Learned qpos"), (nqp, gqp, "Nominal qpos"),
            (lqv, gqv, "Learned qvel"), (nqv, gqv, "Nominal qvel"),
        ]):
            im = ax.imshow(np.abs(pred-gt_y).T, aspect="auto", origin="lower",
                           extent=[ts_np[0], ts_np[-1], -0.5, nq-0.5], cmap="hot_r")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t"); ax.set_ylabel("Joint")
            ax.set_yticks(range(nq)); plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        stem = os.path.join(traj_dir, "error_heatmap")
        save_Figure(fig, stem)
        save_Plot_Data({"ts": ts_np,
                        "lqp_err": np.abs(lqp-gqp).T, "nqp_err": np.abs(nqp-gqp).T,
                        "lqv_err": np.abs(lqv-gqv).T, "nqv_err": np.abs(nqv-gqv).T}, stem)
        plt.close(fig)

        # ── Raw data ─────────────────────────────────────────────────────────
        np.savez(os.path.join(traj_dir, "trajectory_data.npz"),
                 ts=ts_np, ts_acc=ts_acc,
                 gt_qpos=gqp, gt_qvel=gqv, gt_qacc=gqa,
                 learned_qpos=lqp, learned_qvel=lqv, learned_qacc=lqa,
                 nominal_qpos=nqp, nominal_qvel=nqv, nominal_qacc=nqa,
                 residual_acc=res_acc, ctrl=np.array(ctrl))

        # ── MuJoCo comparison video ───────────────────────────────────────────
        model_name = test_options.get("model_name", None)
        if model_name is not None:
            try:
                video_fps      = job_options.get("video_fps", 50)
                playback_speed = job_options.get("video_playback_speed", 1.0)
                mj_utils.render_GT_vs_Nominal_vs_KNODE_Video(
                    model_name, gqp, nqp, lqp,
                    save_path=os.path.join(traj_dir, "comparison_video"),
                    fps=video_fps,
                    playback_speed=playback_speed,
                )
            except Exception as e:
                print(f"    Warning: video render failed: {e}")

        print(f"    Saved → {traj_dir}")

    if should_save:
        print(f"  Visualize job saved to: {job_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# NEW JOB: PHASE PORTRAIT
# ══════════════════════════════════════════════════════════════════════════════
#
# For each joint, plot the trajectory in (qpos, qvel) phase space, with the
# time dimension encoded as colour.  Overlaying several trajectories on the
# same axes immediately reveals whether the learned model reproduces the
# correct attractors, orbits, and energy-dissipation structure.
#
# Expected job_options fields:
#   dataset_name  : str
#   subset_size   : int | "None"   — how many trajectories to overlay
#   which_joints  : list[int]|null — null → all joints
#   should_save   : bool
#
# Example JSON:
#   "phase_portrait": {
#       "type": "phase_portrait",
#       "dataset_name": "dataset_3_fluid_no_coll_valid",
#       "subset_size": 20,
#       "which_joints": null,
#       "should_save": true
#   }
# ══════════════════════════════════════════════════════════════════════════════

def run_Phase_Portrait_Job(model, mjx_data, job_options, job_name, save_path):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, _ = open_Dataset(job_options)
    ts_np       = np.array(ts)
    subset_size = resolve_Subset(job_options, keys)
    # Clamp for visual clarity — too many trajectories makes a muddy plot
    subset_size = min(subset_size, 50)

    which_joints = job_options.get("which_joints", None)

    # Collect individual trajectories (no need to batch, subset is small)
    gt_qpos_all, gt_qvel_all = [], []
    lqp_all, lqv_all, nqp_all, nqv_all = [], [], [], []

    for idx in range(subset_size):
        d    = root[keys[idx]]
        gqp  = jnp.array(d["qpos"][:])
        gqv  = jnp.array(d["qvel"][:])
        ctrl = jnp.array(d["ctrl"][0, :])

        init = {"qpos": gqp[0], "qvel": gqv[0]}
        lt   = model(ts, mjx_data, init, ctrl, with_learned=jnp.array(1.0))
        nt   = model(ts, mjx_data, init, ctrl, with_learned=jnp.array(0.0))

        gt_qpos_all.append(np.array(gqp));  gt_qvel_all.append(np.array(gqv))
        lqp_all.append(np.array(lt["qpos"])); lqv_all.append(np.array(lt["qvel"]))
        nqp_all.append(np.array(nt["qpos"])); nqv_all.append(np.array(nt["qvel"]))

        if (idx + 1) % 5 == 0 or idx == subset_size - 1:
            print(f"    {idx + 1}/{subset_size} trajectories collected.")

    nq     = gt_qpos_all[0].shape[1]
    joints = which_joints if which_joints is not None else list(range(nq))

    fig, axes = plt.subplots(3, len(joints), figsize=(4 * len(joints), 12), squeeze=False)
    fig.suptitle("Phase Portraits  (qvel vs qpos)\n"
                 "Colour encodes time  (blue=start → yellow=end)\n"
                 "Row 1: GT   Row 2: Learned   Row 3: Nominal", fontsize=11)

    cmap = cm.plasma

    for col, j in enumerate(joints):
        for row, (qpos_list, qvel_list, row_label) in enumerate([
            (gt_qpos_all, gt_qvel_all, "GT"),
            (lqp_all,     lqv_all,     "Learned"),
            (nqp_all,     nqv_all,     "Nominal"),
        ]):
            ax = axes[row, col]
            for qpos_traj, qvel_traj in zip(qpos_list, qvel_list):
                T = qpos_traj.shape[0]
                colours = cmap(np.linspace(0, 1, T - 1))
                for t in range(T - 1):
                    ax.plot(
                        qpos_traj[t:t+2, j],
                        qvel_traj[t:t+2, j],
                        color=colours[t],
                        linewidth=0.7,
                        alpha=0.7,
                    )
            ax.set_title(f"J{j}  {row_label}", fontsize=9)
            ax.set_xlabel("qpos", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nqvel", fontsize=8)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=7)

    # Shared colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=ts_np[0], vmax=ts_np[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Time (s)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.92, 1])

    if not should_save:
        plt.close(fig)
        return

    stem = os.path.join(job_dir, "phase_portrait")
    save_Figure(fig, stem)
    save_Plot_Data({
        "ts": ts_np,
        "gt_qpos":      np.stack(gt_qpos_all, 0),
        "gt_qvel":      np.stack(gt_qvel_all, 0),
        "learned_qpos": np.stack(lqp_all, 0),
        "learned_qvel": np.stack(lqv_all, 0),
        "nominal_qpos": np.stack(nqp_all, 0),
        "nominal_qvel": np.stack(nqv_all, 0),
        "which_joints": np.array(joints),
    }, stem)
    plt.close(fig)
    print(f"  Phase portrait saved to: {job_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# NEW JOB: ERROR DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# Violin + strip plots of per-trajectory, per-joint RMSE across the whole test
# set.  Unlike the scalar statistics summary, this shows the *spread*:
# whether the learned model is consistently better, or only on easy
# trajectories.  A narrow violin centred near zero signals genuine improvement;
# a wide, overlapping one suggests the model is inconsistent.
#
# Expected job_options fields:
#   dataset_name, subset_size, batch_size, should_save
#
# Example JSON:
#   "error_distribution": {
#       "type": "error_distribution",
#       "dataset_name": "dataset_3_fluid_no_coll_valid",
#       "subset_size": 200,
#       "batch_size": 100,
#       "should_save": true
#   }
# ══════════════════════════════════════════════════════════════════════════════

def run_Error_Distribution_Job(model, mjx_data, job_options, job_name, save_path):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, step_size = open_Dataset(job_options)
    subset_size = resolve_Subset(job_options, keys)
    batch_size  = job_options["batch_size"]

    lqp, lqv, nqp, nqv, gqp, gqv = collect_Batch_Predictions(
        model, mjx_data, root, keys, ts, subset_size, batch_size
    )

    # Per-trajectory, per-joint RMSE  →  (N, nq)
    lqp_ptj = np.sqrt(np.mean((lqp - gqp)**2, axis=1))  # mean over time
    nqp_ptj = np.sqrt(np.mean((nqp - gqp)**2, axis=1))
    lqv_ptj = np.sqrt(np.mean((lqv - gqv)**2, axis=1))
    nqv_ptj = np.sqrt(np.mean((nqv - gqv)**2, axis=1))

    # Improvement per trajectory per joint
    impr_qpos = (nqp_ptj - lqp_ptj) / (nqp_ptj + 1e-12) * 100.0  # (N, nq)
    impr_qvel = (nqv_ptj - lqv_ptj) / (nqv_ptj + 1e-12) * 100.0

    N, nq = lqp_ptj.shape

    print(f"\n  Error Distribution  ({N} trajectories, {nq} joints)")
    print(f"  Median qpos improvement per joint: "
          + "  ".join(f"J{j}: {np.median(impr_qpos[:, j]):+.1f}%" for j in range(nq)))
    print(f"  Median qvel improvement per joint: "
          + "  ".join(f"J{j}: {np.median(impr_qvel[:, j]):+.1f}%" for j in range(nq)))

    if not should_save:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Per-Trajectory Error Distribution  (N={N} trajectories)\n"
                 "Green=Learned  Red=Nominal", fontsize=12)

    joint_labels = [f"J{j}" for j in range(nq)]
    xs           = np.arange(nq)

    for ax, (l_data, n_data, ylabel, title) in zip(axes.flat, [
        (lqp_ptj, nqp_ptj, "RMSE",          "qpos RMSE per Trajectory"),
        (lqv_ptj, nqv_ptj, "RMSE",          "qvel RMSE per Trajectory"),
        (impr_qpos, None,  "Improvement %", "qpos Improvement % (pos=better)"),
        (impr_qvel, None,  "Improvement %", "qvel Improvement % (pos=better)"),
    ]):
        if n_data is not None:
            # Paired violins for learned vs nominal
            vp_l = ax.violinplot([l_data[:, j] for j in range(nq)],
                                 positions=xs - 0.2, widths=0.35,
                                 showmedians=True, showextrema=False)
            vp_n = ax.violinplot([n_data[:, j] for j in range(nq)],
                                 positions=xs + 0.2, widths=0.35,
                                 showmedians=True, showextrema=False)
            for pc in vp_l["bodies"]: pc.set_facecolor("green"); pc.set_alpha(0.55)
            for pc in vp_n["bodies"]: pc.set_facecolor("red");   pc.set_alpha(0.55)
            vp_l["cmedians"].set_color("darkgreen")
            vp_n["cmedians"].set_color("darkred")
            # Raw strip
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, (N, nq))
            for j in range(nq):
                ax.scatter(xs[j] - 0.2 + jitter[:, j], l_data[:, j],
                           s=2, color="green", alpha=0.25)
                ax.scatter(xs[j] + 0.2 + jitter[:, j], n_data[:, j],
                           s=2, color="red",   alpha=0.25)
        else:
            # Single improvement violin
            vp = ax.violinplot([l_data[:, j] for j in range(nq)],
                               positions=xs, widths=0.6,
                               showmedians=True, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor("steelblue"); pc.set_alpha(0.6)
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
            jitter = np.random.default_rng(1).uniform(-0.08, 0.08, (N, nq))
            for j in range(nq):
                c = "green" if np.median(l_data[:, j]) > 0 else "red"
                ax.scatter(xs[j] + jitter[:, j], l_data[:, j],
                           s=2, color=c, alpha=0.25)

        ax.set_xticks(xs); ax.set_xticklabels(joint_labels)
        ax.set_xlabel("Joint"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    stem = os.path.join(job_dir, "error_distribution")
    save_Figure(fig, stem)
    save_Plot_Data({
        "learned_qpos_rmse_per_traj_joint": lqp_ptj,
        "nominal_qpos_rmse_per_traj_joint": nqp_ptj,
        "learned_qvel_rmse_per_traj_joint": lqv_ptj,
        "nominal_qvel_rmse_per_traj_joint": nqv_ptj,
        "improvement_qpos_per_traj_joint":  impr_qpos,
        "improvement_qvel_per_traj_joint":  impr_qvel,
    }, stem)
    plt.close(fig)
    print(f"  Error distribution saved to: {job_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# NEW JOB: FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
#
# Computes the power spectral density (PSD) of the position and velocity error
# signals averaged across trajectories.  For a robot immersed in fluid, the
# dominant corrections should appear at low frequencies (drag, buoyancy) rather
# than high ones (noise).  If the learned model's error PSD is significantly
# lower than the nominal's at those frequencies, it is capturing the right
# physics.  The plot also shows the PSD of the *residual signal*
# (learned_qacc − nominal_qacc) to visualise what the network actively outputs.
#
# Expected job_options fields:
#   dataset_name, subset_size, batch_size, should_save
#
# Example JSON:
#   "frequency_analysis": {
#       "type": "frequency_analysis",
#       "dataset_name": "dataset_3_fluid_no_coll_valid",
#       "subset_size": 200,
#       "batch_size": 100,
#       "should_save": true
#   }
# ══════════════════════════════════════════════════════════════════════════════

def run_Frequency_Analysis_Job(model, mjx_data, job_options, job_name, save_path):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, step_size = open_Dataset(job_options)
    subset_size = resolve_Subset(job_options, keys)
    batch_size  = job_options["batch_size"]

    lqp, lqv, nqp, nqv, gqp, gqv = collect_Batch_Predictions(
        model, mjx_data, root, keys, ts, subset_size, batch_size
    )

    dt   = step_size
    N, T, nq = lqp.shape
    freqs = np.fft.rfftfreq(T, d=dt)   # Hz

    def mean_psd(signal):
        """Mean PSD across trajectories; shape (F, nq)."""
        # signal: (N, T, nq)
        F   = np.fft.rfft(signal, axis=1)         # (N, F, nq)
        psd = (np.abs(F)**2) / T                   # power per bin
        return np.mean(psd, axis=0)                # (F, nq)

    lqp_err  = lqp - gqp    # (N, T, nq)
    nqp_err  = nqp - gqp
    lqv_err  = lqv - gqv
    nqv_err  = nqv - gqv

    lqp_psd  = mean_psd(lqp_err)
    nqp_psd  = mean_psd(nqp_err)
    lqv_psd  = mean_psd(lqv_err)
    nqv_psd  = mean_psd(nqv_err)

    # Residual acceleration signal
    lqa = (lqv[:, 1:] - lqv[:, :-1]) / dt
    nqa = (nqv[:, 1:] - nqv[:, :-1]) / dt
    res_acc = lqa - nqa   # (N, T-1, nq)
    res_freqs = np.fft.rfftfreq(T - 1, d=dt)
    res_psd   = mean_psd(res_acc)   # (F', nq)

    print(f"\n  Frequency Analysis  ({N} trajectories, {nq} joints, "
          f"fs={1/dt:.1f} Hz, Nyquist={0.5/dt:.1f} Hz)")

    if not should_save:
        return

    fig, axes = plt.subplots(3, nq, figsize=(4 * nq, 12), squeeze=False)
    fig.suptitle("Power Spectral Density of Error Signals\n"
                 "(green=Learned, red=Nominal)  log–log scale", fontsize=12)

    eps = 1e-30   # floor to avoid log(0)

    for j in range(nq):
        # ── qpos error PSD ─────────────────────────────────────────────────
        ax = axes[0, j]
        ax.loglog(freqs[1:], lqp_psd[1:, j] + eps, "g",  label="Learned", linewidth=1.5)
        ax.loglog(freqs[1:], nqp_psd[1:, j] + eps, "r--",label="Nominal", linewidth=1.5)
        ax.set_title(f"J{j}  qpos error", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        ax.set_ylabel("PSD", fontsize=7)
        if j == 0: ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

        # ── qvel error PSD ─────────────────────────────────────────────────
        ax = axes[1, j]
        ax.loglog(freqs[1:], lqv_psd[1:, j] + eps, "g",  linewidth=1.5)
        ax.loglog(freqs[1:], nqv_psd[1:, j] + eps, "r--",linewidth=1.5)
        ax.set_title(f"J{j}  qvel error", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

        # ── Residual acceleration PSD ──────────────────────────────────────
        ax = axes[2, j]
        ax.loglog(res_freqs[1:], res_psd[1:, j] + eps, color="purple", linewidth=1.5)
        ax.set_title(f"J{j}  residual acc", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
        if j == 0:
            ax.set_ylabel("PSD (residual acc)", fontsize=7)

    row_labels = ["qpos error", "qvel error", "residual acc"]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, xy=(0, 0.5), xytext=(-40, 0),
                              xycoords="axes fraction", textcoords="offset points",
                              va="center", ha="right", rotation=90, fontsize=9,
                              fontweight="bold")

    plt.tight_layout()
    stem = os.path.join(job_dir, "frequency_analysis")
    save_Figure(fig, stem)
    save_Plot_Data({
        "freqs": freqs, "res_freqs": res_freqs,
        "learned_qpos_psd": lqp_psd, "nominal_qpos_psd": nqp_psd,
        "learned_qvel_psd": lqv_psd, "nominal_qvel_psd": nqv_psd,
        "residual_acc_psd": res_psd,
    }, stem)
    plt.close(fig)
    print(f"  Frequency analysis saved to: {job_dir}\n")


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
    elif model_type == "fourier":
        model, mjx_data, mjx_model = load_Fourier_Model(test_options)
    elif model_type == "fsindy":
        model, mjx_data, mjx_model = load_FSINDy_Model(test_options)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'.")

    # ── Output directory ──────────────────────────────────────────────────
    save_name = get_New_Name(test_options["save_name"])
    save_path = os.path.join(os.getcwd(), "Test_Outputs", save_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "test_options.json"), "w") as f:
        json.dump(test_options, f, indent=2)

    # ── Run jobs ──────────────────────────────────────────────────────────
    for job_name, job_options in test_options["jobs"].items():
        job_type = job_options["type"]
        print(f"\n{'━'*54}")
        print(f"  Job: {job_name}  (type: {job_type})")
        print(f"{'━'*54}")

        if job_type == "statistics":
            run_Statistics_Job(model, mjx_data, mjx_model, job_options, job_name, save_path)

        elif job_type == "visualize":
            run_Visualize_Job(model, mjx_data, mjx_model, job_options, job_name, save_path, test_options)

        elif job_type == "phase_portrait":
            run_Phase_Portrait_Job(model, mjx_data, job_options, job_name, save_path)

        elif job_type == "error_distribution":
            run_Error_Distribution_Job(model, mjx_data, job_options, job_name, save_path)

        elif job_type == "frequency_analysis":
            run_Frequency_Analysis_Job(model, mjx_data, job_options, job_name, save_path)
            

        else:
            print(f"  Unknown job type '{job_type}' — skipping.")

    print(f"\nAll jobs complete.")
    print(f"Model : {test_options['load_name']}")
    print(f"Output: Test_Outputs/{save_name}\n")


if __name__ == "__main__":
    main()
