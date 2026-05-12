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
import mujoco
import mujoco.mjx as mjx
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  — registers 3D projection
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
    parser.add_argument("-t", "--test_specs", type=str, default="example_test_spec")
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
# NQ / NV DIMENSION HELPERS  (free-body → nq = nv + 1)
# ══════════════════════════════════════════════════════════════════════════════

def dims_from_data(lqp, lqv):
    """Return (nq, nv) from trajectory arrays."""
    nq = lqp.shape[-1]
    nv = lqv.shape[-1]
    return nq, nv


def qpos_qvel_joint_map(nq, nv):
    """
    Build a mapping from qvel joint-index to a corresponding qpos index.

    For a system with a free body (nq == nv + 1):
        qpos [0:3]  = position     ↔  qvel [0:3] = linear velocity
        qpos [3:7]  = quaternion   ↔  qvel [3:6] = angular velocity  (no 1-1 qpos pair)
        qpos [7: ]  = arm joints   ↔  qvel [6: ] = arm joint velocities

    Returns
    -------
    mapping : list of length nv
        mapping[v_idx] is the qpos index to pair with, or None for angular-vel
        components that have no scalar qpos partner.
    """
    if nq == nv:
        return list(range(nv))
    if nq == nv + 1:                       # single free body
        m = list(range(3))                 # lin-vel  →  position
        m += [None, None, None]            # ang-vel  →  no scalar partner
        for i in range(nv - 6):            # arm joints
            m.append(7 + i)
        return m
    # generic fallback
    return list(range(min(nq, nv)))


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
    
    
def load_UVMS_Model(test_options):
    from Core.KNODE_Modules_UVMS import KNODE

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
    
    
def load_FSINDy_UVMS_Model(test_options):
    from Core.Fourier_SINDy_Modules_UVMS    import Fourier_SINDy_Residual, KNODE_Fourier_SINDy
    from Core.Train_Fourier_SINDy_UVMS  import build_Fourier_SINDy_Library

    training_options = get_Training_Options(test_options)

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    library        = build_Fourier_SINDy_Library(training_options, mjx_model)
    sindy_residual = Fourier_SINDy_Residual(
        library, mjx_model.nv,
        jnp.zeros((library.n_library_features, mjx_model.nv))
    )
    model = KNODE_Fourier_SINDy(mjx_model, sindy_residual)

    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data, mjx_model


def load_GP_Model(test_options):
    from Core.GP_Modules import GP_Residual, KNODE_GP

    training_options = get_Training_Options(test_options)
    load_path = os.getcwd() + "/Models/" + test_options["load_name"] + "/"

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    # Load saved numpy arrays to recover shapes
    X_train_norm    = jnp.array(np.load(load_path + "X_train_norm.npy"))
    alpha           = jnp.array(np.load(load_path + "alpha.npy"))
    log_lengthscales = jnp.array(np.load(load_path + "log_lengthscales.npy"))
    log_signal_var  = jnp.array(np.load(load_path + "log_signal_var.npy"))
    log_noise_var   = jnp.array(np.load(load_path + "log_noise_var.npy"))

    # Recover normalisation from mjx_model
    dataset_name = training_options.get("sindy_dataset_name",
                                         training_options.get("dataset_name"))
    dataset_path = "Datasets/" + dataset_name + ".zarr"
    root         = zarr.open(dataset_path)
    qvel_max     = jnp.array(root.attrs["dataset_options"]["qvel_max"])

    qpos_ranges = mjx_model.jnt_range
    ctrl_ranges = mjx_model.actuator_ctrlrange
    norm_add  = jnp.concatenate([-qpos_ranges[:, 0],  qvel_max, -ctrl_ranges[:, 0]])
    norm_mult = jnp.concatenate([2.0 / (qpos_ranges[:, 1] - qpos_ranges[:, 0]),
                                  1.0 / qvel_max,
                                  2.0 / (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])])

    output_num = alpha.shape[-1]

    gp_residual = GP_Residual(
        X_train_norm     = X_train_norm,
        input_norm_add   = norm_add,
        input_norm_mult  = norm_mult,
        log_lengthscales = log_lengthscales,
        log_signal_var   = log_signal_var,
        log_noise_var    = log_noise_var,
        alpha            = alpha,
        output_num       = output_num,
    )

    model = KNODE_GP(mjx_model, gp_residual)

    # Overwrite with serialised leaves (catches any additional state)
    model = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
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

    # R² overall
    ss_res = np.sum(sq)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2_overall = float(1.0 - ss_res / (ss_tot + 1e-12))

    # R² per joint  (D,)
    ss_res_j = np.sum(sq, axis=(0, 1))                          # (D,)
    gt_mean_j = np.mean(gt, axis=(0, 1), keepdims=True)         # (1,1,D)
    ss_tot_j = np.sum((gt - gt_mean_j) ** 2, axis=(0, 1))      # (D,)
    r2_per_joint = 1.0 - ss_res_j / (ss_tot_j + 1e-12)         # (D,)

    scalars = {
        f"{label}_rmse":         float(np.sqrt(np.mean(sq))),
        f"{label}_mae":          float(np.mean(ab)),
        f"{label}_max_err":      float(np.max(ab)),
        f"{label}_rmse_final_t": float(np.sqrt(np.mean(sq[:, -1, :]))),
        f"{label}_r2":           r2_overall,
    }
    arrays = {
        f"{label}_rmse_per_joint":     np.sqrt(np.mean(sq, axis=(0, 1))),  # (D,)
        f"{label}_rmse_per_timestep":  np.sqrt(np.mean(sq, axis=(0, 2))),  # (T,)
        f"{label}_r2_per_joint":       r2_per_joint,                       # (D,)
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
    nv = lqv.shape[-1]
    print("\n" + "═"*54)
    print("  STATISTICS SUMMARY")
    print("═"*54)
    print(f"  Trajectories: {N}   Timesteps: {T}   nq: {nq}   nv: {nv}")
    print("─"*54)
    for name, lsc, nsc in [
        ("qpos", lqp_sc, nqp_sc), ("qvel", lqv_sc, nqv_sc), ("qacc", lqa_sc, nqa_sc),
    ]:
        impr = improvement_pct(lsc[f"learned_{name}_rmse"], nsc[f"nominal_{name}_rmse"])
        print(f"  {name:<6}  RMSE  learned={lsc[f'learned_{name}_rmse']:.5f}  "
              f"nominal={nsc[f'nominal_{name}_rmse']:.5f}  Δ={impr:+.1f}%")
        print(f"         final  learned={lsc[f'learned_{name}_rmse_final_t']:.5f}  "
              f"nominal={nsc[f'nominal_{name}_rmse_final_t']:.5f}")
        print(f"         R²    learned={lsc[f'learned_{name}_r2']:.5f}  "
              f"nominal={nsc[f'nominal_{name}_r2']:.5f}")
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
        "num_trajectories": N, "num_timesteps": T, "nq": nq, "nv": nv,
    }
    with open(os.path.join(job_dir, "summary.json"), "w") as f:
        json.dump({k: json_safe(v) for k, v in summary.items()}, f, indent=2)

    ts_np = np.array(ts)
    np.savez(os.path.join(job_dir, "qpos_errors.npz"), ts=ts_np, **lqp_ar, **nqp_ar)
    np.savez(os.path.join(job_dir, "qvel_errors.npz"), ts=ts_np, **lqv_ar, **nqv_ar)
    np.savez(os.path.join(job_dir, "qacc_errors.npz"), ts=ts_np[:-1], **lqa_ar, **nqa_ar)

    _plot_Statistics(lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar, ts_np, job_dir,
                     lqp, nqp, gqp, lqv, nqv, gqv, lqa, nqa, gqa)
    print(f"  Saved to: {job_dir}\n")


def _plot_Statistics(lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar, ts, job_dir,
                     lqp=None, nqp=None, gqp=None, lqv=None, nqv=None, gqv=None,
                     lqa=None, nqa=None, gqa=None):
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
    w   = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RMSE per Joint", fontsize=12)
    for ax, (lv, nv_arr, title) in zip(axes, [
        (lqp_ar["learned_qpos_rmse_per_joint"], nqp_ar["nominal_qpos_rmse_per_joint"], "qpos"),
        (lqv_ar["learned_qvel_rmse_per_joint"], nqv_ar["nominal_qvel_rmse_per_joint"], "qvel"),
        (lqa_ar["learned_qacc_rmse_per_joint"], nqa_ar["nominal_qacc_rmse_per_joint"], "qacc"),
    ]):
        n_j = lv.shape[0]
        idx_j = np.arange(n_j)
        ax.bar(idx_j - w/2, lv, w, label="Learned", color="green", alpha=0.75)
        ax.bar(idx_j + w/2, nv_arr, w, label="Nominal", color="red",   alpha=0.75)
        ax.set_title(title); ax.set_xlabel("Joint"); ax.set_ylabel("RMSE")
        ax.set_xticks(idx_j); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    stem = os.path.join(plots_dir, "rmse_per_joint")
    save_Figure(fig, stem)
    save_Plot_Data({
        "qpos_joint_idx": np.arange(lqp_ar["learned_qpos_rmse_per_joint"].shape[0]),
        "qvel_joint_idx": np.arange(lqv_ar["learned_qvel_rmse_per_joint"].shape[0]),
        **{k: v for d in [lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar]
           for k, v in d.items() if "per_joint" in k and "traj" not in k},
    }, stem)
    plt.close(fig)

    # ── R² per joint ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("R² per Joint  (higher is better, 1.0 = perfect)", fontsize=12)
    for ax, (lk, nk, title) in zip(axes, [
        ("learned_qpos_r2_per_joint", "nominal_qpos_r2_per_joint", "qpos"),
        ("learned_qvel_r2_per_joint", "nominal_qvel_r2_per_joint", "qvel"),
        ("learned_qacc_r2_per_joint", "nominal_qacc_r2_per_joint", "qacc"),
    ]):
        l_r2 = lqp_ar[lk] if "qpos" in lk else (lqv_ar[lk] if "qvel" in lk else lqa_ar[lk])
        n_r2 = nqp_ar[nk] if "qpos" in nk else (nqv_ar[nk] if "qvel" in nk else nqa_ar[nk])
        n_j = l_r2.shape[0]
        idx_j = np.arange(n_j)
        ax.bar(idx_j - w/2, l_r2, w, label="Learned", color="green", alpha=0.75)
        ax.bar(idx_j + w/2, n_r2, w, label="Nominal", color="red",   alpha=0.75)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="gray",  linewidth=0.5, linestyle=":",  alpha=0.5)
        ax.set_title(title); ax.set_xlabel("Joint"); ax.set_ylabel("R²")
        ax.set_xticks(idx_j); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    stem = os.path.join(plots_dir, "r2_per_joint")
    save_Figure(fig, stem)
    save_Plot_Data({
        "qpos_joint_idx": np.arange(lqp_ar["learned_qpos_r2_per_joint"].shape[0]),
        "qvel_joint_idx": np.arange(lqv_ar["learned_qvel_r2_per_joint"].shape[0]),
        **{k: v for d in [lqp_ar, nqp_ar, lqv_ar, nqv_ar, lqa_ar, nqa_ar]
           for k, v in d.items() if "r2_per_joint" in k},
    }, stem)
    plt.close(fig)

    # ── RMSE distribution histograms ──────────────────────────────────────
    # Show how RMSE is distributed across trajectories (aggregated over joints)
    if lqp is not None:
        lqp_rmse_per_traj = np.sqrt(np.mean((lqp - gqp)**2, axis=(1, 2)))  # (N,)
        nqp_rmse_per_traj = np.sqrt(np.mean((nqp - gqp)**2, axis=(1, 2)))
        lqv_rmse_per_traj = np.sqrt(np.mean((lqv - gqv)**2, axis=(1, 2)))
        nqv_rmse_per_traj = np.sqrt(np.mean((nqv - gqv)**2, axis=(1, 2)))
        lqa_rmse_per_traj = np.sqrt(np.mean((lqa - gqa)**2, axis=(1, 2)))
        nqa_rmse_per_traj = np.sqrt(np.mean((nqa - gqa)**2, axis=(1, 2)))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("RMSE Distribution Across Trajectories", fontsize=12)
        nbins = min(40, max(10, lqp.shape[0] // 5))
        for ax, (l_rmse, n_rmse, title) in zip(axes, [
            (lqp_rmse_per_traj, nqp_rmse_per_traj, "qpos"),
            (lqv_rmse_per_traj, nqv_rmse_per_traj, "qvel"),
            (lqa_rmse_per_traj, nqa_rmse_per_traj, "qacc"),
        ]):
            lo = min(l_rmse.min(), n_rmse.min())
            hi = max(l_rmse.max(), n_rmse.max())
            bins = np.linspace(lo, hi, nbins + 1)
            ax.hist(l_rmse, bins=bins, color="green", alpha=0.55, label="Learned", edgecolor="darkgreen")
            ax.hist(n_rmse, bins=bins, color="red",   alpha=0.45, label="Nominal", edgecolor="darkred")
            ax.axvline(np.median(l_rmse), color="darkgreen", linestyle="--", linewidth=1.5, label=f"L med={np.median(l_rmse):.4f}")
            ax.axvline(np.median(n_rmse), color="darkred",   linestyle="--", linewidth=1.5, label=f"N med={np.median(n_rmse):.4f}")
            ax.set_title(title); ax.set_xlabel("RMSE"); ax.set_ylabel("Count")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        stem = os.path.join(plots_dir, "rmse_distribution")
        save_Figure(fig, stem)
        save_Plot_Data({
            "learned_qpos_rmse_per_traj": lqp_rmse_per_traj,
            "nominal_qpos_rmse_per_traj": nqp_rmse_per_traj,
            "learned_qvel_rmse_per_traj": lqv_rmse_per_traj,
            "nominal_qvel_rmse_per_traj": nqv_rmse_per_traj,
            "learned_qacc_rmse_per_traj": lqa_rmse_per_traj,
            "nominal_qacc_rmse_per_traj": nqa_rmse_per_traj,
        }, stem)
        plt.close(fig)

    # ── Predicted vs GT scatter (45-degree plot) ──────────────────────────
    # A perfect model has all points on y = x.  Subsample in time for clarity.
    if lqp is not None:
        N_traj, T_full, _ = lqp.shape
        # Subsample: take every k-th timestep so we don't plot millions of points
        t_stride = max(1, T_full // 20)
        t_inds   = np.arange(0, T_full, t_stride)

        for qty_label, l_arr, n_arr, g_arr in [
            ("qpos", lqp, nqp, gqp),
            ("qvel", lqv, nqv, gqv),
            ("qacc", lqa, nqa, gqa),
        ]:
            t_ix = t_inds if "acc" not in qty_label else t_inds[t_inds < l_arr.shape[1]]
            n_dim = l_arr.shape[-1]
            fig, axes = plt.subplots(2, n_dim, figsize=(max(3*n_dim, 10), 7), squeeze=False)
            fig.suptitle(f"Predicted vs Ground Truth  —  {qty_label}\n"
                         "(y = x dashed black = perfect fit)", fontsize=11)
            for j in range(n_dim):
                gt_vals = g_arr[:, t_ix, j].ravel()
                for row, (pred_arr, pred_label, colour) in enumerate([
                    (l_arr, "Learned", "green"),
                    (n_arr, "Nominal", "red"),
                ]):
                    ax = axes[row, j]
                    pred_vals = pred_arr[:, t_ix, j].ravel()
                    ax.scatter(gt_vals, pred_vals, s=1, alpha=0.15, color=colour, rasterized=True)
                    lims = [min(gt_vals.min(), pred_vals.min()),
                            max(gt_vals.max(), pred_vals.max())]
                    ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.6)
                    ax.set_xlim(lims); ax.set_ylim(lims)
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_title(f"J{j} {pred_label}", fontsize=9)
                    ax.set_xlabel("GT", fontsize=7); ax.set_ylabel("Predicted", fontsize=7)
                    ax.tick_params(labelsize=6); ax.grid(True, alpha=0.2)
            plt.tight_layout()
            stem = os.path.join(plots_dir, f"pred_vs_gt_{qty_label}")
            save_Figure(fig, stem)
            plt.close(fig)

    # ── Cumulative error (ECDF) ───────────────────────────────────────────
    if lqp is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Cumulative Distribution of Per-Trajectory RMSE", fontsize=12)
        for ax, (l_rmse, n_rmse, title) in zip(axes, [
            (lqp_rmse_per_traj, nqp_rmse_per_traj, "qpos"),
            (lqv_rmse_per_traj, nqv_rmse_per_traj, "qvel"),
            (lqa_rmse_per_traj, nqa_rmse_per_traj, "qacc"),
        ]):
            l_sorted = np.sort(l_rmse)
            n_sorted = np.sort(n_rmse)
            cdf = np.linspace(0, 1, len(l_sorted))
            ax.plot(l_sorted, cdf, color="green", linewidth=1.5, label="Learned")
            ax.plot(n_sorted, cdf, color="red",   linewidth=1.5, linestyle="--", label="Nominal")
            ax.set_title(title); ax.set_xlabel("RMSE"); ax.set_ylabel("Cumulative Fraction")
            ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        stem = os.path.join(plots_dir, "rmse_cdf")
        save_Figure(fig, stem)
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
        nv = gqv.shape[1]

        if not should_save:
            continue

        traj_dir = make_dir(job_dir, f"trajectory_{traj_idx}")

        # ── Plot 1a: per-joint qpos comparison ───────────────────────────────
        fig, axes = plt.subplots(1, nq, figsize=(max(3*nq, 10), 4), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Per-Joint qpos\n"
                     "(black=GT  green=Learned  red=Nominal)", fontsize=11)
        for j in range(nq):
            ax = axes[0, j]
            ax.plot(ts_np, gqp[:, j], "k",  linewidth=1.5)
            ax.plot(ts_np, lqp[:, j], "g--",linewidth=1.0)
            ax.plot(ts_np, nqp[:, j], "r:", linewidth=1.0)
            ax.set_title(f"J{j} qpos", fontsize=9)
            ax.set_xlabel("t", fontsize=7); ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("qpos")
                ax.legend(["GT","Learned","Nominal"], fontsize=6)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_Figure(fig, os.path.join(traj_dir, "joint_comparison_qpos"))
        plt.close(fig)

        # ── Plot 1b: per-joint qvel + qacc comparison ────────────────────────
        fig, axes = plt.subplots(2, nv, figsize=(max(3*nv, 10), 8), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Per-Joint qvel / qacc\n"
                     "(black=GT  green=Learned  red=Nominal)", fontsize=11)
        for j in range(nv):
            for row, (gy, ly, ny, t_arr, yl) in enumerate([
                (gqv[:, j], lqv[:, j], nqv[:, j], ts_np,  "qvel"),
                (gqa[:, j], lqa[:, j], nqa[:, j], ts_acc, "qacc"),
            ]):
                ax = axes[row, j]
                ax.plot(t_arr, gy, "k",  linewidth=1.5)
                ax.plot(t_arr, ly, "g--",linewidth=1.0)
                ax.plot(t_arr, ny, "r:", linewidth=1.0)
                ax.set_title(f"V{j} {yl}", fontsize=9)
                ax.set_xlabel("t", fontsize=7); ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel(yl)
                    if row == 0:
                        ax.legend(["GT","Learned","Nominal"], fontsize=6)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_Figure(fig, os.path.join(traj_dir, "joint_comparison_qvel_qacc"))
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
        fig, axes = plt.subplots(1, nv, figsize=(max(3*nv, 10), 4), squeeze=False)
        fig.suptitle(f"Trajectory {traj_idx}  —  Residual Acceleration  "
                     "(purple=L−N  dashed=GT−N)", fontsize=11)
        for j in range(nv):
            ax = axes[0, j]
            ax.plot(ts_acc, res_acc[:, j],          color="purple", linewidth=1.5)
            ax.plot(ts_acc, gqa[:, j]-nqa[:, j],    "k--", linewidth=1.0)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.set_title(f"V{j}", fontsize=9); ax.set_xlabel("t", fontsize=7)
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
            n_dim = pred.shape[1]
            im = ax.imshow(np.abs(pred-gt_y).T, aspect="auto", origin="lower",
                           extent=[ts_np[0], ts_np[-1], -0.5, n_dim-0.5], cmap="hot_r")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("t"); ax.set_ylabel("Joint")
            ax.set_yticks(range(n_dim)); plt.colorbar(im, ax=ax, shrink=0.85)
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
    nv     = gt_qvel_all[0].shape[1]
    mapping = qpos_qvel_joint_map(nq, nv)

    # Build list of (qvel_idx, qpos_idx) pairs for phase portraits.
    # Skip angular-velocity components that have no scalar qpos partner.
    if which_joints is not None:
        # User-specified: interpret as qvel indices
        pairs = [(v, mapping[v]) for v in which_joints if v < nv and mapping[v] is not None]
    else:
        pairs = [(v, mapping[v]) for v in range(nv) if mapping[v] is not None]

    if len(pairs) == 0:
        print("    No valid qpos-qvel pairs for phase portrait — skipping.")
        return

    fig, axes = plt.subplots(3, len(pairs), figsize=(4 * len(pairs), 12), squeeze=False)
    fig.suptitle("Phase Portraits  (qvel vs qpos)\n"
                 "Colour encodes time  (blue=start → yellow=end)\n"
                 "Row 1: GT   Row 2: Learned   Row 3: Nominal", fontsize=11)

    cmap = cm.plasma

    for col, (v_idx, q_idx) in enumerate(pairs):
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
                        qpos_traj[t:t+2, q_idx],
                        qvel_traj[t:t+2, v_idx],
                        color=colours[t],
                        linewidth=0.7,
                        alpha=0.7,
                    )
            ax.set_title(f"q{q_idx}/v{v_idx}  {row_label}", fontsize=9)
            ax.set_xlabel(f"qpos[{q_idx}]", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nqvel[{v_idx}]", fontsize=8)
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
        "which_pairs": np.array(pairs),
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
    nv = lqv_ptj.shape[1]

    print(f"\n  Error Distribution  ({N} trajectories, nq={nq}, nv={nv})")
    print(f"  Median qpos improvement per joint: "
          + "  ".join(f"J{j}: {np.median(impr_qpos[:, j]):+.1f}%" for j in range(nq)))
    print(f"  Median qvel improvement per joint: "
          + "  ".join(f"V{j}: {np.median(impr_qvel[:, j]):+.1f}%" for j in range(nv)))

    if not should_save:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Per-Trajectory Error Distribution  (N={N} trajectories)\n"
                 "Green=Learned  Red=Nominal", fontsize=12)

    for ax, (l_data, n_data, ylabel, title) in zip(axes.flat, [
        (lqp_ptj, nqp_ptj, "RMSE",          "qpos RMSE per Trajectory"),
        (lqv_ptj, nqv_ptj, "RMSE",          "qvel RMSE per Trajectory"),
        (impr_qpos, None,  "Improvement %", "qpos Improvement % (pos=better)"),
        (impr_qvel, None,  "Improvement %", "qvel Improvement % (pos=better)"),
    ]):
        n_j = l_data.shape[1]
        xs = np.arange(n_j)
        joint_labels = [f"J{j}" for j in range(n_j)]
        if n_data is not None:
            # Paired violins for learned vs nominal
            vp_l = ax.violinplot([l_data[:, j] for j in range(n_j)],
                                 positions=xs - 0.2, widths=0.35,
                                 showmedians=True, showextrema=False)
            vp_n = ax.violinplot([n_data[:, j] for j in range(n_j)],
                                 positions=xs + 0.2, widths=0.35,
                                 showmedians=True, showextrema=False)
            for pc in vp_l["bodies"]: pc.set_facecolor("green"); pc.set_alpha(0.55)
            for pc in vp_n["bodies"]: pc.set_facecolor("red");   pc.set_alpha(0.55)
            vp_l["cmedians"].set_color("darkgreen")
            vp_n["cmedians"].set_color("darkred")
            # Raw strip
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, (N, n_j))
            for j in range(n_j):
                ax.scatter(xs[j] - 0.2 + jitter[:, j], l_data[:, j],
                           s=2, color="green", alpha=0.25)
                ax.scatter(xs[j] + 0.2 + jitter[:, j], n_data[:, j],
                           s=2, color="red",   alpha=0.25)
        else:
            # Single improvement violin
            vp = ax.violinplot([l_data[:, j] for j in range(n_j)],
                               positions=xs, widths=0.6,
                               showmedians=True, showextrema=True)
            for pc in vp["bodies"]:
                pc.set_facecolor("steelblue"); pc.set_alpha(0.6)
            ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
            jitter = np.random.default_rng(1).uniform(-0.08, 0.08, (N, n_j))
            for j in range(n_j):
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
    nv = lqv.shape[-1]
    freqs = np.fft.rfftfreq(T, d=dt)   # Hz

    def mean_psd(signal):
        """Mean PSD across trajectories; shape (F, D)."""
        F   = np.fft.rfft(signal, axis=1)
        psd = (np.abs(F)**2) / signal.shape[1]
        return np.mean(psd, axis=0)

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
    res_acc = lqa - nqa   # (N, T-1, nv)
    res_freqs = np.fft.rfftfreq(T - 1, d=dt)
    res_psd   = mean_psd(res_acc)   # (F', nv)

    print(f"\n  Frequency Analysis  ({N} trajectories, nq={nq}, nv={nv}, "
          f"fs={1/dt:.1f} Hz, Nyquist={0.5/dt:.1f} Hz)")

    if not should_save:
        return

    eps = 1e-30   # floor to avoid log(0)

    # ── qpos error PSD  (nq columns) ─────────────────────────────────────
    fig_p, axes_p = plt.subplots(1, nq, figsize=(4 * nq, 4), squeeze=False)
    fig_p.suptitle("PSD of qpos Error  (green=Learned, red=Nominal)", fontsize=12)
    for j in range(nq):
        ax = axes_p[0, j]
        ax.loglog(freqs[1:], lqp_psd[1:, j] + eps, "g",  label="Learned", linewidth=1.5)
        ax.loglog(freqs[1:], nqp_psd[1:, j] + eps, "r--",label="Nominal", linewidth=1.5)
        ax.set_title(f"J{j}  qpos error", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        if j == 0: ax.set_ylabel("PSD", fontsize=7); ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    stem = os.path.join(job_dir, "frequency_analysis_qpos")
    save_Figure(fig_p, stem)
    plt.close(fig_p)

    # ── qvel error + residual acc PSD  (nv columns) ──────────────────────
    fig, axes = plt.subplots(2, nv, figsize=(4 * nv, 8), squeeze=False)
    fig.suptitle("PSD of qvel Error & Residual Acceleration\n"
                 "(green=Learned, red=Nominal)  log–log scale", fontsize=12)

    for j in range(nv):
        # qvel error PSD
        ax = axes[0, j]
        ax.loglog(freqs[1:], lqv_psd[1:, j] + eps, "g",  linewidth=1.5)
        ax.loglog(freqs[1:], nqv_psd[1:, j] + eps, "r--",linewidth=1.5)
        ax.set_title(f"V{j}  qvel error", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        if j == 0: ax.set_ylabel("PSD", fontsize=7); ax.legend(["Learned","Nominal"], fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

        # Residual acceleration PSD
        ax = axes[1, j]
        ax.loglog(res_freqs[1:], res_psd[1:, j] + eps, color="purple", linewidth=1.5)
        ax.set_title(f"V{j}  residual acc", fontsize=9)
        ax.set_xlabel("Freq (Hz)", fontsize=7)
        if j == 0: ax.set_ylabel("PSD (residual acc)", fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

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
# NEW JOB: ACCELERATION LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════
#
# Fix all but two joint angles (set velocities and ctrl to zero), scan the
# remaining two over a grid, and evaluate the instantaneous qacc from the
# learned model, the nominal (MuJoCo-only) model, and a ground-truth
# fluid-enabled MuJoCo model.  Produces heatmaps of qacc magnitude,
# correction magnitude, error magnitude, and error angle, plus a panel of
# rendered robot images at selected configurations.
#
# Expected job_options fields:
#   vary_joints       : [int, int]   — the two joint indices to scan
#   grid_resolution   : int          — number of points per axis (default 50)
#   joint_range_frac  : float        — fraction of joint range to scan (default 1.0)
#   should_save       : bool
#
# Example JSON:
#   "acc_landscape": {
#       "type": "acc_landscape",
#       "vary_joints": [0, 1],
#       "grid_resolution": 50,
#       "joint_range_frac": 1.0,
#       "should_save": true
#   }
# ══════════════════════════════════════════════════════════════════════════════

def _eval_qacc_at_config(model, mjx_data, qpos, qvel, ctrl, with_learned):
    """Evaluate instantaneous qacc from the KNODE dynamics at a single config."""
    state = {"qpos": qpos, "qvel": qvel}
    result = model.knode_dynamics(0.0, state, (mjx_data, ctrl, with_learned))
    return result["qvel"]   # qvel-dot = qacc


def run_Acc_Landscape_Job(model, mjx_data, mjx_model, job_options, job_name,
                          save_path, test_options):
    import mujoco

    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    vary_joints = job_options["vary_joints"]
    assert len(vary_joints) == 2, "vary_joints must specify exactly 2 joint indices."
    ji, jj = vary_joints
    res   = job_options.get("grid_resolution", 50)
    frac  = job_options.get("joint_range_frac", 1.0)

    nq = mjx_model.nq
    nv = mjx_model.nv
    nu = mjx_model.nu
    jnt_range = np.array(mjx_model.jnt_range)   # (nj, 2)

    # Compute scan ranges (centred on midpoint, scaled by frac)
    def _joint_linspace(j_idx):
        lo, hi = jnt_range[j_idx]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * frac
        return np.linspace(mid - half, mid + half, res)

    qi_vals = _joint_linspace(ji)
    qj_vals = _joint_linspace(jj)
    QI, QJ  = np.meshgrid(qi_vals, qj_vals, indexing="ij")   # (res, res)

    # Build full qpos/qvel/ctrl grids  (res*res, nq/nv/nu)
    n_pts  = res * res
    qpos_grid = np.zeros((n_pts, nq))
    qvel_grid = np.ones((n_pts, nv))*3.14
    ctrl_grid = np.zeros((n_pts, nu))

    flat_qi = QI.ravel()
    flat_qj = QJ.ravel()
    qpos_grid[:, ji] = flat_qi
    qpos_grid[:, jj] = flat_qj

    print(f"  Scanning J{ji} × J{jj}  ({res}×{res} = {n_pts} points)")

    # ── Evaluate learned & nominal qacc via KNODE dynamics ────────────────
    @eqx.filter_jit
    def _eval_grid(mdl, mjx_d, qpos_batch, qvel_batch, ctrl_batch, wl):
        return jax.vmap(
            lambda qp, qv, ct: _eval_qacc_at_config(mdl, mjx_d, qp, qv, ct, wl),
            in_axes=(0, 0, 0),
        )(qpos_batch, qvel_batch, ctrl_batch)

    jqp = jnp.array(qpos_grid)
    jqv = jnp.array(qvel_grid)
    jct = jnp.array(ctrl_grid)

    learned_qacc = np.array(_eval_grid(model, mjx_data, jqp, jqv, jct, jnp.array(1.0)))   # (n_pts, nv)
    nominal_qacc = np.array(_eval_grid(model, mjx_data, jqp, jqv, jct, jnp.array(0.0)))   # (n_pts, nv)
    print("    Learned & nominal qacc computed.")

    # ── Evaluate GT qacc from fluid-enabled MuJoCo ────────────────────────
    gt_qacc = None
    training_options = get_Training_Options(test_options)
    try:
        gt_opts = {**training_options, "is_fluid": True}
        gt_xml  = mj_utils.create_XML_From_Options(gt_opts)
        gt_mj_model = mujoco.MjModel.from_xml_string(gt_xml)
        gt_mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
        gt_mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
        gt_mj_data  = mujoco.MjData(gt_mj_model)

        gt_qacc = np.zeros((n_pts, nv))
        for k in range(n_pts):
            gt_mj_data.qpos[:] = qpos_grid[k]
            gt_mj_data.qvel[:] = qvel_grid[k]
            gt_mj_data.ctrl[:] = ctrl_grid[k]
            mujoco.mj_forward(gt_mj_model, gt_mj_data)
            gt_qacc[k] = gt_mj_data.qacc.copy()
        print("    GT (fluid-enabled) qacc computed.")
    except Exception as e:
        print(f"    Warning: could not compute GT qacc: {e}")
        print("    Plots will omit ground-truth panels.")

    # ── Derived quantities ────────────────────────────────────────────────
    correction   = learned_qacc - nominal_qacc                    # (n_pts, nv)
    corr_mag     = np.linalg.norm(correction, axis=1)             # (n_pts,)
    learned_mag  = np.linalg.norm(learned_qacc, axis=1)
    nominal_mag  = np.linalg.norm(nominal_qacc, axis=1)

    has_gt = gt_qacc is not None
    if has_gt:
        gt_mag     = np.linalg.norm(gt_qacc, axis=1)
        err        = learned_qacc - gt_qacc
        err_mag    = np.linalg.norm(err, axis=1)
        # Error angle between learned and GT qacc vectors
        dot_prod   = np.sum(learned_qacc * gt_qacc, axis=1)
        denom      = learned_mag * gt_mag + 1e-12
        cos_angle  = np.clip(dot_prod / denom, -1.0, 1.0)
        err_angle  = np.degrees(np.arccos(cos_angle))

    if not should_save:
        return

    # ── Reshape for 2D plotting ───────────────────────────────────────────
    def _grid(arr):
        return arr.reshape(res, res)

    extent = [qj_vals[0], qj_vals[-1], qi_vals[0], qi_vals[-1]]

    # ── Panel plot ────────────────────────────────────────────────────────
    n_cols = 3 if has_gt else 2
    n_rows = 2 if has_gt else 2
    if has_gt:
        panels = [
            ("GT ||qacc||",       _grid(gt_mag),     "viridis"),
            ("Learned ||qacc||",  _grid(learned_mag),"viridis"),
            ("Nominal ||qacc||",  _grid(nominal_mag),"viridis"),
            ("Correction ||Δ||",  _grid(corr_mag),   "inferno"),
            ("Error ||L−GT||",    _grid(err_mag),    "hot"),
            ("Error Angle (°)",   _grid(err_angle),  "coolwarm"),
        ]
        n_rows, n_cols = 2, 3
    else:
        panels = [
            ("Learned ||qacc||",  _grid(learned_mag),"viridis"),
            ("Nominal ||qacc||",  _grid(nominal_mag),"viridis"),
            ("Correction ||Δ||",  _grid(corr_mag),   "inferno"),
            ("Corr. / Nominal",   _grid(corr_mag / (nominal_mag.reshape(res,res) + 1e-12)), "plasma"),
        ]
        n_rows, n_cols = 2, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
    fig.suptitle(f"Acceleration Landscape  —  Varying J{ji} × J{jj}\n"
                 f"(all other joints = 0, qvel = 0, ctrl = 0)", fontsize=12)
    for ax, (title, data_2d, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data_2d, origin="lower", aspect="auto",
                       extent=extent, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"J{jj} (rad)", fontsize=9)
        ax.set_ylabel(f"J{ji} (rad)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    stem = os.path.join(job_dir, "acc_landscape")
    save_Figure(fig, stem)
    save_data = {
        "qi_vals": qi_vals, "qj_vals": qj_vals,
        "learned_qacc": learned_qacc.reshape(res, res, nv),
        "nominal_qacc": nominal_qacc.reshape(res, res, nv),
        "correction": correction.reshape(res, res, nv),
    }
    if has_gt:
        save_data["gt_qacc"]   = gt_qacc.reshape(res, res, nv)
        save_data["err_mag"]   = _grid(err_mag)
        save_data["err_angle"] = _grid(err_angle)
    save_Plot_Data(save_data, stem)
    plt.close(fig)

    # ── Per-joint correction heatmaps ─────────────────────────────────────
    fig, axes = plt.subplots(2, nv, figsize=(max(3 * nv, 10), 8), squeeze=False)
    fig.suptitle(f"Per-Joint Correction & Error  —  J{ji} × J{jj}", fontsize=12)
    for j in range(nv):
        ax = axes[0, j]
        corr_j = correction[:, j].reshape(res, res)
        im = ax.imshow(corr_j, origin="lower", aspect="auto",
                       extent=extent, cmap="RdBu_r")
        ax.set_title(f"J{j} correction", fontsize=9)
        ax.set_xlabel(f"J{jj}", fontsize=7); ax.set_ylabel(f"J{ji}", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[1, j]
        if has_gt:
            err_j = (learned_qacc[:, j] - gt_qacc[:, j]).reshape(res, res)
            im = ax.imshow(err_j, origin="lower", aspect="auto",
                           extent=extent, cmap="RdBu_r")
            ax.set_title(f"J{j} error (L−GT)", fontsize=9)
        else:
            # Show learned qacc per joint instead
            l_j = learned_qacc[:, j].reshape(res, res)
            im = ax.imshow(l_j, origin="lower", aspect="auto",
                           extent=extent, cmap="viridis")
            ax.set_title(f"J{j} learned qacc", fontsize=9)
        ax.set_xlabel(f"J{jj}", fontsize=7); ax.set_ylabel(f"J{ji}", fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    stem = os.path.join(job_dir, "per_joint_landscape")
    save_Figure(fig, stem)
    plt.close(fig)

    # ── Robot configuration renders ───────────────────────────────────────
    try:
        model_xml = mj_utils.create_XML_From_Options(training_options)
        mj_model_render = mujoco.MjModel.from_xml_string(model_xml)
        mj_data_render  = mujoco.MjData(mj_model_render)
        renderer = mujoco.Renderer(mj_model_render, height=480, width=640)

        # Pick 5 representative configs: 4 corners + centre
        corner_indices = [
            (0, 0), (0, res-1), (res-1, 0), (res-1, res-1),
            (res//2, res//2),
        ]
        labels = [
            f"J{ji}={qi_vals[ci[0]]:.2f}, J{jj}={qj_vals[ci[1]]:.2f}"
            for ci in corner_indices
        ]

        fig_r, axes_r = plt.subplots(1, 5, figsize=(25, 5))
        fig_r.suptitle(f"Robot Configurations  —  Varying J{ji} × J{jj}", fontsize=12)
        for ax, (ci_i, ci_j), label in zip(axes_r, corner_indices, labels):
            mj_data_render.qpos[:] = 0.0
            mj_data_render.qpos[ji] = qi_vals[ci_i]
            mj_data_render.qpos[jj] = qj_vals[ci_j]
            mujoco.mj_forward(mj_model_render, mj_data_render)
            renderer.update_scene(mj_data_render)
            pixels = renderer.render().copy()
            ax.imshow(pixels)
            ax.set_title(label, fontsize=8)
            ax.axis("off")

        renderer.close()
        plt.tight_layout()
        stem = os.path.join(job_dir, "robot_configurations")
        save_Figure(fig_r, stem)
        plt.close(fig_r)
        print("    Robot configuration renders saved.")
    except Exception as e:
        print(f"    Warning: robot rendering failed: {e}")

    print(f"  Acceleration landscape saved to: {job_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
# NEW JOB: LOSS SURFACE
# ══════════════════════════════════════════════════════════════════════════════
#
# Pick two random trainable parameters from the model, evaluate the training
# loss of a fixed batch of trajectories on a 2D grid of perturbations around
# the current parameter values, and plot the resulting loss surface.  This
# gives a visual sense of the local curvature, flatness, and any saddle-point
# structure near the optimum.
#
# Expected job_options fields:
#   dataset_name        : str
#   subset_size         : int         — trajectories to use (= batch_size)
#   batch_size          : int
#   grid_resolution     : int         — points per axis (default 25)
#   perturbation_scale  : float       — half-width of scan in param-value units
#                                       (default 0.5, meaning ±0.5 around current)
#   seed                : int|null    — RNG seed for picking parameters
#   should_save         : bool
#
# Example JSON:
#   "loss_surface": {
#       "type": "loss_surface",
#       "dataset_name": "dataset_3_fluid_no_coll_valid",
#       "subset_size": 50,
#       "batch_size": 50,
#       "grid_resolution": 25,
#       "perturbation_scale": 0.5,
#       "seed": 42,
#       "should_save": true
#   }
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_trainable_params(model):
    """
    Return a 1D array of all trainable (inexact) parameter values, plus
    metadata needed to reconstruct a modified model.

    Returns
    -------
    flat_params : np.ndarray (P,)
    leaf_info   : list of (leaf_index_in_full_tree, shape, start, end)
    leaves      : list  — all leaves of the pytree
    treedef     : pytree structure
    """
    leaves, treedef = jax.tree_util.tree_flatten(model)
    segments = []
    offset   = 0
    for i, leaf in enumerate(leaves):
        if eqx.is_inexact_array(leaf):
            arr = np.asarray(leaf).ravel()
            n   = arr.shape[0]
            segments.append((i, leaf.shape, offset, offset + n, arr))
            offset += n
    flat = np.concatenate([s[4] for s in segments])
    info = [(s[0], s[1], s[2], s[3]) for s in segments]
    return flat, info, leaves, treedef


def _rebuild_model_with_two_params(leaves, treedef, leaf_info, flat_params,
                                    idx_a, val_a, idx_b, val_b):
    """
    Return a new model with flat_params[idx_a] = val_a, flat_params[idx_b] = val_b.
    """
    fp = flat_params.copy()
    fp[idx_a] = val_a
    fp[idx_b] = val_b

    new_leaves = list(leaves)   # shallow copy
    for (li, shape, start, end) in leaf_info:
        new_leaves[li] = jnp.array(fp[start:end].reshape(shape))
    return treedef.unflatten(new_leaves)


def run_Loss_Surface_Job(model, mjx_data, job_options, job_name, save_path):
    should_save = job_options.get("should_save", True)
    if should_save:
        job_dir = make_dir(save_path, job_name)

    root, keys, ts, step_size = open_Dataset(job_options)
    subset_size = job_options.get("subset_size", 50)
    batch_size  = job_options.get("batch_size", subset_size)
    res         = job_options.get("grid_resolution", 25)
    scale       = job_options.get("perturbation_scale", 0.5)
    seed        = job_options.get("seed", None)

    # ── Load one fixed batch of trajectories ──────────────────────────────
    if subset_size % batch_size != 0:
        subset_size = (subset_size // batch_size) * batch_size
    if subset_size == 0:
        subset_size = batch_size

    gqp_list, gqv_list, ctrl_list = [], [], []
    for idx in range(min(subset_size, len(keys))):
        d = root[keys[idx]]
        gqp_list.append(np.array(d["qpos"][:]))
        gqv_list.append(np.array(d["qvel"][:]))
        ctrl_list.append(np.array(d["ctrl"][0, :]))

    gt_qpos = jnp.array(np.stack(gqp_list, 0))
    gt_qvel = jnp.array(np.stack(gqv_list, 0))
    ctrls   = jnp.array(np.stack(ctrl_list, 0))
    print(f"  Loaded {gt_qpos.shape[0]} trajectories for loss evaluation.")

    # ── Flatten model parameters ──────────────────────────────────────────
    flat_params, leaf_info, leaves, treedef = _flatten_trainable_params(model)
    total_params = flat_params.shape[0]
    print(f"  Total trainable parameters: {total_params}")

    rng = np.random.default_rng(seed)
    idx_a, idx_b = rng.choice(total_params, size=2, replace=False)
    val_a_orig = flat_params[idx_a]
    val_b_orig = flat_params[idx_b]

    # Figure out which leaf/position each belongs to (for labelling)
    def _param_label(flat_idx):
        for (li, shape, start, end) in leaf_info:
            if start <= flat_idx < end:
                local = flat_idx - start
                return f"leaf[{li}] shape={shape} elem[{local}]"
        return f"flat[{flat_idx}]"

    label_a = _param_label(idx_a)
    label_b = _param_label(idx_b)
    print(f"  Param A (idx {idx_a}): {label_a}  val={val_a_orig:.6f}")
    print(f"  Param B (idx {idx_b}): {label_b}  val={val_b_orig:.6f}")

    # ── Define loss function ──────────────────────────────────────────────
    @eqx.filter_jit
    def compute_loss(mdl, mjx_d, ts_, gt_qp, gt_qv, ctrls_):
        init_states = {"qpos": gt_qp[:, 0, :], "qvel": gt_qv[:, 0, :]}
        pred = jax.vmap(mdl, (None, None, 0, 0, None))(
            ts_, mjx_d, init_states, ctrls_, jnp.array(1.0)
        )
        return (jnp.mean((pred["qpos"] - gt_qp) ** 2)*589.0
                + jnp.mean((pred["qvel"] - gt_qv) ** 2)*3.6)

    # ── Scan the grid ─────────────────────────────────────────────────────
    a_vals = np.linspace(val_a_orig - scale, val_a_orig + scale, res)
    b_vals = np.linspace(val_b_orig - scale, val_b_orig + scale, res)

    loss_grid = np.zeros((res, res))
    total_evals = res * res
    print(f"  Evaluating {total_evals} grid points ({res}×{res}) ...")

    for i, va in enumerate(a_vals):
        for j, vb in enumerate(b_vals):
            modified = _rebuild_model_with_two_params(
                leaves, treedef, leaf_info, flat_params, idx_a, va, idx_b, vb
            )
            loss_grid[i, j] = float(compute_loss(modified, mjx_data, ts, gt_qpos, gt_qvel, ctrls))
        if (i + 1) % max(1, res // 5) == 0 or i == res - 1:
            print(f"    Row {i+1}/{res} complete.")

    # ── Current-value loss ────────────────────────────────────────────────
    current_loss = float(compute_loss(model, mjx_data, ts, gt_qpos, gt_qvel, ctrls))
    print(f"  Loss at current params: {current_loss:.6f}")
    print(f"  Loss range on grid: [{loss_grid.min():.6f}, {loss_grid.max():.6f}]")

    if not should_save:
        return

    # ── Surface plot ──────────────────────────────────────────────────────
    A, B = np.meshgrid(a_vals, b_vals, indexing="ij")

    fig = plt.figure(figsize=(16, 6))

    # 3D surface
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(A, B, loss_grid, cmap="viridis", alpha=0.85,
                      edgecolor="none", rstride=1, cstride=1)
    ax3d.scatter([val_a_orig], [val_b_orig], [current_loss],
                 color="red", s=80, zorder=10, label="Current")
    ax3d.set_xlabel(f"Param A\n({label_a})", fontsize=7)
    ax3d.set_ylabel(f"Param B\n({label_b})", fontsize=7)
    ax3d.set_zlabel("Loss", fontsize=8)
    ax3d.set_title("Loss Surface (3D)", fontsize=10)
    ax3d.legend(fontsize=7)

    # 2D heatmap with contours
    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(loss_grid.T, origin="lower", aspect="auto",
                     extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                     cmap="viridis")
    ax2d.contour(A, B, loss_grid, levels=15, colors="white", linewidths=0.5, alpha=0.6)
    ax2d.plot(val_a_orig, val_b_orig, "r*", markersize=14, label="Current")
    ax2d.set_xlabel(f"Param A  ({label_a})", fontsize=7)
    ax2d.set_ylabel(f"Param B  ({label_b})", fontsize=7)
    ax2d.set_title("Loss Surface (contour)", fontsize=10)
    ax2d.legend(fontsize=7)
    plt.colorbar(im, ax=ax2d, shrink=0.85, label="Loss")

    fig.suptitle("Loss Surface  —  2 Random Parameters Perturbed", fontsize=12)
    plt.tight_layout()
    stem = os.path.join(job_dir, "loss_surface")
    save_Figure(fig, stem)
    save_Plot_Data({
        "a_vals": a_vals, "b_vals": b_vals,
        "loss_grid": loss_grid,
        "val_a_orig": np.array(val_a_orig),
        "val_b_orig": np.array(val_b_orig),
        "current_loss": np.array(current_loss),
        "idx_a": np.array(idx_a), "idx_b": np.array(idx_b),
        "label_a": np.array(label_a), "label_b": np.array(label_b),
    }, stem)
    plt.close(fig)

    # ── Log-scale version ─────────────────────────────────────────────────
    fig2, (ax_log, ax_rel) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Loss Surface  —  Log Scale & Relative", fontsize=12)

    log_loss = np.log10(loss_grid + 1e-12)
    im1 = ax_log.imshow(log_loss.T, origin="lower", aspect="auto",
                        extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                        cmap="inferno")
    ax_log.contour(A, B, log_loss, levels=15, colors="white", linewidths=0.4, alpha=0.5)
    ax_log.plot(val_a_orig, val_b_orig, "c*", markersize=14, label="Current")
    ax_log.set_title("log₁₀(Loss)"); ax_log.legend(fontsize=7)
    ax_log.set_xlabel(f"Param A"); ax_log.set_ylabel(f"Param B")
    plt.colorbar(im1, ax=ax_log, shrink=0.85)

    # Relative to current
    rel_loss = loss_grid / (current_loss + 1e-12)
    im2 = ax_rel.imshow(rel_loss.T, origin="lower", aspect="auto",
                        extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]],
                        cmap="RdYlGn_r")
    ax_rel.contour(A, B, rel_loss, levels=[0.5, 0.9, 1.0, 1.1, 2.0, 5.0],
                   colors="black", linewidths=0.6, alpha=0.5)
    ax_rel.plot(val_a_orig, val_b_orig, "k*", markersize=14, label="Current (1.0)")
    ax_rel.set_title("Loss / Current Loss"); ax_rel.legend(fontsize=7)
    ax_rel.set_xlabel(f"Param A"); ax_rel.set_ylabel(f"Param B")
    plt.colorbar(im2, ax=ax_rel, shrink=0.85, label="Relative loss")

    plt.tight_layout()
    stem2 = os.path.join(job_dir, "loss_surface_log")
    save_Figure(fig2, stem2)
    plt.close(fig2)

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = {
        "param_a_flat_idx":  int(idx_a),
        "param_b_flat_idx":  int(idx_b),
        "param_a_label":     label_a,
        "param_b_label":     label_b,
        "param_a_value":     float(val_a_orig),
        "param_b_value":     float(val_b_orig),
        "current_loss":      float(current_loss),
        "grid_min_loss":     float(loss_grid.min()),
        "grid_max_loss":     float(loss_grid.max()),
        "grid_resolution":   res,
        "perturbation_scale": scale,
        "total_trainable_params": int(total_params),
        "num_trajectories":  int(gt_qpos.shape[0]),
    }
    with open(os.path.join(job_dir, "summary.json"), "w") as f:
        json.dump({k: json_safe(v) for k, v in summary.items()}, f, indent=2)

    print(f"  Loss surface saved to: {job_dir}\n")


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
    elif model_type == "uvms":
        model, mjx_data, mjx_model = load_UVMS_Model(test_options)
    elif model_type == "fsindy_uvms":
        model, mjx_data, mjx_model = load_FSINDy_UVMS_Model(test_options)
    elif model_type == "gp":
        model, mjx_data, mjx_model = load_GP_Model(test_options)
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

        elif job_type == "acc_landscape":
            run_Acc_Landscape_Job(model, mjx_data, mjx_model, job_options,
                                 job_name, save_path, test_options)

        elif job_type == "loss_surface":
            run_Loss_Surface_Job(model, mjx_data, job_options, job_name, save_path)

        else:
            print(f"  Unknown job type '{job_type}' — skipping.")

    print(f"\nAll jobs complete.")
    print(f"Model : {test_options['load_name']}")
    print(f"Output: Test_Outputs/{save_name}\n")


if __name__ == "__main__":
    main()
