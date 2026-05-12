import os
import argparse
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import json
import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import time
import Utils.Mujoco_Utils as mj_utils
import zarr


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT AND SPECIFICATION PARSING
# ══════════════════════════════════════════════════════════════════════════════

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--spec", type=str, default="visualization_spec")
    args = parser.parse_args()
    spec_path = "Library/Test_Specifications/" + args.spec + ".json"
    return spec_path


def parse_Spec(path):
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL OPTION HELPERS  (reused across loaders)
# ══════════════════════════════════════════════════════════════════════════════

def get_Training_Options(load_name):
    path = "Models/" + load_name + "/training_options.json"
    with open(path) as f:
        return json.load(f)


def get_Model_Options(load_name):
    path = "Models/" + load_name + "/model_options.json"
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS  (adapted from Test_Suite_C5.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_Old_Model(load_name):
    from Core.KNODE_Modules_Old import KNODE

    training_options = get_Training_Options(load_name)
    model_options    = get_Model_Options(load_name)

    dataset_path = "Datasets/" + training_options["dataset_name"] + ".zarr"
    root         = zarr.open(dataset_path)
    qvel_max     = jnp.array(root.attrs["dataset_options"]["qvel_max"])

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    key   = jr.key(int(time.time()))
    model = KNODE(key, mjx_model, model_options["width"], model_options["depth"], qvel_max)

    load_path = os.getcwd() + "/Models/" + load_name + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data


def load_GP_Model(load_name):
    from Core.GP_Modules import GP_Residual, KNODE_GP

    training_options = get_Training_Options(load_name)
    load_path        = os.getcwd() + "/Models/" + load_name + "/"

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    # Load saved numpy arrays to recover shapes
    X_train_norm     = jnp.array(np.load(load_path + "X_train_norm.npy"))
    alpha            = jnp.array(np.load(load_path + "alpha.npy"))
    log_lengthscales = jnp.array(np.load(load_path + "log_lengthscales.npy"))
    log_signal_var   = jnp.array(np.load(load_path + "log_signal_var.npy"))
    log_noise_var    = jnp.array(np.load(load_path + "log_noise_var.npy"))

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
    model = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data


def load_FSINDy_Model(load_name):
    from Core.Fourier_SINDy_Modules   import Fourier_SINDy_Residual, KNODE_Fourier_SINDy
    from Core.Train_Fourier_SINDy     import build_Fourier_SINDy_Library

    training_options = get_Training_Options(load_name)

    model_xml            = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data  = mj_utils.init_mjx_model(model_xml)

    library        = build_Fourier_SINDy_Library(training_options, mjx_model)
    sindy_residual = Fourier_SINDy_Residual(
        library, mjx_model.nq,
        jnp.zeros((library.n_library_features, mjx_model.nq))
    )
    model = KNODE_Fourier_SINDy(mjx_model, sindy_residual)

    load_path = os.getcwd() + "/Models/" + load_name + "/"
    model     = eqx.tree_deserialise_leaves(load_path + "model.eqx", model)
    return model, mjx_data


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY RUNNING
# ══════════════════════════════════════════════════════════════════════════════

def run_Model(model, mjx_data, ts, init_states, ctrl):
    """Run a model forward from initial conditions.
    Returns (learned_qpos, nominal_qpos) as numpy arrays."""
    learned_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
    nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))
    return np.array(learned_traj["qpos"]), np.array(nominal_traj["qpos"])


# ══════════════════════════════════════════════════════════════════════════════
# 5-MODEL MUJOCO VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_5_Model_Comparison(comparison_xml_name,
                                  old_qpos, gt_qpos, gp_qpos, fsindy_qpos, nominal_qpos):
    """
    Launch the MuJoCo viewer with 5 overlaid robots.

    Robot order in the comparison XML (mjx_panda_5.txt):
        0  (no prefix)  – KNODE_Old     (green-tinted)
        1  (prefix '1') – Ground Truth  (opaque / full colour)
        2  (prefix '2') – GP            (red-tinted)
        3  (prefix '3') – Fourier SINDy (orange-tinted)
        4  (prefix '4') – Nominal       (blue-tinted)
    """
    xml = mj_utils.get_XML_From_Name(comparison_xml_name)

    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    nq    = model.nq // 5

    print(f"\nComparison viewer launched  (nq per robot = {nq})")
    print("  Robot 0 : KNODE_Old")
    print("  Robot 1 : Ground Truth")
    print("  Robot 2 : GP")
    print("  Robot 3 : Fourier SINDy")
    print("  Robot 4 : Nominal")

    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for i in range(old_qpos.shape[0]):
                data.qpos[0*nq : 1*nq] = old_qpos[i]
                data.qpos[1*nq : 2*nq] = gt_qpos[i]
                data.qpos[2*nq : 3*nq] = nominal_qpos[i] 
                data.qpos[3*nq : 4*nq] = gp_qpos[i]
                data.qpos[4*nq : 5*nq] = fsindy_qpos[i]

                mujoco.mj_forward(model, data)
                time.sleep(0.002 * 50)
                viewer.sync()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    spec_path = return_Arguments()
    spec      = parse_Spec(spec_path)

    dataset_name     = spec["dataset_name"]
    trajectory_index = spec["trajectory_index"]
    comparison_xml   = spec["comparison_xml_name"]

    # ── Open dataset and extract the chosen trajectory ────────────────────
    root      = zarr.open(os.getcwd() + "/Datasets/" + dataset_name + ".zarr", mode="r")
    keys      = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    step_size = root.attrs["dataset_options"]["step_size"]
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts        = jnp.arange(0, step_size * num_steps, step=step_size)

    d       = root[keys[trajectory_index]]
    gt_qpos = jnp.array(d["qpos"][:])
    gt_qvel = jnp.array(d["qvel"][:])
    ctrl    = jnp.array(d["ctrl"][0, :])

    init_states = {"qpos": gt_qpos[0, :], "qvel": gt_qvel[0, :]}

    print(f"Dataset     : {dataset_name}")
    print(f"Trajectory  : {keys[trajectory_index]}  (index {trajectory_index})")
    print(f"Timesteps   : {len(ts)}   dt = {step_size}")

    # ── Load and run KNODE_Old ────────────────────────────────────────────
    print("\nLoading KNODE_Old ...")
    old_model, old_mjx_data = load_Old_Model(spec["old"]["load_name"])
    old_qpos, nominal_qpos  = run_Model(old_model, old_mjx_data, ts, init_states, ctrl)
    print("  Done.  (also obtained nominal trajectory)")

    # ── Load and run GP ───────────────────────────────────────────────────
    print("Loading GP ...")
    gp_model, gp_mjx_data = load_GP_Model(spec["gp"]["load_name"])
    gp_qpos, _            = run_Model(gp_model, gp_mjx_data, ts, init_states, ctrl)
    print("  Done.")

    # ── Load and run Fourier SINDy ────────────────────────────────────────
    print("Loading Fourier SINDy ...")
    fsindy_model, fsindy_mjx_data = load_FSINDy_Model(spec["fsindy"]["load_name"])
    fsindy_qpos, _                = run_Model(fsindy_model, fsindy_mjx_data, ts, init_states, ctrl)
    print("  Done.")

    # ── Launch 5-model comparison viewer ──────────────────────────────────
    gt_qpos_np = np.array(gt_qpos)

    visualize_5_Model_Comparison(
        comparison_xml,
        old_qpos,
        gt_qpos_np,
        gp_qpos,
        fsindy_qpos,
        nominal_qpos,
    )


if __name__ == "__main__":
    main()
