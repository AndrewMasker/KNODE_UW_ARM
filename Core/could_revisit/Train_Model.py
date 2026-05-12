import argparse
import collections
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
import zarr

from Core.KNODE_Modules import KNODE
import Utils.Data_Utils as data_utils
import Utils.Mujoco_Utils as mj_utils


# ── Argument and specification parsing ──────────────────────────────────────

def return_Arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_specs", type=str, default="model_fluid")
    parser.add_argument("-t", "--training_specs", type=str, default="training_fluid")
    args = parser.parse_args()
    model_path = f"Library/Model_Specifications/{args.model_specs}.json"
    training_path = f"Library/Training_Specifications/{args.training_specs}.json"
    return model_path, training_path


def parse_specs(path):
    with open(path) as f:
        return json.load(f)


# ── Normalization statistics ────────────────────────────────────────────────

def compute_normalization_stats(dataset_path, num_samples, qpos_input_mask):
    """Compute mean/std normalization statistics from a subset of the dataset.

    Returns a dict with keys:
        input_mean / input_std   – for the residual network input
        output_mean / output_std – for the residual network output (qacc scale)
        qpos_std / qvel_std      – for normalizing the training loss
    """
    root = zarr.open(dataset_path, mode="r")
    keys = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    n = min(num_samples, len(keys))
    step_size = float(root.attrs["dataset_options"]["step_size"])

    qpos_list, qvel_list, ctrl_list = [], [], []
    for i in range(n):
        d = root[keys[i]]
        qpos_list.append(d["qpos"][:])
        qvel_list.append(d["qvel"][:])
        ctrl_list.append(d["ctrl"][0, :])

    all_qpos = np.concatenate(qpos_list, axis=0)     # (n*T, nq)
    all_qvel = np.concatenate(qvel_list, axis=0)      # (n*T, nq)
    all_ctrl = np.stack(ctrl_list, axis=0)             # (n, nu)

    qpos_mean = np.mean(all_qpos, axis=0)
    qpos_std  = np.std(all_qpos, axis=0) + 1e-8
    qvel_mean = np.mean(all_qvel, axis=0)
    qvel_std  = np.std(all_qvel, axis=0) + 1e-8
    ctrl_mean = np.mean(all_ctrl, axis=0)
    ctrl_std  = np.std(all_ctrl, axis=0) + 1e-8

    # Finite-difference accelerations for output normalization
    qacc_list = [(qv[1:] - qv[:-1]) / step_size for qv in qvel_list]
    all_qacc = np.concatenate(qacc_list, axis=0)
    qacc_mean = np.mean(all_qacc, axis=0)
    qacc_std  = np.std(all_qacc, axis=0) + 1e-8

    # Network input stats: masked-qpos ++ qvel ++ ctrl
    input_mean = jnp.concatenate([
        jnp.array(qpos_mean[qpos_input_mask]),
        jnp.array(qvel_mean),
        jnp.array(ctrl_mean),
    ])
    input_std = jnp.concatenate([
        jnp.array(qpos_std[qpos_input_mask]),
        jnp.array(qvel_std),
        jnp.array(ctrl_std),
    ])

    return {
        "input_mean":  input_mean,
        "input_std":   input_std,
        "output_mean": jnp.array(qvel_mean),
        "output_std":  jnp.array(qvel_std),
        "qpos_std":    jnp.array(qpos_std),
        "qvel_std":    jnp.array(qvel_std),
    }


# ── Save / load model ──────────────────────────────────────────────────────

def get_unique_name(name, base_dir="Models"):
    """Return *name* if unused, otherwise append an incrementing suffix."""
    path = os.path.join(os.getcwd(), base_dir, name)
    if not os.path.exists(path):
        return name
    i = 1
    while os.path.exists(f"{path}_{i}"):
        i += 1
    return f"{name}_{i}"


def save_model(model, loss_list, model_options, training_options, norm_stats):
    name = get_unique_name(training_options["name"])
    save_path = os.path.join(os.getcwd(), "Models", name)
    os.makedirs(save_path)

    eqx.tree_serialise_leaves(os.path.join(save_path, "model.eqx"), model)

    for fname, data in [
        ("model_options.json", model_options),
        ("training_options.json", training_options),
        ("loss_list.json", loss_list),
    ]:
        with open(os.path.join(save_path, fname), "w") as f:
            json.dump(data, f)

    serializable_stats = {k: v.tolist() for k, v in norm_stats.items()}
    with open(os.path.join(save_path, "norm_stats.json"), "w") as f:
        json.dump(serializable_stats, f)

    return save_path


def load_model(model_options, training_options):
    """Build (or reload) the KNODE model and compute normalization stats."""
    dataset_path = f"Datasets/{training_options['dataset_name']}.zarr"
    model_xml = mj_utils.create_XML_From_Options(training_options)
    mjx_model, mjx_data = mj_utils.init_mjx_model(model_xml)

    nq = mjx_model.nq
    mask_spec = training_options.get("qpos_input_mask")
    qpos_input_mask = jnp.array(mask_spec if mask_spec is not None else list(range(nq)))

    key = jr.key(int(time.time()))

    if training_options["should_load"]:
        load_path = os.path.join(os.getcwd(), "Models", training_options["load_name"])
        model_options = parse_specs(os.path.join(load_path, "model_options.json"))

        with open(os.path.join(load_path, "norm_stats.json")) as f:
            norm_stats = {k: jnp.array(v) for k, v in json.load(f).items()}

        model = KNODE(key, mjx_model, model_options["width"], model_options["depth"],
                      norm_stats, qpos_input_mask)
        model = eqx.tree_deserialise_leaves(os.path.join(load_path, "model.eqx"), model)
    else:
        norm_stats = compute_normalization_stats(
            dataset_path, training_options["subset_size"], np.array(qpos_input_mask)
        )
        model = KNODE(key, mjx_model, model_options["width"], model_options["depth"],
                      norm_stats, qpos_input_mask)

    return model, mjx_data, norm_stats


# ── Visualization ───────────────────────────────────────────────────────────

def visualize_trained_model(model, training_options, loss_list, norm_stats):
    dataset_path = f"Datasets/{training_options['dataset_name']}.zarr"
    root = zarr.open(dataset_path, mode="r")
    dt = float(root.attrs["dataset_options"]["step_size"])
    xml = root.attrs.get("xml")

    mj_model, _ = mj_utils.init_mujoco_model(xml)
    _, mjx_data = mj_utils.init_mjx_model(xml)

    keys = sorted(root.keys(), key=lambda k: int(k.split("_")[-1]))
    traj_key = keys[training_options["which_to_show_at_end"]]
    d = root[traj_key]
    print(f"Showing trajectory: {traj_key}")

    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0, dt * num_steps, step=dt)

    init_states = {"qpos": jnp.array(d["qpos"][0, :]), "qvel": jnp.array(d["qvel"][0, :])}
    ctrl = jnp.array(d["ctrl"][0, :])

    knode_traj   = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(1.0))
    nominal_traj = model(ts, mjx_data, init_states, ctrl, with_learned=jnp.array(0.0))

    gt_qpos = d["qpos"][:]
    gt_qvel = d["qvel"][:]
    qpos_std = norm_stats["qpos_std"]
    qvel_std = norm_stats["qvel_std"]

    def normalized_loss(pred, gt_q, gt_v):
        return (jnp.mean(((pred["qpos"] - gt_q) / qpos_std) ** 2)
              + jnp.mean(((pred["qvel"] - gt_v) / qvel_std) ** 2))

    print(f"Learned Loss:  {normalized_loss(knode_traj, gt_qpos, gt_qvel)}")
    print(f"Nominal Loss:  {normalized_loss(nominal_traj, gt_qpos, gt_qvel)}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    loss_inds = np.arange(1, len(loss_list) + 1)

    axes[0].plot(loss_inds, loss_list);              axes[0].set_title("Loss")
    axes[1].loglog(loss_inds, loss_list);             axes[1].set_title("Loss (log-log)")

    for ax, key, title in [(axes[2], "qpos", "qpos"), (axes[3], "qvel", "qvel")]:
        ax.plot(np.array(knode_traj[key]),   color="green", label="KNODE")
        ax.plot(np.array(nominal_traj[key]), color="red",   label="Nominal")
        ax.plot(d[key][:],                   color="black", label="GT")
        ax.set_title(title); ax.legend()

    plt.tight_layout()
    plt.show()

    mj_utils.visualize_GT_vs_Nominal_vs_KNODE_Trajectory(
        training_options["model_name"], gt_qpos,
        np.array(nominal_traj["qpos"]), np.array(knode_traj["qpos"]),
    )


# ── Training ────────────────────────────────────────────────────────────────

def prefetch(loader, size=5):
    """Prefetch batches onto device to overlap data transfer and compute."""
    queue = collections.deque()
    def enqueue(n):
        for _ in range(n):
            queue.append(jtu.tree_map(jax.device_put, next(loader)))
    enqueue(size)
    while queue:
        yield queue.popleft()
        enqueue(1)


def trajectory_train(model, mjx_data, num_batches, steps_strategy, lr,
                     training_options, filter_spec, loss_list, norm_stats):

    dataset_path = f"Datasets/{training_options['dataset_name']}.zarr"
    root = zarr.open(dataset_path, mode="r")
    step_size = float(root.attrs["dataset_options"]["step_size"])
    num_steps = root.attrs["dataset_options"]["num_steps"]
    ts = jnp.arange(0, step_size * num_steps, step=step_size)

    dataset = data_utils.zarr_Trajectory_Dataset(dataset_path, training_options["subset_size"])
    dataset = (dataset
               .shuffle(seed=int(time.time()))
               .batch(batch_size=training_options["batch_size"], drop_remainder=True)
               .repeat())
    iter_loader = prefetch(iter(dataset.to_iter_dataset()))

    optim = optax.adam(lr)
    diff_model, static_model = eqx.partition(model, filter_spec)
    opt_state = optim.init(eqx.filter(diff_model, eqx.is_inexact_array))

    qpos_std = norm_stats["qpos_std"]
    qvel_std = norm_stats["qvel_std"]

    @eqx.filter_value_and_grad
    def grad_loss(diff_model, static_model, ts, gt_trajs, ctrls, mjx_data, init_states):
        model = eqx.combine(diff_model, static_model)
        pred_trajs = jax.vmap(model, (None, None, 0, 0, None))(
            ts, mjx_data, init_states, ctrls, jnp.array(1.0)
        )
        qpos_err = (pred_trajs["qpos"] - gt_trajs["qpos"]) / qpos_std
        qvel_err = (pred_trajs["qvel"] - gt_trajs["qvel"]) / qvel_std
        return jnp.mean(qpos_err ** 2) + jnp.mean(qvel_err ** 2)

    @eqx.filter_jit
    def make_step(model, ts, gt_trajs, ctrls, mjx_data, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        init_states = {
            "qpos": gt_trajs["qpos"][:, 0, :],
            "qvel": gt_trajs["qvel"][:, 0, :],
        }
        loss, grads = grad_loss(diff_model, static_model, ts, gt_trajs,
                                ctrls, mjx_data, init_states)
        updates, opt_state = optim.update(grads, opt_state)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = eqx.combine(diff_model, static_model)
        return loss, model, opt_state

    _num_steps = int(num_steps * steps_strategy)

    for batch_id in range(num_batches):
        t1 = time.time()
        gt_trajs, ctrls = next(iter_loader)
        gt_trajs = jtu.tree_map(jax.device_put, gt_trajs)
        ctrls = jax.device_put(ctrls)

        _ts = ts[:_num_steps]
        _gt_trajs = {
            "qpos": gt_trajs["qpos"][:, :_num_steps, :],
            "qvel": gt_trajs["qvel"][:, :_num_steps, :],
        }
        loss, model, opt_state = make_step(model, _ts, _gt_trajs, ctrls, mjx_data, opt_state)
        loss_list.append(loss.item())

        if batch_id % training_options["print_every"] == 0 or (batch_id + 1) == num_batches:
            print(f"Batch: {batch_id + 1}/{num_batches}  "
                  f"Loss: {loss:.6f}  Time: {time.time() - t1:.2f}s")

    return model, loss_list


def train_model(model_options, training_options):
    model, mjx_data, norm_stats = load_model(model_options, training_options)
    loss_list = []
    
#    print(norm_stats)
#    poop()

    # Only train the residual MLP and the mixing matrices
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda t: t.knode_dynamics.residual_network.mlp, filter_spec, replace=True
    )
#    filter_spec = eqx.tree_at(
#        lambda t: t.knode_dynamics.mjx_mixing_matrix, filter_spec, replace=True
#    )
    filter_spec = eqx.tree_at(
        lambda t: t.knode_dynamics.res_mixing_matrix, filter_spec, replace=True
    )

    for num_batches, steps_strategy, lr in zip(
        training_options["num_batches_strategy"],
        training_options["steps_strategy"],
        training_options["lr_strategy"],
    ):
        model, loss_list = trajectory_train(
            model, mjx_data, num_batches, steps_strategy, lr,
            training_options, filter_spec, loss_list, norm_stats,
        )

    return model, loss_list, norm_stats


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_path, training_path = return_Arguments()
    model_options = parse_specs(model_path)
    training_options = parse_specs(training_path)

    model, loss_list, norm_stats = train_model(model_options, training_options)

    if training_options["should_save"]:
        save_model(model, loss_list, model_options, training_options, norm_stats)
    if training_options["should_plot"]:
        visualize_trained_model(model, training_options, loss_list, norm_stats)
