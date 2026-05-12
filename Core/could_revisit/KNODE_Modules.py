import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx
import Utils.MJX_Diff as mjx_diff


class Residual_Network(eqx.Module):
    """Neural network that predicts residual acceleration corrections.

    Inputs and outputs are normalized using mean/std statistics computed
    from a batch of training data (passed in at construction time).
    """

    mlp: eqx.nn.MLP
    input_mean:  jax.Array
    input_std:   jax.Array
    output_mean: jax.Array
    output_std:  jax.Array

    def __init__(self, input_num, output_num, width, depth,
                 input_mean, input_std, output_mean, output_std, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=input_num,
            out_size=output_num,
            width_size=width,
            depth=depth,
            activation=jax.nn.gelu,
            key=key,
        )
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def __call__(self, y):
        normalized = (y - self.input_mean) / self.input_std
        raw_output = self.mlp(normalized)
        return raw_output * self.output_std + self.output_mean


class KNODE_Dynamics(eqx.Module):
    """Combined MuJoCo + learned residual dynamics."""

    residual_network: Residual_Network
    mjx_mixing_matrix: jax.Array
    res_mixing_matrix: jax.Array
    mjx_model: mjx.Model
    qpos_input_mask: jax.Array = eqx.field(static=True)

    def __init__(self, mjx_model, network_args, qpos_input_mask, key):
        keys = jr.split(key, 3)
        self.mjx_model = mjx_model
        self.residual_network = Residual_Network(*network_args, key=keys[0])

        output_num = network_args[1]
        self.res_mixing_matrix = jr.uniform(
            keys[1], (output_num, output_num), minval=-1.0, maxval=1.0
        )
        self.mjx_mixing_matrix = jnp.eye(output_num) + jr.uniform(
            keys[2], (output_num, output_num), minval=-0.1, maxval=0.1
        )
        self.qpos_input_mask = qpos_input_mask

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        mjx_data, ctrl, with_learned = args

        # Mask qpos before feeding to the residual network
        qpos_masked = qpos[self.qpos_input_mask]
        res_input = jnp.concatenate((qpos_masked, qvel, ctrl))
        res_state_vel = with_learned * self.residual_network(res_input)

        # MuJoCo forward dynamics
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        mjx_data = mjx_diff.forward(self.mjx_model, mjx_data)

        final_qacc = mjx_data.qacc + res_state_vel @ self.res_mixing_matrix

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE(eqx.Module):
    """Knowledge-embedded Neural ODE model.

    Args:
        key:              PRNG key.
        mjx_model:        Compiled MJX model.
        width:            Hidden layer width.
        depth:            Network depth.
        norm_stats:       Dict with keys ``input_mean``, ``input_std``,
                          ``output_mean``, ``output_std`` (jax arrays).
        qpos_input_mask:  Integer index array selecting which qpos
                          dimensions are fed to the residual network.
    """

    knode_dynamics: KNODE_Dynamics

    def __init__(self, key, mjx_model, width, depth, norm_stats, qpos_input_mask):
        output_num = mjx_model.nq
        input_num = len(qpos_input_mask) + output_num + mjx_model.nu

        network_args = (
            input_num, output_num, width, depth,
            norm_stats["input_mean"], norm_stats["input_std"],
            norm_stats["output_mean"], norm_stats["output_std"],
        )

        self.knode_dynamics = KNODE_Dynamics(mjx_model, network_args, qpos_input_mask, key)

    def __call__(self, ts, mjx_data, init_state, ctrl, with_learned=jnp.array(1.0)):
        mjx_data = mjx_data.replace(
            qpos=init_state["qpos"], qvel=init_state["qvel"], ctrl=ctrl
        )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics),
            diffrax.Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=init_state,
            args=(mjx_data, ctrl, with_learned),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
