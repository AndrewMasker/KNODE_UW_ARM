import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx
import Utils.MJX_Diff as mjx_diff


class Fourier_Residual_Network(eqx.Module):
    coeff_matrix:       jax.Array  # FIXED  - shape (num_features, input_dim)
    output_matrix:      jax.Array  # TRAINED - shape (output_num, 2*num_features)
    input_norm_add:     jax.Array
    input_norm_mult:    jax.Array
    output_denorm_add:  jax.Array
    output_denorm_mult: jax.Array

    def __init__(self, input_num, output_num, num_features, coeff_std,
                 qpos_ranges, qvel_max, ctrl_ranges, *, key):
        keys = jr.split(key, 2)

        # Fixed random Fourier feature frequency matrix
        # Sampled from N(0, coeff_std^2), never updated
        self.coeff_matrix = jr.normal(keys[0], (num_features, input_num)) * coeff_std

        # Trainable linear readout: (output_num, 2*num_features)
        # Initialized small so residual starts near zero
        self.output_matrix = jr.normal(keys[1], (output_num, 2 * num_features)) * 0.01

        # Input normalization (identical convention to original)
        qpos_add  = -qpos_ranges[:, 0]
        qvel_add  =  qvel_max
        ctrl_add  = -ctrl_ranges[:, 0]
        qpos_mult =  2 / (qpos_ranges[:, 1] - qpos_ranges[:, 0])
        qvel_mult =  1 / qvel_max
        ctrl_mult =  2 / (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])

        self.input_norm_add   = jnp.concatenate((qpos_add,  qvel_add,  ctrl_add),  axis=0)
        self.input_norm_mult  = jnp.concatenate((qpos_mult, qvel_mult, ctrl_mult), axis=0)
        self.output_denorm_add  = -qvel_add
        self.output_denorm_mult =  1 / qvel_mult

    def __call__(self, y):
        # Normalize to [-1, 1]
        x = self.input_norm_mult * (y + self.input_norm_add) - 1  # (input_dim,)

        # Random Fourier Features
        proj     = self.coeff_matrix @ x                           # (num_features,)
        features = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])# (2*num_features,)

        # Linear readout
        out = self.output_matrix @ features                        # (output_num,)

        # Denormalize to acceleration units
        return (out + 1) * self.output_denorm_mult + self.output_denorm_add


class KNODE_Dynamics(eqx.Module):
    residual_network: Fourier_Residual_Network
    mjx_mixing_matrix: jax.Array
    res_mixing_matrix:  jax.Array
    mjx_model: mjx.Model

    def __init__(self, mjx_model, network_args, key):
        keys = jr.split(key, 3)
        self.mjx_model        = mjx_model
        self.residual_network = Fourier_Residual_Network(*network_args, key=keys[0])

        output_num = network_args[1]
        self.res_mixing_matrix  = jr.uniform(keys[1], (output_num, output_num),
                                             minval=-1.0, maxval=1.0)
        self.mjx_mixing_matrix  = (jnp.eye(output_num)
                                   + jr.uniform(keys[2], (output_num, output_num),
                                                minval=-0.1, maxval=0.1))

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]

        mjx_data, ctrl, with_learned = args
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        res_state_vel = with_learned * self.residual_network(
            jnp.concatenate((qpos, qvel, ctrl)))
        mjx_data  = mjx_diff.forward(self.mjx_model, mjx_data)
        mjx_qacc  = mjx_data.qacc

        final_qacc = mjx_qacc + res_state_vel @ self.res_mixing_matrix

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics

    def __init__(self, key, mjx_model, num_features, coeff_std, qvel_max):
        """
        Parameters
        ----------
        key          : jr.PRNGKey
        mjx_model    : mjx.Model
        num_features : int   — number of Fourier features (from model_specs)
        coeff_std    : float — std of the fixed frequency matrix (from model_specs)
        qvel_max     : jnp.ndarray — per-joint velocity max for normalization
        """
        output_num = mjx_model.nq
        input_num  = 2 * output_num + mjx_model.nu

        qpos_ranges = mjx_model.jnt_range
        ctrl_ranges = mjx_model.actuator_ctrlrange

        network_args = (input_num, output_num, num_features, coeff_std,
                        qpos_ranges, qvel_max, ctrl_ranges)

        self.knode_dynamics = KNODE_Dynamics(mjx_model, network_args, key)

    def __call__(self, ts, mjx_data, init_state, ctrl, with_learned=jnp.array(1.0)):
        mjx_data = mjx_data.replace(
            qpos=init_state["qpos"], qvel=init_state["qvel"], ctrl=ctrl)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics),
            diffrax.Euler(),
            t0  = ts[0],
            t1  = ts[-1],
            dt0 = ts[1] - ts[0],
            y0  = init_state,
            args    = (mjx_data, ctrl, with_learned),
            saveat  = diffrax.SaveAt(ts=ts)
        )
        return solution.ys
