import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from mujoco import mjx
import Utils.MJX_Diff as mjx_diff


# =============================================================================
# Power-law fluid force residual model
# =============================================================================
#
# Replaces the neural-network residuals with a learnable, physically-motivated
# power-law model inspired by MuJoCo's inertial_box fluid model.
#
# The inertial_box model produces forces of the form:
#   f_D[i] ∝ ρ * area[i] * v[i] * |v[i]|     (quadratic drag, power = 2)
#   f_V[i] ∝ β * perimeter[i] * v[i]           (viscous drag,   power = 1)
#
# Here we generalise this to num_powers learnable power-law terms per body.
# For body b, power-law term k, and velocity channel j (j ∈ {wx,wy,wz,vx,vy,vz}):
#
#   contribution[k, j] = coeffs[k, j] * v[j] * (|v[j]| + ε)^(power[k] − 1)
#
# This is the sign-preserving generalisation of |v|^a:
#   v * |v|^(a−1) = sign(v) * |v|^a
# with a small ε for numerical safety near v = 0 (especially when a < 1).
#
# The total 6-DOF wrench output for body b is:
#   F[b, j] = Σ_k  coeffs[b, k, j] * v[j] * (|v[j]| + ε)^(power[b, k] − 1)
#
# Powers are initialised spread linearly from 1.0 to 3.0 so each term starts
# at a distinct operating point (linear → quadratic → cubic).  Coefficients
# are initialised near zero so the model starts close to the nominal MuJoCo
# physics and learns corrections.
# =============================================================================


class Body_Power_Law(eqx.Module):
    """
    Learnable power-law fluid-force residual for a single body.

    Parameters
    ----------
    num_powers : int
        Number of power-law terms to assign to this body.
    key : jr.PRNGKey

    Attributes
    ----------
    coeffs : jax.Array, shape (num_powers, 6)
        Per-term, per-channel learnable scaling coefficients.
        Channel order matches network_inputs: [w_x, w_y, w_z, v_x, v_y, v_z].
        Output channel order: [τ_x, τ_y, τ_z, f_x, f_y, f_z].
    powers : jax.Array, shape (num_powers,)
        Per-term learnable velocity exponents.  Initialised spread from 1→3.
    """

    coeffs: jax.Array   # (num_powers, 6)
    powers: jax.Array   # (num_powers,)
    num_powers: int = eqx.field(static=True)

    def __init__(self, num_powers: int, *, key):
        self.num_powers = num_powers
        k1, k2 = jr.split(key)
        # Small random init keeps the model close to pure MuJoCo at the start
        self.coeffs = jr.normal(k1, (num_powers, 6)) * .01
        # Spread powers evenly so each term covers a different velocity regime
#        self.powers = jr.uniform(k2 , (num_powers,) , minval = 0.5 , maxval = 3.5)#
        self.powers = jnp.linspace(1.0, 3.0, num_powers)

    def __call__(self, v6: jax.Array) -> jax.Array:
        """
        Compute the 6-DOF power-law force/torque for this body.

        Parameters
        ----------
        v6 : jax.Array, shape (6,)
            Body-frame velocity [w_x, w_y, w_z, v_x, v_y, v_z].

        Returns
        -------
        jax.Array, shape (6,)
            Body-frame wrench residual [τ_x, τ_y, τ_z, f_x, f_y, f_z].
        """
        eps = 1e-6
        powers = jnp.abs(self.powers) + .1
        abs_v = jnp.abs(v6)                                          # (6,)
        # |v[j]|^(power[k] - 1)  →  (num_powers, 6)
        abs_v_pow = (abs_v[None, :] + eps) ** (powers[:, None] - 1.0)
        # sign-preserving: coeff * v * |v|^(a-1)
        contributions = -jnp.abs(self.coeffs) * v6[None, :] * abs_v_pow       # (num_powers, 6)
#        contributions = self.coeffs * jnp.sign(v6) * ((abs_v[None, :]) ** (jnp.abs(self.powers[:, None]) + eps ))     # (num_powers, 6)
        return jnp.sum(contributions, axis=0)                        # (6,)


# This class holds the per-body power-law residuals.
# It is a drop-in replacement for the original Residual_Network and exposes
# the same __call__ signature.
class Residual_Network(eqx.Module):
    body_num: int = eqx.field(static=True)
    num_powers: int = eqx.field(static=True)
    # One Body_Power_Law per non-world body (matches the -2 offset of the original)
    body_models: list

    def __init__(self, body_num: int, num_powers: int, *, key):
        keys = jr.split(key, body_num)
        self.body_num = body_num
        self.num_powers = num_powers
        self.body_models = []
        for i in range(body_num - 2):   # CHANGE HERE — matches original -2 offset
            self.body_models.append(Body_Power_Law(num_powers, key=keys[i]))

    def __call__(self, y: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        y : jax.Array, shape (n_bodies_effective, 6)
            Body-frame velocities for each non-world body.

        Returns
        -------
        jax.Array, shape (n_bodies_effective, 6)
            Body-frame wrench residuals.
        """
        vectors = self.body_models[0](y[0])
        vectors = jnp.expand_dims(vectors, 0)
        for i in range(1, self.body_num - 2):   # REMOVE -2 to match original change
            vectors = jnp.concatenate(
                (vectors, jnp.expand_dims(self.body_models[i](y[i]), 0)), 0
            )
        return vectors


# This class holds the combined state dynamics.
# The only change vs. the original is that Residual_Network is now the
# power-law model above, and the constructor takes num_powers instead of
# the MLP width/depth/normalisation arguments.
class KNODE_Dynamics(eqx.Module):
    residual_network: Residual_Network
    mjx_model: mjx.Model

    def __init__(self, mjx_model, num_powers: int, key):
        self.mjx_model = mjx_model
        self.residual_network = Residual_Network(
            mjx_model.nbody, num_powers, key=key
        )

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]

        mjx_data, ctrl, with_learned = args
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        mjx_data = mjx_diff.fwd_position(self.mjx_model, mjx_data)
        mjx_data = mjx_diff.fwd_velocity(self.mjx_model, mjx_data)

        cvel = mjx_data.cvel
        world_w = cvel[2:, :3]            # REMOVE 2
        world_v = cvel[2:, 3:]            # REMOVE 2

        bf_to_world_mat = mjx_data.ximat[2:, :, :]   # REMOVE 2
        world_to_bf_mat = jnp.matrix_transpose(bf_to_world_mat)

        matmul = lambda mat, x: mat @ x
        matmul_mapped = jax.vmap(matmul, (0, 0))

        bf_w = matmul_mapped(world_to_bf_mat, world_w)
        bf_v = matmul_mapped(world_to_bf_mat, world_v)

        # Input to residual: body-frame [angular vel | linear vel]
        network_inputs = jnp.concatenate((bf_w, bf_v), 1)   # (n_bodies_eff, 6)

        # Power-law residual forces/torques in body frame
        vectors = self.residual_network(network_inputs) * with_learned
        bf_torque = vectors[:, :3]
        bf_force  = vectors[:, 3:]

        world_torque = matmul_mapped(bf_to_world_mat, bf_torque)
        world_force  = matmul_mapped(bf_to_world_mat, bf_force)

        world_6_force = jnp.concatenate((world_torque, world_force), 1)

        pad = jnp.zeros((2, 6))                                          # REMOVE
        world_6_force = jnp.concatenate((pad, world_6_force), axis=0)   # REMOVE

        mjx_data = mjx_data.replace(xfrc_applied=world_6_force)

        mjx_data = mjx_diff.forward(self.mjx_model, mjx_data)
        mjx_qacc = mjx_data.qacc

        final_qacc = mjx_qacc

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE(eqx.Module):
    knode_dynamics: KNODE_Dynamics

    def __init__(self, key, mjx_model, num_powers: int, qvel_max=None):
        """
        Parameters
        ----------
        key        : PRNGKey
        mjx_model  : mjx.Model
        num_powers : int
            Number of power-law terms per body (from model_options["num_powers"]).
        qvel_max   : unused — kept for API symmetry with the original if desired.
        """
        self.knode_dynamics = KNODE_Dynamics(mjx_model, num_powers, key)

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
