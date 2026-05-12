"""
GP_Modules.py
=============
Gaussian Process residual model for KNODE.

Each output DOF gets an independent exact GP with an ARD RBF kernel.
Hyperparameters (lengthscales, signal variance, noise variance) are
optimised by maximising the marginal log-likelihood.

At prediction time the precomputed alpha vectors (K⁻¹y) make inference
O(n_train) per output — fast enough for use inside a diffrax ODE loop.

Architecture
------------
  kernel(x, x') = σ² exp(-½ Σ_d (x_d - x'_d)² / ℓ_d²)
  α  = (K + σ_n²I)⁻¹ y          — precomputed once after fitting
  μ* = k(x*, X_train)ᵀ α         — prediction

Drop-in classes
---------------
  GP_Residual          — same __call__ signature as Residual_Network / Fourier_SINDy_Residual
  KNODE_GP_Dynamics    — diffrax-compatible dynamics
  KNODE_GP             — same interface as KNODE / KNODE_Fourier_SINDy
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from mujoco import mjx

import Utils.MJX_Diff as mjx_diff


# ============================================================================
# KERNEL HELPERS
# ============================================================================

def rbf_kernel_vector(x, X_train, log_lengthscales, log_signal_var):
    """
    Compute k(x, X_train) for a single test point.

    Parameters
    ----------
    x             : (input_dim,)
    X_train       : (n_train, input_dim)   — already normalised
    log_lengthscales : (input_dim,)
    log_signal_var   : scalar

    Returns
    -------
    k : (n_train,)
    """
    ls = jnp.exp(log_lengthscales)
    sv = jnp.exp(log_signal_var)
    diff = (x - X_train) / ls                        # (n_train, input_dim)
    return sv * jnp.exp(-0.5 * jnp.sum(diff ** 2, axis=-1))   # (n_train,)


def rbf_kernel_matrix(X, log_lengthscales, log_signal_var):
    """
    Full kernel matrix K(X, X).

    Parameters
    ----------
    X                : (n, input_dim)
    log_lengthscales : (input_dim,)
    log_signal_var   : scalar

    Returns
    -------
    K : (n, n)
    """
    ls = jnp.exp(log_lengthscales)
    sv = jnp.exp(log_signal_var)
    X_scaled = X / ls                                 # (n, input_dim)
    sq_dists = jnp.sum(X_scaled ** 2, axis=-1, keepdims=True) \
               - 2.0 * X_scaled @ X_scaled.T \
               + jnp.sum(X_scaled ** 2, axis=-1, keepdims=True).T
    sq_dists = jnp.maximum(sq_dists, 0.0)             # numerical safety
    return sv * jnp.exp(-0.5 * sq_dists)


# ============================================================================
# MARGINAL LOG-LIKELIHOOD
# ============================================================================

def neg_log_marginal_likelihood(log_lengthscales, log_signal_var, log_noise_var,
                                X_train, y_train, jitter=1e-6):
    """
    Negative log marginal likelihood for a single-output GP.

    Uses Cholesky decomposition for numerical stability.
    """
    n = X_train.shape[0]
    K = rbf_kernel_matrix(X_train, log_lengthscales, log_signal_var)
    K_noisy = K + (jnp.exp(log_noise_var) + jitter) * jnp.eye(n)

    L = jnp.linalg.cholesky(K_noisy)
    alpha = jax.scipy.linalg.cho_solve((L, True), y_train)

    # -log p(y|X) = 0.5 yᵀα + Σ log L_ii + n/2 log(2π)
    nlml = (0.5 * y_train @ alpha
            + jnp.sum(jnp.log(jnp.diag(L)))
            + 0.5 * n * jnp.log(2.0 * jnp.pi))
    return nlml


# ============================================================================
# GP RESIDUAL MODEL
# ============================================================================

class GP_Residual(eqx.Module):
    """
    Multi-output GP residual model (independent GP per DOF).

    Stores pre-normalised training data and precomputed alpha vectors
    so that __call__ is cheap (one kernel-vector dot product per output).
    """
    X_train_norm:    jax.Array    # (n_train, input_dim)
    input_norm_add:  jax.Array    # (input_dim,)
    input_norm_mult: jax.Array    # (input_dim,)

    # Per-output hyperparameters  — shape (output_num, ...)
    log_lengthscales: jax.Array   # (output_num, input_dim)
    log_signal_var:   jax.Array   # (output_num,)
    log_noise_var:    jax.Array   # (output_num,)

    # Precomputed prediction weights
    alpha:      jax.Array         # (n_train, output_num)
    output_num: int

    def __init__(self, X_train_norm, input_norm_add, input_norm_mult,
                 log_lengthscales, log_signal_var, log_noise_var,
                 alpha, output_num):
        self.X_train_norm    = X_train_norm
        self.input_norm_add  = input_norm_add
        self.input_norm_mult = input_norm_mult
        self.log_lengthscales = log_lengthscales
        self.log_signal_var   = log_signal_var
        self.log_noise_var    = log_noise_var
        self.alpha      = alpha
        self.output_num = output_num

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x : (input_dim,)  →  residual qacc : (output_num,).
        """
        x_norm = self.input_norm_mult * (x + self.input_norm_add) - 1.0

        def _predict_one(log_ls, log_sv, alpha_col):
            k = rbf_kernel_vector(x_norm, self.X_train_norm, log_ls, log_sv)
            return k @ alpha_col

        return jax.vmap(_predict_one)(
            self.log_lengthscales,
            self.log_signal_var,
            self.alpha.T,                              # (output_num, n_train)
        )

    def sparsity_report(self) -> dict:
        """Summary statistics (mirrors Fourier_SINDy_Residual interface)."""
        n_train = self.X_train_norm.shape[0]
        return {
            "n_training_points": n_train,
            "output_num":        self.output_num,
            "input_dim":         self.X_train_norm.shape[1],
        }


# ============================================================================
# KNODE WRAPPER
# ============================================================================

class KNODE_GP_Dynamics(eqx.Module):
    """diffrax-compatible dynamics.  Mirrors KNODE_Dynamics / KNODE_Fourier_SINDy_Dynamics."""
    gp_residual: GP_Residual
    mjx_model:   mjx.Model

    def __init__(self, mjx_model, residual):
        self.mjx_model   = mjx_model
        self.gp_residual = residual

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        mjx_data, ctrl, with_learned = args

        mjx_data   = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        res_qacc   = with_learned * self.gp_residual(
            jnp.concatenate([qpos, qvel, ctrl])
        )
        mjx_data   = mjx_diff.forward(self.mjx_model, mjx_data)
        final_qacc = mjx_data.qacc + res_qacc

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE_GP(eqx.Module):
    """Full KNODE model with GP residual.  Same interface as KNODE / KNODE_Fourier_SINDy."""
    knode_dynamics: KNODE_GP_Dynamics

    def __init__(self, mjx_model, residual):
        self.knode_dynamics = KNODE_GP_Dynamics(mjx_model, residual)

    def __call__(self, ts, mjx_data, init_state, ctrl,
                 with_learned=jnp.array(1.0)):
        import diffrax
        mjx_data = mjx_data.replace(
            qpos=init_state["qpos"], qvel=init_state["qvel"], ctrl=ctrl
        )
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.knode_dynamics),
            diffrax.Euler(),
            t0=ts[0], t1=ts[-1], dt0=ts[1] - ts[0],
            y0=init_state,
            args=(mjx_data, ctrl, with_learned),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


# ============================================================================
# FITTING
# ============================================================================

def _build_normalisation(mjx_model, qvel_max):
    """Same normalisation convention as Residual_Network / Fourier_SINDy_Library."""
    qpos_ranges = mjx_model.jnt_range
    ctrl_ranges = mjx_model.actuator_ctrlrange

    qpos_add  = -qpos_ranges[:, 0]
    qvel_add  =  qvel_max
    ctrl_add  = -ctrl_ranges[:, 0]
    qpos_mult =  2.0 / (qpos_ranges[:, 1] - qpos_ranges[:, 0])
    qvel_mult =  1.0 / qvel_max
    ctrl_mult =  2.0 / (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])

    norm_add  = jnp.concatenate([qpos_add,  qvel_add,  ctrl_add])
    norm_mult = jnp.concatenate([qpos_mult, qvel_mult, ctrl_mult])
    return norm_add, norm_mult


def fit_single_GP(X_norm, y, lr, num_iters, jitter, key, verbose=True, dof_id=0):
    """
    Optimise hyperparameters for a single-output GP via marginal log-likelihood.

    Parameters
    ----------
    X_norm   : (n, d) normalised training inputs
    y        : (n,)   target residual accelerations for one DOF
    lr       : learning rate for Adam
    num_iters: number of gradient-descent steps
    jitter   : diagonal jitter for Cholesky stability
    key      : PRNG key
    verbose  : print progress

    Returns
    -------
    log_lengthscales, log_signal_var, log_noise_var, alpha
    """
    n, d = X_norm.shape

    # Initialise hyperparameters
    log_ls = jnp.zeros(d)                              # lengthscales = 1
    log_sv = jnp.log(jnp.var(y) + 1e-8)               # signal var ≈ data variance
    log_nv = jnp.log(0.01 * jnp.var(y) + 1e-8)        # noise var ≈ 1% of data var

    params = {"log_ls": log_ls, "log_sv": log_sv, "log_nv": log_nv}
    optim  = optax.adam(lr)
    opt_state = optim.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            return neg_log_marginal_likelihood(
                p["log_ls"], p["log_sv"], p["log_nv"],
                X_norm, y, jitter
            )
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optim.update(grads, opt_state)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    for i in range(num_iters):
        params, opt_state, loss = step(params, opt_state)
        if verbose and (i % max(1, num_iters // 10) == 0 or i == num_iters - 1):
            print(f"    DOF {dof_id} | iter {i+1:5d}/{num_iters} | NLML = {loss:.4f}")

    # Precompute alpha = K⁻¹y
    K = rbf_kernel_matrix(X_norm, params["log_ls"], params["log_sv"])
    K_noisy = K + (jnp.exp(params["log_nv"]) + jitter) * jnp.eye(n)
    L = jnp.linalg.cholesky(K_noisy)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)

    return params["log_ls"], params["log_sv"], params["log_nv"], alpha


def fit_GP(mjx_model, qvel_max,
           qpos_arr, qvel_arr, ctrl_arr, residual_qacc,
           num_training_points=2000,
           lr=0.01, num_iters=500, jitter=1e-6,
           key=None, verbose=True):
    """
    Fit independent GPs for each output DOF.

    Parameters
    ----------
    mjx_model          : MJX model (for normalisation ranges)
    qvel_max           : max joint velocities from dataset
    qpos_arr           : (N, nq) all available qpos samples
    qvel_arr           : (N, nv) all available qvel samples
    ctrl_arr           : (N, nu) all available ctrl samples
    residual_qacc      : (N, nq) target residual accelerations
    num_training_points: how many points to keep for the GP
    lr                 : Adam learning rate for hyperparameters
    num_iters          : gradient-descent iterations per DOF
    jitter             : Cholesky diagonal jitter
    key                : PRNG key
    verbose            : print progress

    Returns
    -------
    GP_Residual with precomputed alpha vectors
    """
    if key is None:
        key = jr.key(0)

    output_num = mjx_model.nq
    N = qpos_arr.shape[0]

    # ---- Sub-sample training points ----
    if num_training_points < N:
        key, subkey = jr.split(key)
        idx = jr.choice(subkey, N, shape=(num_training_points,), replace=False)
        idx = jnp.sort(idx)
        qpos_train = qpos_arr[idx]
        qvel_train = qvel_arr[idx]
        ctrl_train = ctrl_arr[idx]
        res_train  = residual_qacc[idx]
        print(f"  Sub-sampled {num_training_points} training points from {N} available.")
    else:
        qpos_train = qpos_arr
        qvel_train = qvel_arr
        ctrl_train = ctrl_arr
        res_train  = residual_qacc
        num_training_points = N
        print(f"  Using all {N} points for GP training.")

    # ---- Normalise inputs ----
    norm_add, norm_mult = _build_normalisation(mjx_model, qvel_max)
    X_raw = jnp.concatenate([jnp.array(qpos_train),
                              jnp.array(qvel_train),
                              jnp.array(ctrl_train)], axis=-1)
    X_norm = norm_mult * (X_raw + norm_add) - 1.0

    Y = jnp.array(res_train)

    print(f"\n{'='*60}")
    print(f"  Gaussian Process Fitting")
    print(f"  training points : {num_training_points}")
    print(f"  input dim       : {X_norm.shape[1]}")
    print(f"  outputs (DOFs)  : {output_num}")
    print(f"  optimiser       : Adam (lr={lr}, {num_iters} iters)")
    print(f"  jitter          : {jitter}")
    print(f"{'='*60}\n")

    # ---- Fit each DOF independently ----
    all_log_ls = []
    all_log_sv = []
    all_log_nv = []
    all_alpha  = []

    for j in range(output_num):
        key, subkey = jr.split(key)
        print(f"  Fitting DOF {j}/{output_num - 1} ...")
        log_ls, log_sv, log_nv, alpha = fit_single_GP(
            X_norm, Y[:, j],
            lr=lr, num_iters=num_iters, jitter=jitter,
            key=subkey, verbose=verbose, dof_id=j,
        )
        all_log_ls.append(log_ls)
        all_log_sv.append(log_sv)
        all_log_nv.append(log_nv)
        all_alpha.append(alpha)
        print()

    log_lengthscales = jnp.stack(all_log_ls)            # (output_num, input_dim)
    log_signal_var   = jnp.stack(all_log_sv)             # (output_num,)
    log_noise_var    = jnp.stack(all_log_nv)             # (output_num,)
    alpha_matrix     = jnp.stack(all_alpha, axis=-1)     # (n_train, output_num)

    residual = GP_Residual(
        X_train_norm    = X_norm,
        input_norm_add  = norm_add,
        input_norm_mult = norm_mult,
        log_lengthscales = log_lengthscales,
        log_signal_var   = log_signal_var,
        log_noise_var    = log_noise_var,
        alpha      = alpha_matrix,
        output_num = output_num,
    )

    print(f"  GP fitting complete.")
    print(f"  Median lengthscales per DOF:")
    for j in range(output_num):
        ls = np.exp(np.array(log_lengthscales[j]))
        print(f"    DOF {j}: median ℓ = {np.median(ls):.4f},  "
              f"signal σ² = {np.exp(float(log_signal_var[j])):.4e},  "
              f"noise σ² = {np.exp(float(log_noise_var[j])):.4e}")

    return residual
