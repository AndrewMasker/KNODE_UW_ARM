"""
Fourier_SINDy_Modules.py
========================
SINDy with a Random Fourier Feature (RFF) library.

The library Θ(x) = [sin(Wx), cos(Wx)]  where W ~ N(0, coeff_std²) is fixed
at construction.  The coefficient matrix Ξ is then fit via STLSQ or SR3.

Architecture
------------
  Θ(x̃)   : (2 * num_features,)   — RFF library evaluated at normalised input
  Ξ       : (2 * num_features, nq) — sparse coefficient matrix (fitted)
  output  : Θ @ Ξ                  — residual qacc in physical units

Drop-in classes
---------------
  Fourier_SINDy_Residual   — same __call__ signature as Residual_Network
  KNODE_Fourier_SINDy      — same interface as KNODE / KNODE_SINDy
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from mujoco import mjx

import Utils.MJX_Diff as mjx_diff


# ============================================================================
# LIBRARY
# ============================================================================

class Fourier_SINDy_Library(eqx.Module):
    """
    Fixed Random Fourier Feature library.

    W ~ N(0, coeff_std²), shape (num_features, input_dim).
    __call__(x) → [sin(Wx̃), cos(Wx̃)], shape (2*num_features,).
    Input is normalised to ≈ [-1, 1] before projection.
    """
    coeff_matrix:    jax.Array   # (num_features, input_dim)  — fixed, not trained
    input_norm_add:  jax.Array   # (input_dim,)
    input_norm_mult: jax.Array   # (input_dim,)
    num_features:        int
    n_library_features:  int     # = 2 * num_features

    def __init__(
        self,
        input_dim:   int,
        num_features: int,
        coeff_std:   float,
        qpos_ranges: jax.Array,   # (nq, 2)
        qvel_max:    jax.Array,   # (nv,)
        ctrl_ranges: jax.Array,   # (nu, 2)
        *,
        key: jr.PRNGKey,
    ):
        self.num_features       = num_features
        self.n_library_features = 2 * num_features

        self.coeff_matrix = jr.normal(key, (num_features, input_dim)) * coeff_std

        # Same normalisation convention as Fourier_Modules.py
        qpos_add  = -qpos_ranges[:, 0]
        qvel_add  =  qvel_max
        ctrl_add  = -ctrl_ranges[:, 0]
        qpos_mult =  2.0 / (qpos_ranges[:, 1] - qpos_ranges[:, 0])
        qvel_mult =  1.0 / qvel_max
        ctrl_mult =  2.0 / (ctrl_ranges[:, 1] - ctrl_ranges[:, 0])

        self.input_norm_add  = jnp.concatenate([qpos_add,  qvel_add,  ctrl_add])
        self.input_norm_mult = jnp.concatenate([qpos_mult, qvel_mult, ctrl_mult])

    def __call__(self, x: jax.Array) -> jax.Array:
        x_norm = self.input_norm_mult * (x + self.input_norm_add) - 1.0
        proj   = self.coeff_matrix @ x_norm          # (num_features,)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])  # (2*num_features,)


# ============================================================================
# RESIDUAL MODEL
# ============================================================================

class Fourier_SINDy_Residual(eqx.Module):
    """
    Sparse residual model:  residual_qacc(x) = Θ(x) @ Ξ.

    Ξ is fitted via STLSQ / SR3 (see fit_Fourier_SINDy below).
    Stored as a differentiable JAX array for eqx serialisation; not updated
    by gradient descent.
    """
    library:      Fourier_SINDy_Library
    coefficients: jax.Array   # (2*num_features, output_num)
    output_num:   int

    def __init__(
        self,
        library:      Fourier_SINDy_Library,
        output_num:   int,
        coefficients: Optional[jax.Array] = None,
    ):
        self.library    = library
        self.output_num = output_num
        self.coefficients = (
            jnp.zeros((library.n_library_features, output_num))
            if coefficients is None
            else jnp.asarray(coefficients)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """x : (input_dim,) → residual qacc : (output_num,)."""
        return self.library(x) @ self.coefficients

    def sparsity_report(self) -> dict:
        xi       = np.array(self.coefficients)
        n_total  = xi.size
        n_nz     = int(np.sum(np.abs(xi) > 0))
        return {
            "n_total":    n_total,
            "n_nonzero":  n_nz,
            "n_zero":     n_total - n_nz,
            "sparsity_%": 100.0 * (1.0 - n_nz / n_total),
        }


# ============================================================================
# KNODE WRAPPER
# ============================================================================

class KNODE_Fourier_SINDy_Dynamics(eqx.Module):
    """Mirrors KNODE_Dynamics / KNODE_SINDy_Dynamics. diffrax-compatible."""
    fourier_sindy_residual: Fourier_SINDy_Residual
    mjx_model:              mjx.Model

    def __init__(self, mjx_model: mjx.Model, residual: Fourier_SINDy_Residual):
        self.mjx_model              = mjx_model
        self.fourier_sindy_residual = residual

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        mjx_data, ctrl, with_learned = args

        mjx_data   = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        res_qacc   = with_learned * self.fourier_sindy_residual(
            jnp.concatenate([qpos, qvel, ctrl])
        )
        mjx_data   = mjx_diff.forward(self.mjx_model, mjx_data)
        final_qacc = mjx_data.qacc + res_qacc

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE_Fourier_SINDy(eqx.Module):
    """Full KNODE model with Fourier-SINDy residual. Same interface as KNODE."""
    knode_dynamics: KNODE_Fourier_SINDy_Dynamics

    def __init__(self, mjx_model: mjx.Model, residual: Fourier_SINDy_Residual):
        self.knode_dynamics = KNODE_Fourier_SINDy_Dynamics(mjx_model, residual)

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
# SPARSE OPTIMISERS  (NumPy, one-shot)
# ============================================================================

def stlsq(
    Theta:     np.ndarray,
    Xdot:      np.ndarray,
    threshold: float = 0.05,
    alpha:     float = 1e-5,
    max_iter:  int   = 20,
    verbose:   bool  = True,
) -> np.ndarray:
    """Sequentially Thresholded Least Squares (Brunton et al. 2016)."""
    n_features = Theta.shape[1]
    n_outputs  = Xdot.shape[1]

    Xi = np.linalg.solve(
        Theta.T @ Theta + alpha * np.eye(n_features), Theta.T @ Xdot
    )
    if verbose:
        print(f"  STLSQ init: {int(np.sum(np.abs(Xi) > 0))}/{n_features*n_outputs} nonzero")

    for it in range(max_iter):
        Xi_prev   = Xi.copy()
        small     = np.abs(Xi) < threshold
        Xi[small] = 0.0

        for k in range(n_outputs):
            active = ~small[:, k]
            if not active.any():
                continue
            Th_a = Theta[:, active]
            Xi[active, k] = np.linalg.solve(
                Th_a.T @ Th_a + alpha * np.eye(int(active.sum())),
                Th_a.T @ Xdot[:, k],
            )

        n_changed = int(np.sum(np.abs(np.sign(Xi) - np.sign(Xi_prev)) > 0))
        if verbose:
            print(f"  STLSQ iter {it+1:3d}: "
                  f"{int(np.sum(np.abs(Xi)>0))}/{n_features*n_outputs} nonzero, "
                  f"{n_changed} changed.")
        if n_changed == 0:
            if verbose:
                print(f"  Converged after {it+1} iteration(s).")
            break
    return Xi


def sr3(
    Theta:     np.ndarray,
    Xdot:      np.ndarray,
    threshold: float = 0.05,
    nu:        float = 1.0,
    tol:       float = 1e-6,
    max_iter:  int   = 500,
    verbose:   bool  = True,
) -> np.ndarray:
    """Sparse Relaxed Regularised Regression (Zheng et al. 2019)."""
    n_features = Theta.shape[1]
    n_outputs  = Xdot.shape[1]

    TtT   = Theta.T @ Theta
    TtX   = Theta.T @ Xdot
    M_inv = np.linalg.solve(TtT + (1.0/nu)*np.eye(n_features), np.eye(n_features))

    W  = np.linalg.solve(TtT + 1e-5*np.eye(n_features), TtX)
    Xi = np.where(np.abs(W) > threshold, W, 0.0)

    if verbose:
        print(f"  SR3 init: {int(np.sum(np.abs(Xi)>0))}/{n_features*n_outputs} nonzero")

    for it in range(max_iter):
        W_prev, Xi_prev = W.copy(), Xi.copy()
        W  = M_inv @ (TtX + (1.0/nu)*Xi)
        Xi = np.where(np.abs(W) > threshold, W, 0.0)
        dW  = np.linalg.norm(W  - W_prev)  / (np.linalg.norm(W_prev)  + 1e-12)
        dXi = np.linalg.norm(Xi - Xi_prev) / (np.linalg.norm(Xi_prev) + 1e-12)
        if verbose and (it % 50 == 0 or it < 3):
            print(f"  SR3 iter {it+1:4d}: {int(np.sum(np.abs(Xi)>0))}/{n_features*n_outputs} nonzero, "
                  f"ΔW={dW:.2e}, ΔΞ={dXi:.2e}")
        if dW < tol and dXi < tol:
            if verbose:
                print(f"  SR3 converged after {it+1} iteration(s).")
            break
    return Xi


# ============================================================================
# LIBRARY MATRIX + FIT
# ============================================================================

def build_library_matrix(
    library:  Fourier_SINDy_Library,
    qpos_arr: np.ndarray,
    qvel_arr: np.ndarray,
    ctrl_arr: np.ndarray,
) -> np.ndarray:
    """Evaluate Θ at every sample. Returns (n_samples, 2*num_features)."""
    X = jnp.concatenate(
        [jnp.array(qpos_arr), jnp.array(qvel_arr), jnp.array(ctrl_arr)],
        axis=-1,
    )
    return np.array(jax.vmap(library)(X))


def fit_Fourier_SINDy(
    library:       Fourier_SINDy_Library,
    qpos_arr:      np.ndarray,
    qvel_arr:      np.ndarray,
    ctrl_arr:      np.ndarray,
    residual_qacc: np.ndarray,
    output_num:    int,
    optimizer:     str   = "stlsq",
    threshold:     float = 0.05,
    alpha:         float = 1e-5,
    nu:            float = 1.0,
    max_iter:      int   = 20,
    verbose:       bool  = True,
) -> Fourier_SINDy_Residual:
    """
    Build Θ and fit Ξ via sparse regression.

    residual_qacc is the target Ẋ (no numerical differentiation needed).
    Returns a fitted Fourier_SINDy_Residual.
    """
    n_lib = library.n_library_features
    print(f"\n{'='*60}")
    print(f"  Fourier-SINDy Fitting")
    print(f"  samples        : {qpos_arr.shape[0]}")
    print(f"  library size   : {n_lib}  ({library.num_features} RFF × 2)")
    print(f"  outputs        : {output_num}")
    print(f"  optimizer      : {optimizer.upper()}  (threshold={threshold})")
    print(f"{'='*60}")

    print("  Building Θ …")
    Theta = build_library_matrix(library, qpos_arr, qvel_arr, ctrl_arr).astype(np.float64)
    Xdot  = np.array(residual_qacc, dtype=np.float64)

    cond = np.linalg.cond(Theta)
    print(f"  Θ shape = {Theta.shape},  cond(Θ) = {cond:.3e}")
    if cond > 1e10:
        print("  WARNING: Θ ill-conditioned. "
              "Try reducing num_fourier_features, increasing alpha, "
              "or collecting more diverse data.")

    if optimizer == "stlsq":
        Xi = stlsq(Theta, Xdot, threshold=threshold, alpha=alpha,
                   max_iter=max_iter, verbose=verbose)
    elif optimizer == "sr3":
        Xi = sr3(Theta, Xdot, threshold=threshold, nu=nu,
                 max_iter=max_iter, verbose=verbose)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use 'stlsq' or 'sr3'.")

    n_nz = int(np.sum(np.abs(Xi) > 0))
    print(f"\n  Fitting complete.")
    print(f"  Active terms : {n_nz} / {n_lib * output_num} "
          f"({100.0 * n_nz / (n_lib * output_num):.1f}% dense)\n")

    return Fourier_SINDy_Residual(library, output_num, jnp.array(Xi))
