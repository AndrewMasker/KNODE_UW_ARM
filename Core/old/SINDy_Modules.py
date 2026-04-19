"""
SINDy_Modules.py
================
JAX/Equinox implementation of SINDy with Control (SINDYc) designed as a
drop-in residual-learning replacement for the MLP in KNODE_Modules.

Architecture mirrors PySINDy's design:
  Θ̇ ≈ Θ(qpos, qvel, ctrl) @ Ξ

Feature libraries
-----------------
  • PolynomialLibrary  – all monomials up to degree d (with cross-terms)
  • FourierLibrary     – sin(k·x̃), cos(k·x̃) for k = 1 … n_frequencies
  • CombinedLibrary    – concat of the two above (default)

Inputs are normalised to ≈ [-1, 1] before library evaluation, exactly
mirroring the normalisation in Residual_Network.

Optimisers (CPU/NumPy – one-shot fitting, no JIT needed)
---------------------------------------------------------
  • stlsq  – Sequentially Thresholded Least Squares (Brunton et al. 2016)
  • sr3    – Sparse Relaxed Regularised Regression  (Zheng et al. 2019)

Drop-in classes
---------------
  • SINDy_Residual   – same __call__ signature as Residual_Network
  • KNODE_SINDy      – same interface as KNODE_Dynamics; plugs into diffrax

Usage
-----
See Train_SINDy.py for the full fitting workflow.
"""

import itertools
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from mujoco import mjx

# Assumes your project has this module (same as KNODE_Modules_Old.py uses it)
import Utils.MJX_Diff as mjx_diff


# ============================================================================
# SECTION 1 — POLYNOMIAL EXPONENT HELPERS
# ============================================================================

def _make_poly_exponents(n_inputs: int,
                         degree: int,
                         include_bias: bool = True) -> np.ndarray:
    """
    Build an (n_terms, n_inputs) integer exponent matrix for all monomials
    up to `degree`.  Row i encodes one monomial: x[j]^exponents[i, j].

    Example (n_inputs=2, degree=2, include_bias=True):
        [0, 0]  →  1
        [1, 0]  →  x0
        [0, 1]  →  x1
        [2, 0]  →  x0²
        [1, 1]  →  x0·x1
        [0, 2]  →  x1²
    """
    rows = []
    start = 0 if include_bias else 1
    for d in range(start, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_inputs), d):
            row = np.zeros(n_inputs, dtype=np.int32)
            for idx in combo:
                row[idx] += 1
            rows.append(row)
    return np.array(rows, dtype=np.int32)   # (n_terms, n_inputs)


def _make_poly_names(n_inputs: int,
                     degree: int,
                     input_names: List[str],
                     include_bias: bool = True) -> List[str]:
    names = []
    start = 0 if include_bias else 1
    for d in range(start, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_inputs), d):
            if len(combo) == 0:
                names.append("1")
                continue
            counts: dict = {}
            for idx in combo:
                counts[idx] = counts.get(idx, 0) + 1
            term = "".join(
                f"{input_names[i]}^{p}" if p > 1 else input_names[i]
                for i, p in sorted(counts.items())
            )
            names.append(term)
    return names


# ============================================================================
# NEW HELPER FOR SUM COMBINATIONS
# ============================================================================

def _make_sum_indices(n_inputs: int, degree: int) -> np.ndarray:
    """
    Returns an (n_combinations, n_inputs) binary matrix where each row 
    indicates which inputs to sum. 
    Example: [1, 1, 0] means (x0 + x1).
    """
    rows = []
    # We start from degree 2 because degree 1 is already covered by standard Fourier
    for d in range(2, degree + 1):
        for combo in itertools.combinations(range(n_inputs), d):
            row = np.zeros(n_inputs, dtype=np.int32)
            row[list(combo)] = 1
            rows.append(row)
    return np.array(rows, dtype=np.int32) if rows else np.empty((0, n_inputs), dtype=np.int32)

def _make_sum_names(n_inputs: int, degree: int, input_names: List[str], include_sin: bool, include_cos: bool) -> List[str]:
    names = []
    for d in range(2, degree + 1):
        for combo in itertools.combinations(range(n_inputs), d):
            inner_sum = "+".join([input_names[i] for i in combo])
            if include_sin: names.append(f"sin({inner_sum})")
            if include_cos: names.append(f"cos({inner_sum})")
    return names


# ============================================================================
# SECTION 2 — FEATURE LIBRARY (JAX/Equinox, fully JIT-compatible)
# ============================================================================

class SINDy_Library(eqx.Module):
    """
    Evaluates the combined SINDy feature library Θ(x) at a single normalised
    input vector x = concat(qpos, qvel, ctrl).

    Fields
    ------
    poly_exponents    : (n_poly_terms, n_inputs) JAX int array of monomial
                        powers.  Fixed at construction; not differentiable.
    input_norm_add    : additive shift so that raw inputs → zero-centred.
    input_norm_mult   : multiplicative scale so that shifted inputs → [-1, 1].
    n_inputs          : total input dimension (nq + nv + nu).
    n_poly_terms      : number of polynomial library columns.
    n_fourier_terms   : number of Fourier library columns.
    n_library_features: total library width.
    n_frequencies     : number of Fourier harmonics (0 = no Fourier terms).
    include_sin       : include sin(k·x̃) terms.
    include_cos       : include cos(k·x̃) terms.
    feature_names     : human-readable label for each library column.
    """

    # ---------- JAX array fields (non-differentiable leaves) ----------
    poly_exponents:     jax.Array   # (n_poly_terms, n_inputs), int32
#    input_norm_add:     jax.Array   # (n_inputs,)
#    input_norm_mult:    jax.Array   # (n_inputs,)

    # ---------- static / structural fields ----------------------------
    n_inputs:           int
    n_poly_terms:       int
    n_fourier_terms:    int
    n_library_features: int
    n_frequencies:      int
    include_sin:        bool
    include_cos:        bool
    feature_names:      list        # List[str]
    
    sum_indices:        jax.Array   # (n_sum_combinations, n_inputs)
    n_sum_terms:        int
    include_sum_fourier: bool

    def __init__(
        self,
        nq:             int,
        nv:             int,
        nu:             int,
        qpos_ranges:    jax.Array,   # (nq, 2)  – mjx_model.jnt_range
        qvel_max:       jax.Array,   # (nv,)    – from dataset attrs
        ctrl_ranges:    jax.Array,   # (nu, 2)  – mjx_model.actuator_ctrlrange
        poly_degree:    int   = 2,
        n_frequencies:  int   = 1,
        include_bias:   bool  = True,
        include_sin:    bool  = True,
        include_cos:    bool  = True,
        include_sum_fourier: bool = True,
        sum_fourier_degree:  int  = 2,
    ):
        n_inputs = nq + nv + nu
        self.n_inputs      = n_inputs
        self.n_frequencies = n_frequencies
        self.include_sin   = include_sin
        self.include_cos   = include_cos
        self.include_sum_fourier = include_sum_fourier

        # ---- Polynomial exponents (stored as JAX array) ----
        exp_np = _make_poly_exponents(n_inputs, poly_degree, include_bias)
        self.poly_exponents = jnp.array(exp_np, dtype=jnp.int32)
        self.n_poly_terms   = exp_np.shape[0]

        # ---- Fourier term count ----
        fourier_per_freq = (int(include_sin) + int(include_cos)) * n_inputs
        self.n_fourier_terms    = fourier_per_freq * n_frequencies
        
        
        # ---- NEW: Sum-Fourier setup ----
        if include_sum_fourier:
            sum_idx_np = _make_sum_indices(n_inputs, sum_fourier_degree)
            self.sum_indices = jnp.array(sum_idx_np, dtype=jnp.int32)
            # Each combination produces a sin AND/OR a cos
            self.n_sum_terms = sum_idx_np.shape[0] * (int(include_sin) + int(include_cos))
        else:
            self.sum_indices = jnp.empty((0, n_inputs), dtype=jnp.int32)
            self.n_sum_terms = 0
            
        self.n_library_features = self.n_poly_terms + self.n_fourier_terms + self.n_sum_terms

        # ---- Normalisation (mirrors Residual_Network exactly) ----
        # Clamp range widths to avoid division by zero for unbounded joints.
#        qpos_width = jnp.maximum(qpos_ranges[:, 1] - qpos_ranges[:, 0], 1e-6)
#        ctrl_width = jnp.maximum(ctrl_ranges[:, 1] - ctrl_ranges[:, 0], 1e-6)
#
#        qpos_add  = -qpos_ranges[:, 0]
#        qvel_add  =  qvel_max                       # shift so 0 maps to 0
#        ctrl_add  = -ctrl_ranges[:, 0]
#
#        qpos_mult = 2.0 / qpos_width
#        qvel_mult = 1.0 / jnp.maximum(qvel_max, 1e-6)
#        ctrl_mult = 2.0 / ctrl_width
#
#        self.input_norm_add  = jnp.concatenate([qpos_add,  qvel_add,  ctrl_add])
#        self.input_norm_mult = jnp.concatenate([qpos_mult, qvel_mult, ctrl_mult])

        # ---- Human-readable feature names ----
        input_names = (
            [f"q{i}"  for i in range(nq)] +
            [f"dq{i}" for i in range(nv)] +
            [f"u{i}"  for i in range(nu)]
        )
        poly_names    = _make_poly_names(n_inputs, poly_degree, input_names, include_bias)
        fourier_names = []
        for k in range(1, n_frequencies + 1):
            for name in input_names:
                if include_sin:
                    fourier_names.append(f"sin({k}*{name})")
                if include_cos:
                    fourier_names.append(f"cos({k}*{name})")
                    
        sum_names = []
        if include_sum_fourier:
            sum_names = _make_sum_names(n_inputs, sum_fourier_degree, input_names, include_sin, include_cos)
            
        self.feature_names = poly_names + fourier_names + sum_names

    # ------------------------------------------------------------------
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Evaluate Θ(x) for a single raw input vector x of shape (n_inputs,).
        Returns feature vector of shape (n_library_features,).

        Normalisation:  x̃ = input_norm_mult * (x + input_norm_add) − 1
                            ≈ O(1) and centred near 0.
        """
        # Normalise to ≈ [-1, 1]
#        x_norm = self.input_norm_mult * (x + self.input_norm_add) - 1.0
        x_norm = x

        # ---- Polynomial features ----
        # exps: (n_poly_terms, n_inputs); each row is a monomial power vector.
        # For row e:  feature = Π_j  x_norm[j]^e[j]
        # Note: x^0 = 1 by floating-point convention (covers bias term).
        exps_f = self.poly_exponents.astype(jnp.float32)   # (n_poly, n_inputs)
        poly_features = jax.vmap(
            lambda e: jnp.prod(x_norm ** e)
        )(exps_f)                                           # (n_poly,)

        if self.n_fourier_terms == 0:
            return poly_features

        # ---- Fourier features ----
        fourier_parts = []
        for k in range(1, self.n_frequencies + 1):
            if self.include_sin:
                fourier_parts.append(jnp.sin(k * x_norm))  # (n_inputs,)
            if self.include_cos:
                fourier_parts.append(jnp.cos(k * x_norm))  # (n_inputs,)
        fourier_features = jnp.concatenate(fourier_parts)   # (n_fourier,)
        
        
        
        # ---- NEW: Sum-Fourier features ----
        sum_features = jnp.array([])
        if self.include_sum_fourier and self.n_sum_terms > 0:
            # Vectorized sum of inputs based on the pre-built index mask
            # sums: (n_combinations,)
            sums = jnp.dot(self.sum_indices.astype(jnp.float32), x_norm)
            
            sum_parts = []
            if self.include_sin: sum_parts.append(jnp.sin(sums))
            if self.include_cos: sum_parts.append(jnp.cos(sums))
            
            # Interleave if both sin and cos are present to match name order
            if self.include_sin and self.include_cos:
                sum_features = jnp.stack([jnp.sin(sums), jnp.cos(sums)], axis=1).ravel()
            else:
                sum_features = jnp.concatenate(sum_parts)

        return jnp.concatenate([poly_features, fourier_features , sum_features])


# ============================================================================
# SECTION 3 — SINDY RESIDUAL MODEL (drop-in for Residual_Network)
# ============================================================================

class SINDy_Residual(eqx.Module):
    """
    SINDy residual model with the same call signature as Residual_Network.

    Given x = concat(qpos, qvel, ctrl), computes:

        residual_qacc(x) = Θ(x) @ Ξ

    where Ξ (coefficients) has shape (n_library_features, output_num).
    After fitting, most entries of Ξ are exactly zero.

    This class is an eqx.Module so it can be serialised with
    eqx.tree_serialise_leaves / eqx.tree_deserialise_leaves.
    Note: coefficients ARE stored as a differentiable JAX array so the model
    can optionally be fine-tuned with gradient descent after sparse fitting.
    """

    library:      SINDy_Library
    coefficients: jax.Array     # (n_library_features, output_num)
    output_num:   int

    def __init__(
        self,
        library:      SINDy_Library,
        output_num:   int,
        coefficients: Optional[jax.Array] = None,
    ):
        self.library    = library
        self.output_num = output_num
        if coefficients is None:
            self.coefficients = jnp.zeros(
                (library.n_library_features, output_num)
            )
        else:
            self.coefficients = jnp.asarray(coefficients)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x : concatenated (qpos, qvel, ctrl), shape (n_inputs,).
        Returns residual qacc, shape (output_num,).

        Identical call signature to Residual_Network.__call__.
        """
        theta = self.library(x)          # (n_library_features,)
        return theta @ self.coefficients  # (output_num,)

    # ------------------------------------------------------------------
    def print_equations(self, output_names: Optional[List[str]] = None) -> None:
        """Print the discovered sparse dynamical equations."""
        if output_names is None:
            output_names = [f"residual_qacc[{i}]" for i in range(self.output_num)]
        xi   = np.array(self.coefficients)    # (n_lib, n_outputs)
        names = self.library.feature_names

        print("\n========== SINDy Discovered Equations ==========")
        for k, out_name in enumerate(output_names):
            active = [(names[j], xi[j, k]) for j in range(len(names))
                      if abs(xi[j, k]) > 1e-10]
            if active:
                eq = " + ".join(
                    f"({c:+.6f})*{n}" for n, c in active
                )
            else:
                eq = "0  (fully zeroed – try lowering threshold)"
            print(f"  {out_name}' = {eq}")
        print("=================================================\n")

    def sparsity_report(self) -> dict:
        """Return sparsity statistics as a dict."""
        xi = np.array(self.coefficients)
        n_total   = xi.size
        n_nonzero = int(np.sum(np.abs(xi) > 0))
        return {
            "n_total":    n_total,
            "n_nonzero":  n_nonzero,
            "n_zero":     n_total - n_nonzero,
            "sparsity_%": 100.0 * (1 - n_nonzero / n_total),
        }


# ============================================================================
# SECTION 4 — KNODE_SINDy (drop-in for the full KNODE_Dynamics + KNODE stack)
# ============================================================================

class KNODE_SINDy_Dynamics(eqx.Module):
    """
    Mirrors KNODE_Dynamics but uses SINDy_Residual instead of an MLP.
    Compatible with diffrax ODETerm.
    """
    sindy_residual:    SINDy_Residual
#    res_mixing_matrix: jax.Array
    mjx_model:         mjx.Model

    def __init__(
        self,
        mjx_model:      mjx.Model,
        sindy_residual: SINDy_Residual,
        mixing_scale:   float = 0.0,   # 0 = identity init (no extra mixing)
    ):
        self.mjx_model      = mjx_model
        self.sindy_residual = sindy_residual
        output_num          = mjx_model.nq
        # Identity + small random perturbation (same as original KNODE_Dynamics)
#        self.res_mixing_matrix = (
#            jnp.eye(output_num) +
#            mixing_scale * jr.uniform(key, (output_num, output_num),
#                                      minval=-1.0, maxval=1.0)
#        )

    def __call__(self, t, state, args):
        qpos = state["qpos"]
        qvel = state["qvel"]
        mjx_data, ctrl, with_learned = args

        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        res_qacc = (
            with_learned
            * self.sindy_residual(jnp.concatenate((qpos, qvel, ctrl)))
        )
        mjx_data = mjx_diff.forward(self.mjx_model, mjx_data)
        final_qacc = mjx_data.qacc + res_qacc# @ self.res_mixing_matrix

        return {"qpos": qvel, "qvel": final_qacc}


class KNODE_SINDy(eqx.Module):
    """
    Full KNODE model with SINDy residual.  Same __call__ interface as KNODE.
    Pass to diffrax via diffrax.ODETerm(model.knode_dynamics).
    """
    knode_dynamics: KNODE_SINDy_Dynamics

    def __init__(self, mjx_model, sindy_residual: SINDy_Residual):
        self.knode_dynamics = KNODE_SINDy_Dynamics(
            mjx_model, sindy_residual
        )

    def __call__(self, ts, mjx_data, init_state, ctrl,
                 with_learned=jnp.array(1.0)):
        import diffrax
        mjx_data = mjx_data.replace(
            qpos=init_state["qpos"],
            qvel=init_state["qvel"],
            ctrl=ctrl,
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


# ============================================================================
# SECTION 5 — OPTIMISERS  (NumPy – one-shot fitting, not JIT-compiled)
# ============================================================================

def stlsq(
    Theta:     np.ndarray,   # (n_samples, n_features)
    Xdot:      np.ndarray,   # (n_samples, n_outputs)
    threshold: float = 0.05,
    alpha:     float = 1e-5,  # L2 ridge for conditioning
    max_iter:  int   = 20,
    verbose:   bool  = True,
) -> np.ndarray:
    """
    Sequentially Thresholded Least Squares (STLSQ) — Brunton et al. 2016.

    Iteratively:
      1. Zero coefficients whose absolute value < threshold.
      2. Re-fit the surviving (active) coefficients via ridge regression.

    Parameters
    ----------
    Theta     : (m, p) library matrix.
    Xdot      : (m, n) derivative / target matrix.
    threshold : sparsity threshold λ.
    alpha     : ridge regularisation for numerical stability.
    max_iter  : maximum thresholding iterations.
    verbose   : print per-iteration sparsity.

    Returns
    -------
    Xi : (p, n) sparse coefficient matrix.
    """
    n_samples, n_features = Theta.shape
    n_outputs             = Xdot.shape[1]

    # Initial ridge solve
    TtT = Theta.T @ Theta + alpha * np.eye(n_features)
    TtX = Theta.T @ Xdot
    Xi  = np.linalg.solve(TtT, TtX)               # (n_features, n_outputs)

    if verbose:
        init_nz = int(np.sum(np.abs(Xi) > 0))
        print(f"  STLSQ  init: {init_nz}/{n_features * n_outputs} nonzero coefficients")

    for it in range(max_iter):
        Xi_prev = Xi.copy()

        # Threshold
        small     = np.abs(Xi) < threshold
        Xi[small] = 0.0

        # Per-output re-fit on active features
        for k in range(n_outputs):
            active = ~small[:, k]
            if not active.any():
                continue
            Th_a   = Theta[:, active]
            TtT_a  = Th_a.T @ Th_a + alpha * np.eye(int(active.sum()))
            TtX_a  = Th_a.T @ Xdot[:, k]
            Xi[active, k] = np.linalg.solve(TtT_a, TtX_a)

        n_nonzero = int(np.sum(np.abs(Xi) > 0))
        n_changed = int(np.sum(np.abs(np.sign(Xi) - np.sign(Xi_prev)) > 0))

        if verbose:
            print(f"  STLSQ  iter {it + 1:3d}: "
                  f"{n_nonzero}/{n_features * n_outputs} nonzero, "
                  f"{n_changed} coefficient(s) changed.")

        if n_changed == 0:
            if verbose:
                print(f"  Converged after {it + 1} iteration(s).")
            break

    return Xi


def sr3(
    Theta:     np.ndarray,   # (n_samples, n_features)
    Xdot:      np.ndarray,   # (n_samples, n_outputs)
    threshold: float = 0.05,
    nu:        float = 1.0,   # relaxation: smaller ν → more aggressive sparsity
    tol:       float = 1e-6,
    max_iter:  int   = 500,
    verbose:   bool  = True,
) -> np.ndarray:
    """
    Sparse Relaxed Regularised Regression (SR3) — Zheng et al. 2019.

    Solves the relaxed bi-level problem:
        min_{W, Ξ}  ‖Θ W − Ẋ‖² + (1/2ν)‖W − Ξ‖² + R(Ξ)
    where R is L0 hard-thresholding.

    W update: closed-form least squares.
    Ξ update: hard threshold of W at `threshold`.

    Parameters
    ----------
    Theta     : (m, p) library matrix.
    Xdot      : (m, n) target matrix.
    threshold : hard sparsity threshold.
    nu        : relaxation parameter (PySINDy default ≈ 0.05–1.0).
    tol       : relative convergence tolerance on W and Ξ.
    max_iter  : maximum alternating iterations.
    verbose   : print progress every 50 iterations.

    Returns
    -------
    Xi : (p, n) sparse coefficient matrix.
    """
    n_samples, n_features = Theta.shape
    n_outputs             = Xdot.shape[1]

    TtT   = Theta.T @ Theta                               # (p, p)
    TtX   = Theta.T @ Xdot                               # (p, n)
    M     = TtT + (1.0 / nu) * np.eye(n_features)
    M_inv = np.linalg.solve(M, np.eye(n_features))       # pre-compute

    # Initialise from ridge solve
    W  = np.linalg.solve(TtT + 1e-5 * np.eye(n_features), TtX)
    Xi = np.where(np.abs(W) > threshold, W, 0.0)

    if verbose:
        print(f"  SR3    init: {int(np.sum(np.abs(Xi) > 0))}/{n_features * n_outputs} nonzero")

    for it in range(max_iter):
        W_prev  = W.copy()
        Xi_prev = Xi.copy()

        # W update
        W  = M_inv @ (TtX + (1.0 / nu) * Xi)
        # Ξ update (hard threshold)
        Xi = np.where(np.abs(W) > threshold, W, 0.0)

        # Convergence
        dW  = np.linalg.norm(W  - W_prev)  / (np.linalg.norm(W_prev)  + 1e-12)
        dXi = np.linalg.norm(Xi - Xi_prev) / (np.linalg.norm(Xi_prev) + 1e-12)

        if verbose and (it % 50 == 0 or it < 3):
            nz = int(np.sum(np.abs(Xi) > 0))
            print(f"  SR3    iter {it + 1:4d}: {nz}/{n_features * n_outputs} nonzero, "
                  f"ΔW={dW:.2e}, ΔΞ={dXi:.2e}")

        if dW < tol and dXi < tol:
            if verbose:
                print(f"  SR3 converged after {it + 1} iteration(s).")
            break

    return Xi


# ============================================================================
# SECTION 6 — BUILD Θ MATRIX AND FIT
# ============================================================================

def build_library_matrix(
    library:  SINDy_Library,
    qpos_arr: np.ndarray,    # (n_samples, nq)
    qvel_arr: np.ndarray,    # (n_samples, nv)
    ctrl_arr: np.ndarray,    # (n_samples, nu)
) -> np.ndarray:
    """
    Evaluate the feature library at every data point.

    Vectorised with jax.vmap for efficiency.

    Returns
    -------
    Theta : (n_samples, n_library_features) NumPy array.
    """
    X_jax   = jnp.concatenate(
        [jnp.array(qpos_arr), jnp.array(qvel_arr), jnp.array(ctrl_arr)],
        axis=-1
    )                                       # (n_samples, n_inputs)
    Theta   = jax.vmap(library)(X_jax)      # (n_samples, n_lib)
    return np.array(Theta)


def fit_SINDy(
    library:       SINDy_Library,
    qpos_arr:      np.ndarray,         # (n_samples, nq)
    qvel_arr:      np.ndarray,         # (n_samples, nv)
    ctrl_arr:      np.ndarray,         # (n_samples, nu)
    residual_qacc: np.ndarray,         # (n_samples, nq)
    output_num:    int,
    optimizer:     str   = "stlsq",
    threshold:     float = 0.05,
    alpha:         float = 1e-5,       # STLSQ ridge
    nu:            float = 1.0,        # SR3 relaxation
    max_iter:      int   = 20,
    verbose:       bool  = True,
) -> SINDy_Residual:
    """
    Build the library matrix Θ and run sparse regression to identify Ξ.

    Since the user already has a residual_qacc dataset, no numerical
    differentiation is needed — residual_qacc IS the target Ẋ.

    Parameters
    ----------
    library       : a constructed SINDy_Library instance.
    qpos_arr      : (n_samples, nq)  – joint positions.
    qvel_arr      : (n_samples, nv)  – joint velocities.
    ctrl_arr      : (n_samples, nu)  – control inputs.
    residual_qacc : (n_samples, nq)  – residual accelerations (targets).
    output_num    : nq.
    optimizer     : "stlsq" or "sr3".
    threshold     : sparsity threshold λ.
    alpha         : ridge strength for STLSQ.
    nu            : relaxation for SR3.
    max_iter      : max optimiser iterations.
    verbose       : print progress.

    Returns
    -------
    SINDy_Residual eqx.Module with fitted (sparse) coefficients.
    """
    n_samples = qpos_arr.shape[0]
    n_lib     = library.n_library_features

    print(f"\n{'='*60}")
    print(f"  SINDy Fitting")
    print(f"  samples       : {n_samples}")
    print(f"  library size  : {n_lib} features")
    print(f"  outputs       : {output_num}")
    print(f"  optimizer     : {optimizer.upper()}  (threshold={threshold})")
    print(f"{'='*60}")

    print("  Building Θ matrix …")
    Theta = build_library_matrix(library, qpos_arr, qvel_arr, ctrl_arr)
    Xdot  = np.array(residual_qacc, dtype=np.float64)

    # Condition number diagnostic
    cond = np.linalg.cond(Theta)
    print(f"  Θ shape = {Theta.shape},  cond(Θ) = {cond:.3e}")
    if cond > 1e10:
        print("  WARNING: Θ is ill-conditioned. "
              "Consider reducing poly_degree / n_frequencies, "
              "increasing alpha, or collecting more diverse data.")

    Theta = Theta.astype(np.float64)

    if optimizer == "stlsq":
        Xi = stlsq(Theta, Xdot, threshold=threshold,
                   alpha=alpha, max_iter=max_iter, verbose=verbose)
    elif optimizer == "sr3":
        Xi = sr3(Theta, Xdot, threshold=threshold,
                 nu=nu, max_iter=max_iter, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Choose 'stlsq' or 'sr3'."
        )

    n_nz = int(np.sum(np.abs(Xi) > 0))
    print(f"\n  Fitting complete.")
    print(f"  Active terms  : {n_nz} / {n_lib * output_num} "
          f"({100.0 * n_nz / (n_lib * output_num):.1f}% dense)\n")

    return SINDy_Residual(library, output_num, jnp.array(Xi))
    
    
    
    
    
