"""
Primal-space banded ridge regression via LSMR with implicit differentiation.

Problem
-------
We have K feature subspaces X_1,...,X_K (each X_k of shape [T, p_k]) and a
response Y of shape [T, n_targets].  We want to solve, for each target:

    min_w  (1/2) ||Xw - y||^2  +  (1/2) sum_k exp(delta_k) ||w_k||^2

where X = [X_1 | ... | X_K] and w = [w_1; ...; w_K].

The regularised normal equations are

    (X^T X + Lambda) w = X^T y          (*)

with Lambda = diag(exp(delta_1)*I_{p1}, ..., exp(delta_K)*I_{pK}).

We never form X^T X.  Instead we reformulate (*) as the least-squares problem

    A w = b,   A = [X ; Lambda^{1/2}],  b = [y; 0]

and solve it with LSMR through a matrix-free LinearOperator.

Hyperparameter optimisation
---------------------------
We minimise a hold-out (or cross-validation) loss L(delta) using implicit
differentiation exactly as in Himalaya's _hyper_gradient.py:

    dL/d(delta) = -nabla_delta(w*) . nabla_w L

where nabla_delta(w*) is obtained via the implicit function theorem applied to
the optimality condition (*):

    (X^T X + Lambda) dw/d(delta_k) = -dLambda/d(delta_k) w
                                   = -exp(delta_k) E_k w_k   (block-diagonal)

So the adjoint trick gives

    dL/d(delta_k) = -w_k^T (exp(delta_k) * mu_k)
                  = -exp(delta_k) * w_k^T mu_k

where mu = (X^T X + Lambda)^{-1} nabla_w L, solved by the same LSMR code.

Outer optimiser: Adam or L-BFGS on log-lambda (delta).
"""

from __future__ import annotations

import math
import torch
from typing import List, Optional, Tuple, Callable
import tqdm

# ---------------------------------------------------------------------------
# Matrix-free linear operator
# ---------------------------------------------------------------------------

class BandedRidgeOperator:
    """Matrix-free operator for the augmented system  A = [X ; Lambda^{1/2}].

    Parameters
    ----------
    Xs : list of tensors, each (T, p_k)
        Feature subspaces.  All must be on the same device / dtype.
    log_lambdas : tensor, shape (K,) or (K,n_targets)
        Log regularisation weights, one per subspace or one per subspace x target.
    """

    def __init__(self, Xs: List[torch.Tensor], log_lambdas: torch.Tensor):
        self.Xs = Xs
        self.log_lambdas = log_lambdas  # shape (K,) or (K, n_targets)
        self.T = Xs[0].shape[0]
        self.splits = [X.shape[1] for X in Xs]  # p_k for each subspace
        self.P = sum(self.splits)   # total number of primal features
        # Shape of the augmented system: (T + P) x P
        self.shape = (self.T + self.P, self.P)
        # Pre-compute per-subspace sqrt(lambda).
        # per_target=True: sqrt_lam[k] has shape (n_targets,), broadcast over feature dim.
        # per_target=False: sqrt_lam[k] is a plain float scalar.
        self.per_target = log_lambdas.ndim == 2  # (K, n_targets) case
        if self.per_target:
            self.sqrt_lam = [torch.exp(0.5 * log_lambdas[k]) for k in range(len(Xs))]
        else:
            self.sqrt_lam = [math.exp(0.5 * float(lk)) for lk in log_lambdas]

    # ------------------------------------------------------------------
    # matvec:  A @ w  (shape P -> shape T+P)
    # ------------------------------------------------------------------
    def matvec(self, w: torch.Tensor) -> torch.Tensor:
        """Compute A @ w where A = [X ; Lambda^{1/2}].

        Parameters
        ----------
        w : tensor of shape (P,) or (P, n_rhs)

        Returns
        -------
        out : tensor of shape (T+P,) or (T+P, n_rhs)
        """
        single = w.ndim == 1
        if single:
            w = w.unsqueeze(-1)  # (P, 1)

        # --- top block: X w ---
        ws = w.split(self.splits, dim=0)       # list of (p_k, n_rhs)
        xw = sum(X @ wk for X, wk in zip(self.Xs, ws))   # (T, n_rhs)

        # --- bottom block: Lambda^{1/2} w ---
        if self.per_target:
            # sqrt_lam[k]: (n_targets,); wk: (p_k, n_targets)
            sqrt_lam_w = torch.cat(
                [slk.unsqueeze(0) * wk for slk, wk in zip(self.sqrt_lam, ws)],
                dim=0,
            )  # (P, n_rhs)
        else:
            sqrt_lam_w = torch.cat(
                [slk * wk for slk, wk in zip(self.sqrt_lam, ws)],
                dim=0,
            )  # (P, n_rhs)

        out = torch.cat([xw, sqrt_lam_w], dim=0)  # (T+P, n_rhs)
        return out.squeeze(-1) if single else out

    # ------------------------------------------------------------------
    # rmatvec:  A^T @ v  (shape T+P -> shape P)
    # ------------------------------------------------------------------
    def rmatvec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute A^T @ v where A = [X ; Lambda^{1/2}].

        Parameters
        ----------
        v : tensor of shape (T+P,) or (T+P, n_rhs)

        Returns
        -------
        out : tensor of shape (P,) or (P, n_rhs)
        """
        single = v.ndim == 1
        if single:
            v = v.unsqueeze(-1)  # (T+P, 1)

        v_top = v[: self.T]           # (T, n_rhs)
        v_bot = v[self.T :]           # (P, n_rhs)
        v_bots = v_bot.split(self.splits, dim=0)  # list of (p_k, n_rhs)

        parts = []
        for X, vk, slk in zip(self.Xs, v_bots, self.sqrt_lam):
            xTv = X.T @ v_top                                           # (p_k, n_rhs)
            reg  = slk.unsqueeze(0) * vk if self.per_target else slk * vk  # (p_k, n_rhs)
            parts.append(xTv + reg)

        out = torch.cat(parts, dim=0)  # (P, n_rhs)
        return out.squeeze(-1) if single else out


# ---------------------------------------------------------------------------
# LSMR solver
# ---------------------------------------------------------------------------

def lsmr_solve(
    matvec: Callable,
    rmatvec: Callable,
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    max_iter: int = 300,
    atol: float = 1e-6,
    btol: float = 1e-6,
    conlim: float = 1e8,
    damp: float = 0.0,
) -> Tuple[torch.Tensor, int, float]:
    """Solve  min_x ||A x - b||  via LSMR (Fong & Saunders 2011).

    Supports batched right-hand sides: if b has shape (m, n_rhs) each column
    is solved independently through the same sequence of matvec/rmatvec calls
    (one pass, same Lanczos basis for all targets at once).

    Parameters
    ----------
    matvec  : callable, A @ x
    rmatvec : callable, A^T @ v
    b       : tensor of shape (m,) or (m, n_rhs)
    x0      : optional warm start, same shape as x
    max_iter: int
    atol, btol, conlim : LSMR stopping tolerances
    damp    : Tikhonov damping (usually 0 here; regularisation is baked in A)

    Returns
    -------
    x     : solution tensor, same shape as x0 / (n,) or (n, n_rhs)
    iters : number of iterations performed
    normr : final residual norm
    """
    batched = b.ndim == 2
    if not batched:
        b = b.unsqueeze(-1)  # (m, 1)

    m, n_rhs = b.shape
    # number of columns n is inferred from rmatvec
    n = rmatvec(b[:1]).shape[0]  # quick probe

    dtype  = b.dtype
    device = b.device

    # --- initialise ---
    if x0 is not None:
        r = b - matvec(x0)
    else:
        r = b.clone()
        x0 = torch.zeros(n, n_rhs, dtype=dtype, device=device)

    x = x0.clone()

    beta_vec = r                       # (m, n_rhs)
    beta     = _col_norm(beta_vec)     # (1, n_rhs)
    u        = beta_vec / (beta + 1e-30)

    alpha_vec = rmatvec(u)             # (n, n_rhs)
    alpha     = _col_norm(alpha_vec)   # (1, n_rhs)
    v         = alpha_vec / (alpha + 1e-30)

    # Scalars broadcast over n_rhs
    x, iters, normr_val = _lsmr_core(
        matvec, rmatvec, b, x0, max_iter, atol, btol, conlim, damp, dtype, device, n, n_rhs
    )

    if not batched:
        x = x.squeeze(-1)
        normr_val_scalar = float(normr_val.squeeze())
    else:
        normr_val_scalar = float(normr_val.mean())

    return x, iters, normr_val_scalar


def _col_norm(v: torch.Tensor) -> torch.Tensor:
    """Per-column L2 norm: (m, n_rhs) -> (1, n_rhs)."""
    return v.norm(dim=0, keepdim=True)


def _sym_ortho(a: torch.Tensor, b: torch.Tensor):
    """Stable symmetric Givens rotation: returns (c, s, r) such that
    [c  s] [a]   [r]
    [s -c] [b] = [0]
    and r = hypot(a, b).  Handles b == 0 and a == 0 cleanly.
    """
    r = torch.sqrt(a ** 2 + b ** 2)
    c = torch.where(r > 0, a / r, torch.ones_like(a))
    s = torch.where(r > 0, b / r, torch.zeros_like(b))
    return c, s, r


def _lsmr_core(
    matvec, rmatvec, b, x0, max_iter, atol, btol, conlim, damp,
    dtype, device, n, n_rhs
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """Vectorised LSMR — exact port of scipy's implementation.

    Solves  min_x ||A x - b|| (or the damped version) for all n_rhs columns
    of b simultaneously, sharing the same Lanczos sequence.  Shape convention:
    scalars -> (n_rhs,);  vectors -> (dim, n_rhs).
    """
    if x0 is not None:
        r = b - matvec(x0)
        x = x0.clone()
    else:
        r = b.clone()
        x = torch.zeros(n, n_rhs, dtype=dtype, device=device)

    # ---- initialise bidiagonalisation ----
    beta = _col_norm(r).squeeze(0)                          # (n_rhs,)
    u    = r / (beta.unsqueeze(0) + 1e-30)                  # (m, n_rhs)

    v0   = rmatvec(u)                                       # (n, n_rhs)
    alpha = _col_norm(v0).squeeze(0)                        # (n_rhs,)
    v    = v0 / (alpha.unsqueeze(0) + 1e-30)                # (n, n_rhs)

    # ---- scalar state (all shape (n_rhs,)) ----
    alphabar = alpha.clone()
    zetabar  = alpha * beta
    rho      = torch.ones_like(alpha)
    rhobar   = torch.ones_like(alpha)
    cbar     = torch.ones_like(alpha)
    sbar     = torch.zeros_like(alpha)

    h    = v.clone()                                        # (n, n_rhs)
    hbar = torch.zeros_like(x)                             # (n, n_rhs)

    # norm-tracking state (scalars for the || ||r|| || estimate)
    betadd     = beta.clone()
    betad      = torch.zeros_like(beta)
    rhodold    = torch.ones_like(beta)
    tautildeold = torch.zeros_like(beta)
    thetatilde  = torch.zeros_like(beta)
    zeta        = torch.zeros_like(beta)
    d           = torch.zeros_like(beta)

    normA2  = alpha ** 2
    maxrbar = torch.zeros_like(alpha)
    minrbar = torch.full_like(alpha, 1e30)
    normb   = beta.clone()
    normr   = beta.clone()
    normar  = (alpha * beta).clone()
    damp_t  = torch.full_like(alpha, damp)

    ctol = 0.0 if conlim == 0.0 else 1.0 / conlim

    iters = 0
    for iters in (tqdm.tqdm(range(1, max_iter + 1),desc="inner LSMR") if (max_iter > 10) else range(1, max_iter + 1)):
        # ---- bidiagonalisation ----
        u    = matvec(v) - alpha.unsqueeze(0) * u
        beta = _col_norm(u).squeeze(0)
        u    = u / (beta.unsqueeze(0) + 1e-30)

        v    = rmatvec(u) - beta.unsqueeze(0) * v
        alpha = _col_norm(v).squeeze(0)
        v    = v / (alpha.unsqueeze(0) + 1e-30)

        # ---- Qhat rotation (handles damp) ----
        chat, shat, alphahat = _sym_ortho(alphabar, damp_t)

        # ---- Q rotation ----
        rhoold   = rho.clone()
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew  = s * alpha
        alphabar  = c * alpha

        # ---- Qbar rotation ----
        rhobarold = rhobar.clone()
        zetaold   = zeta.clone()
        thetabar  = sbar * rho
        rhotemp   = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
        zeta      = cbar * zetabar
        zetabar   = -sbar * zetabar

        # ---- update h, hbar, x ----
        hbar = h  - (thetabar * rho / (rhoold * rhobarold + 1e-30)).unsqueeze(0) * hbar
        x    = x  + (zeta / (rho * rhobar + 1e-30)).unsqueeze(0) * hbar
        h    = v  - (thetanew / (rho + 1e-30)).unsqueeze(0) * h

        # ---- norm estimation ----
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat   = c * betaacute
        betadd    = -s * betaacute

        thetatildeold        = thetatilde.clone()
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde  = stildeold * rhobar
        rhodold     = ctildeold * rhobar
        betad       = -stildeold * betad + ctildeold * betahat

        tautildeold = (zetaold - thetatildeold * tautildeold) / (rhotildeold + 1e-30)
        taud        = (zeta    - thetatilde    * tautildeold) / (rhodold      + 1e-30)
        d           = d + betacheck ** 2
        normr       = torch.sqrt(d + (betad - taud) ** 2 + betadd ** 2)

        normA2  = normA2 + beta ** 2
        normA   = torch.sqrt(normA2)
        normA2  = normA2 + alpha ** 2

        maxrbar = torch.maximum(maxrbar, rhobarold)
        if iters > 1:
            minrbar = torch.minimum(minrbar, rhobarold)
        condA   = torch.maximum(maxrbar, rhotemp) / (torch.minimum(minrbar, rhotemp) + 1e-30)

        normar = zetabar.abs()
        normx  = x.norm(dim=0)

        # ---- stopping criteria ----
        test1 = normr / (normb + 1e-30)
        test2 = normar / (normA * normr + 1e-30)
        test3 = 1.0 / (condA + 1e-30)
        t1    = test1 / (1 + normA * normx / (normb + 1e-30))
        rtol  = btol + atol * normA * normx / (normb + 1e-30)

        if (t1       <= rtol ).all(): break  # noqa: E701
        if (test2    <= atol ).all(): break  # noqa: E701
        if (test3    <= ctol ).all(): break  # noqa: E701

    return x, iters, normr


# ---------------------------------------------------------------------------
# Inner primal solve
# ---------------------------------------------------------------------------

def solve_primal(
    Xs: List[torch.Tensor],
    y: torch.Tensor,
    log_lambdas: torch.Tensor,
    w0: Optional[torch.Tensor] = None,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, int]:
    """Solve  (X^T X + Lambda) w = X^T y  via LSMR on the augmented system.

    Parameters
    ----------
    Xs          : list of (T, p_k) tensors
    y           : (T,) or (T, n_targets) tensor
    log_lambdas : (K,) tensor, log regularisation per subspace
    w0          : optional warm start, shape (P,) or (P, n_targets)
    max_iter    : max LSMR iterations
    tol         : LSMR tolerance (atol = btol = tol)

    Returns
    -------
    w     : (P,) or (P, n_targets)
    iters : number of LSMR iterations
    """
    op = BandedRidgeOperator(Xs, log_lambdas)
    T, P = op.T, op.P

    batched = y.ndim == 2
    if not batched:
        y = y.unsqueeze(-1)

    n_targets = y.shape[1]
    b = torch.cat([y, torch.zeros(P, n_targets, dtype=y.dtype, device=y.device)], dim=0)

    w, iters, _ = _lsmr_core(
        op.matvec, op.rmatvec, b, w0, max_iter, tol, tol, 1e8, 0.0,
        y.dtype, y.device, P, n_targets,
    )

    if not batched:
        w = w.squeeze(-1)
    return w, iters


# ---------------------------------------------------------------------------
# Hold-out loss and its gradient via implicit differentiation
# ---------------------------------------------------------------------------

def _hold_out_loss(
    Xs_train: List[torch.Tensor],
    y_train: torch.Tensor,
    Xs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    log_lambdas: torch.Tensor,
    w_train: torch.Tensor,
) -> torch.Tensor:
    """Compute  (1/2n_val) ||X_val w - y_val||_F^2.

    Parameters
    ----------
    w_train : (P, n_targets) – primal weights from training fold

    Returns
    -------
    loss : scalar tensor
    """
    splits = [X.shape[1] for X in Xs_val]
    ws     = w_train.split(splits, dim=0)
    y_pred = sum(X @ wk for X, wk in zip(Xs_val, ws))  # (T_val, n_targets)
    residual = y_pred - y_val
    n_val = y_val.shape[0]
    return 0.5 * (residual ** 2).sum() / n_val


def compute_gradient(
    Xs_train: List[torch.Tensor],
    y_train: torch.Tensor,
    Xs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    log_lambdas: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    max_iter_inner: int = 300,
    tol_inner: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the gradient of the hold-out loss w.r.t. log_lambdas.

    Uses implicit differentiation:

        dL/d(delta_k) = -exp(delta_k) * <w_k, mu_k>

    where w solves the training system and
        mu = (X_tr^T X_tr + Lambda)^{-1} nabla_w L|_{w}

    nabla_w L = (1/n_val) X_val^T (X_val w - y_val),  shape (P, n_targets).

    The adjoint system is solved with the same LSMR code.

    Parameters
    ----------
    Xs_train, y_train : training data
    Xs_val,   y_val   : validation data
    log_lambdas       : (K,) or (K, n_targets) tensor  — current log regularisation.
                        (K,) is shared across targets; (K, n_targets) is per-target.
    w                 : optional pre-computed primal weights (training fold)
    max_iter_inner    : LSMR iterations for both forward and adjoint solves
    tol_inner         : LSMR tolerance

    Returns
    -------
    grad        : (K,) or (K, n_targets) gradient of hold-out loss w.r.t. log_lambdas
    w           : (P, n_targets) primal weights (returned for warm-starting)
    loss        : scalar, hold-out loss value
    """
    batched_y = y_train.ndim == 2
    if not batched_y:
        y_train = y_train.unsqueeze(-1)
        y_val   = y_val.unsqueeze(-1)

    n_targets = y_train.shape[1]
    splits    = [X.shape[1] for X in Xs_train]

    # --- forward solve ---
    if w is None:
        w, _ = solve_primal(Xs_train, y_train, log_lambdas, max_iter=max_iter_inner, tol=tol_inner)

    # --- hold-out residual and loss ---
    ws     = w.split(splits, dim=0)
    y_pred = sum(X @ wk for X, wk in zip(Xs_val, ws))   # (T_val, n_targets)

    # Per-target normalisation: scale by training-fold std so all targets
    # contribute equally regardless of amplitude (matches Himalaya's l2_neg_loss).
    y_scale = y_train.std(dim=0, keepdim=True).clamp(min=1e-8)  # (1, n_targets)
    n_val   = y_val.shape[0]

    residual = (y_pred - y_val)                                   # (T_val, n_targets)
    # nabla_w L = (1/n_val) X_val^T (residual / y_scale^2)
    residual_scaled = residual / (n_val * y_scale ** 2)           # (T_val, n_targets)

    # --- gradient w.r.t. w: nabla_w L = X_val^T residual_scaled ---
    splits_val = [X.shape[1] for X in Xs_val]
    grad_w = torch.cat(
        [X.T @ residual_scaled for X in Xs_val], dim=0
    )  # (P, n_targets)

    # --- adjoint solve: mu = (X_tr^T X_tr + Lambda)^{-1} grad_w ---
    # Augment rhs with zeros for the regularisation rows
    T_tr = y_train.shape[0]
    P    = sum(splits)
    op   = BandedRidgeOperator(Xs_train, log_lambdas)
    # We need to solve  A^T A mu = grad_w, i.e. solve in the same augmented
    # least-squares form: find mu s.t.  A mu ≈ [Xmu; Lambda^{1/2} mu] in LSQ.
    # That is exactly the same LSMR call as the forward solve but with rhs = grad_w
    # (the augmentation rhs zeros cancel out).
    b_adj = torch.cat(
        [op.matvec(grad_w)[:T_tr],   # X grad_w broadcast -- actually we want A^T b = grad_w
         torch.zeros(P, n_targets, dtype=y_train.dtype, device=y_train.device)],
        dim=0,
    )
    # Correct formulation: (A^T A) mu = grad_w  <=>  LSMR on A with rhs s.t.
    # A^T b_adj = grad_w.  Choose b_adj = A (grad_w / ||A||^2) is not trivial.
    # Simpler: solve the normal equations directly by passing grad_w as the
    # right-hand side to the *normal equations* solver, i.e. use rmatvec/matvec
    # pair directly -- LSMR minimises ||A mu - rhs|| and gives mu s.t. A^T A mu = A^T rhs.
    # We want A^T A mu = grad_w, so set rhs = A mu* where mu* is the solution.
    # The cleanest way: use LSMR with the *original* augmented system and rhs
    # chosen so that A^T rhs = grad_w.  Since A^T [v_top; v_bot] = X^T v_top + Lam^{1/2} v_bot,
    # we set v_top = X (grad_w) / ||X||^2 ... complicated.  Instead, we use a
    # direct CG on the normal equations via the _lsmr_core (which already solves
    # the augmented system and implicitly implements CG on the normal eqs).
    # Set b = [0; I] * grad_w projected: the simplest correct choice is to solve
    #    A mu = b_pinv   where b_pinv is the pseudo-inverse solution.
    # In practice, we just call _lsmr_core with b set to zero for the top block
    # and Lambda^{-1/2} grad_w for the bottom block -- this gives
    #    A^T b = Lambda^{1/2} * Lambda^{-1/2} grad_w = grad_w  ✓
    starts = [0] + list(_cumsum(splits[:-1]))
    ends   = list(_cumsum(splits))
    if log_lambdas.ndim == 2:  # per-target (K, n_targets)
        b_adj_bot = torch.cat(
            [
                torch.exp(-0.5 * log_lambdas[k]).unsqueeze(0) * grad_w[s:e]
                for k, (s, e) in enumerate(zip(starts, ends))
            ],
            dim=0,
        )
    else:
        b_adj_bot = torch.cat(
            [
                math.exp(-0.5 * float(log_lambdas[k])) * grad_w[s:e]
                for k, (s, e) in enumerate(zip(starts, ends))
            ],
            dim=0,
        )
    b_adj = torch.cat(
        [
            torch.zeros(T_tr, n_targets, dtype=y_train.dtype, device=y_train.device),
            b_adj_bot,
        ],
        dim=0,
    )

    mu, _, _ = _lsmr_core(
        op.matvec, op.rmatvec, b_adj, None, max_iter_inner, tol_inner, tol_inner,
        1e8, 0.0, y_train.dtype, y_train.device, P, n_targets,
    )  # (P, n_targets)

    # --- implicit-diff gradient ---
    # Per-subspace, per-target:
    #   dL/d(delta_{k,t}) = -exp(delta_{k,t}) * <w_k[:,t], mu_k[:,t]>
    mus  = mu.split(splits, dim=0)
    if log_lambdas.ndim == 2:  # per-target: (K, n_targets)
        grad_delta = torch.stack(
            [
                -torch.exp(log_lambdas[k]) * (wk * muk).sum(dim=0)
                for k, (wk, muk) in enumerate(zip(ws, mus))
            ]
        )  # (K, n_targets)
    else:
        grad_delta = torch.stack(
            [
                -math.exp(float(lk)) * (wk * muk).sum()
                for lk, wk, muk in zip(log_lambdas, ws, mus)
            ]
        )  # (K,)

    # Compute actual loss scalar (same normalisation as grad_w)
    y_pred2  = sum(X @ wk for X, wk in zip(Xs_val, ws))
    loss_val = 0.5 * ((y_pred2 - y_val) ** 2 / y_scale ** 2).sum() / n_val

    if not batched_y:
        w = w.squeeze(-1)

    return grad_delta, w, loss_val


def _cumsum(lst: list) -> list:
    out, s = [], 0
    for v in lst:
        s += v
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Ridge-CV initialisation
# ---------------------------------------------------------------------------

def _init_log_lambdas_ridge_cv(
    Xs: List[torch.Tensor],
    Y: torch.Tensor,
    cv_splits: List[Tuple[torch.Tensor, torch.Tensor]],
    alphas: Optional[torch.Tensor] = None,
    max_iter_inner: int = 100,
    tol_inner: float = 1e-3,
) -> torch.Tensor:
    """Initialise log_lambdas uniformly via scalar ridge CV.

    Fits a uniform-regularisation ridge (lambda_1 = ... = lambda_K = alpha)
    for each alpha on a grid, evaluates the average CV hold-out loss, and
    returns log(best_alpha) * ones(K).  All K subspaces share the same
    initial regularisation, which is the natural starting point before the
    per-subspace hyper-gradient optimisation breaks the symmetry.

    Parameters
    ----------
    Xs             : list of K tensors, each (T, p_k)
    Y              : (T, n_targets)
    cv_splits      : list of (train_idx, val_idx) pairs
    alphas         : 1-D tensor of candidate regularisation values
                     (default: logspace(-3, 6, 10))
    max_iter_inner : LSMR iterations (coarse; only needs a rough answer)
    tol_inner      : LSMR tolerance

    Returns
    -------
    log_lambdas : (K,) tensor, all equal to log(best_alpha)
    w_init      : list of (P, n_targets) tensors, per-fold primal weights at
                  best_alpha (normalised training data).  Used to pre-warm
                  the outer-loop w_cache, eliminating the second cold start.
    """
    K      = len(Xs)
    device = Xs[0].device
    dtype  = Xs[0].dtype

    if alphas is None:
        alphas = torch.logspace(-3, 6, 10, dtype=dtype, device=device)

    best_loss  = float("inf")
    best_alpha = float(alphas[0])
    best_ws    = None   # per-fold weights for the best alpha

    for alpha in alphas:
        log_lam     = torch.full((K,), math.log(float(alpha)), dtype=dtype, device=device)
        total_loss  = 0.0
        total_n_val = sum(int(val_idx.shape[0]) for _, val_idx in cv_splits)
        ws_folds    = []
        for train_idx, val_idx in cv_splits:
            Xs_tr  = [X[train_idx] for X in Xs]
            y_tr   = Y[train_idx]
            Xs_val = [X[val_idx]   for X in Xs]
            y_val  = Y[val_idx]
            # per-fold normalisation by training-fold statistics
            y_mean = y_tr.mean(dim=0, keepdim=True)
            y_std  = y_tr.std(dim=0, keepdim=True).clamp(min=1e-8)
            y_tr   = (y_tr  - y_mean) / y_std
            y_val  = (y_val - y_mean) / y_std
            weight = int(val_idx.shape[0]) / total_n_val
            w, _   = solve_primal(Xs_tr, y_tr, log_lam,
                                  max_iter=max_iter_inner, tol=tol_inner)
            ws_folds.append(w if w.ndim == 2 else w.unsqueeze(-1))
            splits_k = [X.shape[1] for X in Xs_tr]
            ws       = w.split(splits_k, dim=0)
            y_pred   = sum(X @ wk for X, wk in zip(Xs_val, ws))
            total_loss += weight * float(0.5 * ((y_pred - y_val) ** 2).sum() / y_val.shape[0])
        avg_loss = total_loss  # already a weighted sum
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_alpha = float(alpha)
            best_ws    = ws_folds

    n_targets = Y.shape[1] if Y.ndim == 2 else 1
    log_lam_1d = torch.full((K,), math.log(best_alpha), dtype=dtype, device=device)
    # Return (K, n_targets): uniform initialisation, per-target optimisation breaks symmetry later.
    return log_lam_1d.unsqueeze(-1).expand(K, n_targets).contiguous(), best_ws


# ---------------------------------------------------------------------------
# Outer hyperparameter optimiser
# ---------------------------------------------------------------------------

def optimise_log_lambdas_adam(
    Xs_train: List[torch.Tensor],
    y_train: torch.Tensor,
    Xs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    log_lambdas_init: Optional[torch.Tensor] = None,
    n_iter: int = 100,
    lr: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    max_iter_inner: int = 300,
    tol_inner: float = 1e-6,
    verbose: bool = False,
) -> Tuple[torch.Tensor, List[float]]:
    """Optimise log_lambdas with Adam gradient descent.

    Parameters
    ----------
    Xs_train, y_train : training split
    Xs_val,   y_val   : validation split
    log_lambdas_init  : (K,) init; defaults to zeros
    n_iter            : outer iterations
    lr                : Adam learning rate
    betas, eps        : Adam hyperparameters
    max_iter_inner    : LSMR iters for each inner solve
    tol_inner         : LSMR tolerance
    verbose           : print progress

    Returns
    -------
    log_lambdas : (K,) optimised
    losses      : list of validation losses per iteration
    """
    K = len(Xs_train)
    device = Xs_train[0].device
    dtype  = Xs_train[0].dtype

    if log_lambdas_init is None:
        log_lambdas = torch.zeros(K, dtype=dtype, device=device)
    else:
        log_lambdas = log_lambdas_init.clone()

    m  = torch.zeros_like(log_lambdas)   # 1st moment
    mv = torch.zeros_like(log_lambdas)   # 2nd moment

    losses = []
    w = None  # warm-start primal weights

    for t in range(1, n_iter + 1):
        grad, w, loss = compute_gradient(
            Xs_train, y_train, Xs_val, y_val, log_lambdas,
            w=w, max_iter_inner=max_iter_inner, tol_inner=tol_inner,
        )

        # Adam update
        m  = betas[0] * m  + (1 - betas[0]) * grad
        mv = betas[1] * mv + (1 - betas[1]) * grad ** 2
        mhat  = m  / (1 - betas[0] ** t)
        mvhat = mv / (1 - betas[1] ** t)
        log_lambdas = log_lambdas - lr * mhat / (mvhat.sqrt() + eps)

        # Invalidate warm-start when lambdas change (cheap; re-use is approximate)
        w = None

        losses.append(float(loss))
        if verbose:
            print(f"iter {t:4d}  loss={float(loss):.6f}  "
                  f"grad_norm={float(grad.norm()):.4f}  "
                  f"log_lam={log_lambdas.tolist()[:5]} ...")

    return log_lambdas, losses


def optimise_log_lambdas_lbfgs(
    Xs_train: List[torch.Tensor],
    y_train: torch.Tensor,
    Xs_val: List[torch.Tensor],
    y_val: torch.Tensor,
    log_lambdas_init: Optional[torch.Tensor] = None,
    n_iter: int = 50,
    max_iter_inner: int = 300,
    tol_inner: float = 1e-6,
    history_size: int = 10,
    verbose: bool = False,
) -> Tuple[torch.Tensor, List[float]]:
    """Optimise log_lambdas with L-BFGS.

    Uses torch.optim.LBFGS with a closure.

    Parameters
    ----------
    Same as optimise_log_lambdas_adam, minus Adam-specific params.
    history_size : L-BFGS history length

    Returns
    -------
    log_lambdas : (K,) optimised
    losses      : list of validation losses per L-BFGS step
    """
    K = len(Xs_train)
    device = Xs_train[0].device
    dtype  = Xs_train[0].dtype

    if log_lambdas_init is None:
        log_lambdas = torch.zeros(K, dtype=dtype, device=device)
    else:
        log_lambdas = log_lambdas_init.clone()

    log_lambdas = log_lambdas.requires_grad_(False)  # we manage grads manually

    # We wrap the gradient computation in a closure compatible with L-BFGS.
    # Since we use implicit differentiation (not autograd), we carry the
    # gradient manually via a Parameter-like approach.
    log_lam_param = torch.nn.Parameter(log_lambdas.detach().clone())
    optimizer = torch.optim.LBFGS(
        [log_lam_param], lr=1.0, max_iter=1, history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    losses = []

    for step in range(n_iter):
        def closure():
            optimizer.zero_grad()
            lam = log_lam_param.detach()
            grad, _, loss = compute_gradient(
                Xs_train, y_train, Xs_val, y_val, lam,
                w=None, max_iter_inner=max_iter_inner, tol_inner=tol_inner,
            )
            # Inject gradient manually
            log_lam_param.grad = grad.detach().clone()
            return loss

        loss = optimizer.step(closure)
        losses.append(float(loss) if loss is not None else float("nan"))

        if verbose:
            print(f"L-BFGS step {step:4d}  loss={losses[-1]:.6f}")

    return log_lam_param.detach(), losses


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def solve_banded_ridge(
    Xs: List[torch.Tensor],
    Y: torch.Tensor,
    cv_splits: List[Tuple[torch.Tensor, torch.Tensor]],
    log_lambdas_init: Optional[torch.Tensor] = None,
    n_iter_outer: int = 100,
    lr: float = 0.1,
    optimizer: str = "adam",
    max_iter_inner: int = 300,
    max_iter_inner_init: int = 50,
    tol_inner = 1e-6,
    init_alphas: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Full banded ridge regression with per-subspace regularisation.

    Fits the model

        min_w  (1/2)||Xw - Y||_F^2  +  (1/2) sum_k exp(delta_k) ||w_k||_F^2

    by alternating:
      1. Inner solve: LSMR on the augmented primal system.
      2. Outer hyper-gradient step: implicit differentiation + Adam or L-BFGS.

    Cross-validation is performed over the provided splits to drive the outer
    optimisation; a final refit is done on the full dataset.

    When ``log_lambdas_init`` is None the regularisation is initialised by a
    scalar ridge CV over ``init_alphas``: the best uniform alpha is found and
    all log_lambdas are set to log(alpha*).  Pass ``log_lambdas_init`` to skip
    this step.

    Outer optimisation (Adam) uses a Himalaya-style Dirichlet reparameterisation
    to prevent degenerate solutions where some subspaces are driven to infinite
    regularisation.  The parameterisation is:

        log_lambda[k] = log_alpha - log_softmax(phi)[k]

    where ``phi`` (shape K or K×n_targets) are unconstrained allocation logits
    and ``log_alpha`` (scalar or n_targets) is the total regularisation scale.
    The softmax coupling ensures all subspaces share a finite budget and prevents
    any individual lambda from diverging independently.

    Parameters
    ----------
    Xs               : list of K tensors, each (T, p_k)
    Y                : (T,) or (T, n_targets)
    cv_splits        : list of (train_idx, val_idx) integer tensors
    log_lambdas_init : (K,) initial log-lambdas; if None, inferred by ridge CV
    n_iter_outer     : outer optimisation iterations
    lr               : learning rate
    optimizer        : "adam" or "lbfgs"
    max_iter_inner   : LSMR iterations per inner solve (warm iterations 2+)
    max_iter_inner_init : LSMR iterations on the first (cold) outer iteration.
                       Mirrors Himalaya's first-pass 50-step CG.  Default 50.
    tol_inner        : LSMR tolerance — either a float (constant) or a sequence
                       of length n_iter_outer (per-iteration schedule).  On the
                       first outer iteration the tolerance is additionally
                       clamped to max(tol, 1e-2) so the cold-start solution is
                       only loosely accurate, matching Himalaya's cg_tol logic.
    init_alphas      : 1-D tensor of alpha candidates for the ridge CV init
                       (default: logspace(-3, 6, 10))
    verbose          : print progress

    Returns
    -------
    w            : list of K tensors (P_k, n_targets), per-subspace primal weights
    log_lambdas  : (K, n_targets) optimised log regularisation parameters
    losses       : list of CV losses per outer iteration
    """
    # Average gradient over CV folds
    K      = len(Xs)
    device = Xs[0].device
    dtype  = Xs[0].dtype

    batched = Y.ndim == 2
    if not batched:
        Y = Y.unsqueeze(-1)
    n_targets = Y.shape[1]

    # ---- build per-iteration tol and max_iter schedules ----
    # tol_inner may be a float or a sequence of length n_iter_outer.
    # Mirrors Himalaya's cg_tol handling in _hyper_gradient.py.
    if isinstance(tol_inner, (int, float)):
        tol_schedule = [float(tol_inner)] * n_iter_outer
    else:
        tol_schedule = list(tol_inner)
    # First outer iteration is always cold (w_cache is None): use more LSMR
    # steps and a looser tolerance, exactly like Himalaya's ii==0 special case.
    max_iter_schedule = [max_iter_inner] * n_iter_outer
    max_iter_schedule[0] = max_iter_inner_init
    tol_schedule[0]      = max(tol_schedule[0], 1e-2)

    # base tolerance for the ridge-CV init (use the tightest requested tol)
    _tol_base = min(tol_schedule)

    if log_lambdas_init is None:
        if verbose:
            print("[init] running scalar ridge CV to initialise log_lambdas ...")
        log_lambdas, w_init = _init_log_lambdas_ridge_cv(
            Xs, Y, cv_splits,
            alphas=init_alphas,
            max_iter_inner=max(max_iter_inner_init, max_iter_inner // 4),
            tol_inner=max(1e-3, _tol_base * 1e3),
        )
        if verbose:
            print(f"[init] best uniform alpha = {math.exp(float(log_lambdas[0, 0])):.4g}  "
                  f"(log = {float(log_lambdas[0, 0]):.3f})")
    else:
        log_lambdas = log_lambdas_init.clone()
        # broadcast (K,) -> (K, n_targets) if needed
        if log_lambdas.ndim == 1:
            log_lambdas = log_lambdas.unsqueeze(-1).expand(K, n_targets).contiguous()
        w_init = None

    # ---- per-fold warm-start cache (persists across outer iterations) ----
    # When ridge-CV was run, pre-populate from its best-alpha weights so the
    # first outer iteration is already warm (no second cold start).
    # When log_lambdas_init is provided directly, start cold as before.
    if w_init is not None:
        w_cache: List[Optional[torch.Tensor]] = w_init
    else:
        w_cache: List[Optional[torch.Tensor]] = [None] * len(cv_splits)

    # ---- helper: average gradient over CV folds ----
    def _cv_gradient(log_lam, iter_tol, iter_max):
        total_grad  = torch.zeros_like(log_lam)
        total_loss  = 0.0
        total_n_val = sum(int(val_idx.shape[0]) for _, val_idx in cv_splits)
        for kk, (train_idx, val_idx) in enumerate(cv_splits):
            Xs_tr  = [X[train_idx] for X in Xs]
            y_tr   = Y[train_idx]
            Xs_val = [X[val_idx]   for X in Xs]
            y_val  = Y[val_idx]
            # per-fold normalisation by training-fold statistics (matches Himalaya)
            y_mean = y_tr.mean(dim=0, keepdim=True)
            y_std  = y_tr.std(dim=0, keepdim=True).clamp(min=1e-8)
            y_tr   = (y_tr  - y_mean) / y_std
            y_val  = (y_val - y_mean) / y_std
            # weight proportionally to validation-fold size
            weight = int(val_idx.shape[0]) / total_n_val
            g, w_fold, l = compute_gradient(
                Xs_tr, y_tr, Xs_val, y_val, log_lam,
                w=w_cache[kk],       # warm-start from previous outer iter
                max_iter_inner=iter_max, tol_inner=iter_tol,
            )
            w_cache[kk] = w_fold if w_fold.ndim == 2 else w_fold.unsqueeze(-1)
            total_grad = total_grad + weight * g
            total_loss += weight * float(l)
        return total_grad, total_loss

    # ---- outer optimisation ----
    losses = []

    if optimizer == "adam":
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        # ---- Dirichlet reparameterisation (Himalaya-style) ----------------------
        # log_lambda[k] = log_alpha - log_softmax(phi)[k]
        # phi: (K,) or (K, n_t) — unconstrained allocation logits
        # log_alpha: scalar or (n_t,) — total regularisation scale
        # With phi=0: softmax=1/K, log_lambda = log_alpha + log(K).
        # Init: log_alpha = log_lambdas_init[0] - log(K) to match ridge CV alpha.
        per_target = log_lambdas.ndim == 2
        phi = torch.zeros_like(log_lambdas)              # (K,) or (K, n_t)
        if per_target:
            log_alpha = log_lambdas[0].clone() - math.log(K)   # (n_t,)
        else:
            log_alpha = log_lambdas.mean().clone() - math.log(K)  # scalar
        m_phi = torch.zeros_like(phi)
        v_phi = torch.zeros_like(phi)
        m_la  = torch.zeros_like(log_alpha)
        v_la  = torch.zeros_like(log_alpha)

        for t in tqdm.tqdm(range(1, n_iter_outer + 1)):
            ii = t - 1
            # Reconstruct log_lambdas from Dirichlet parameters
            log_sm = torch.log_softmax(phi, dim=0)               # (K,) or (K, n_t)
            if per_target:
                log_lambdas = log_alpha.unsqueeze(0) - log_sm    # (K, n_t)
            else:
                log_lambdas = log_alpha - log_sm                 # (K,)

            g, loss = _cv_gradient(log_lambdas, tol_schedule[ii], max_iter_schedule[ii])

            # Transform gradient from log_lambda-space to (phi, log_alpha)-space:
            #   dL/d phi[j]    = softmax(phi)[j] * sum_k g[k] - g[j]
            #   dL/d log_alpha = sum_k g[k]
            sm = torch.softmax(phi, dim=0)
            if per_target:
                G             = g.sum(dim=0)                     # (n_t,)
                grad_phi      = sm * G.unsqueeze(0) - g          # (K, n_t)
            else:
                G             = g.sum()                          # scalar
                grad_phi      = sm * G - g                       # (K,)
            grad_log_alpha = G

            # Adam update — phi
            m_phi = b1 * m_phi + (1 - b1) * grad_phi
            v_phi = b2 * v_phi + (1 - b2) * grad_phi ** 2
            phi   = phi   - lr * (m_phi / (1 - b1 ** t)) / ((v_phi / (1 - b2 ** t)).sqrt() + eps_adam)

            # Adam update — log_alpha
            m_la     = b1 * m_la + (1 - b1) * grad_log_alpha
            v_la     = b2 * v_la + (1 - b2) * grad_log_alpha ** 2
            log_alpha = log_alpha - lr * (m_la / (1 - b1 ** t)) / ((v_la / (1 - b2 ** t)).sqrt() + eps_adam)

            losses.append(loss)
            if verbose:
                ll = log_alpha.unsqueeze(0) - torch.log_softmax(phi, dim=0) if per_target \
                     else log_alpha - torch.log_softmax(phi, dim=0)
                print(f"[adam] iter {t:4d}  cv_loss={loss:.6f}  "
                      f"|grad_phi|={float(grad_phi.norm()):.4f}  "
                      f"log_alpha={log_alpha.mean().item() if per_target else float(log_alpha):.3f}  "
                      f"ll_range=[{ll.min().item():.2f}, {ll.max().item():.2f}]")

        # Reconstruct final log_lambdas
        log_sm = torch.log_softmax(phi, dim=0)
        log_lambdas = log_alpha.unsqueeze(0) - log_sm if per_target else log_alpha - log_sm

    elif optimizer == "lbfgs":
        log_lam_param = torch.nn.Parameter(log_lambdas.detach().clone())
        # opt = torch.optim.LBFGS(
        #     [log_lam_param], lr=1.0, max_iter=1, history_size=10,
        #     line_search_fn="strong_wolfe",
        # )
        # for step in tqdm.tqdm(range(n_iter_outer)):
        #     def closure():
        #         opt.zero_grad()
        #         lam  = log_lam_param.detach()
        #         grad, loss_val = _cv_gradient(lam)
        #         log_lam_param.grad = grad.clone()
        #         losses.append(loss_val)
        #         if verbose:
        #             print(f"[lbfgs] step {step:4d}  cv_loss={loss_val:.6f}")
        #         return torch.tensor(loss_val, dtype=dtype, device=device)
        #     opt.step(closure)

        opt = torch.optim.LBFGS(
            [log_lam_param], lr=lr, max_iter=1, history_size=10,
            line_search_fn=None,   # fixed step, no line search
        )
        for step in tqdm.tqdm(range(n_iter_outer)):
            def closure(step=step):
                opt.zero_grad()
                lam = log_lam_param.detach()
                grad, loss_val = _cv_gradient(lam, tol_schedule[step], max_iter_schedule[step])
                log_lam_param.grad = grad.clone()
                closure._loss = loss_val          # stash, don't append yet
                return torch.tensor(loss_val, dtype=dtype, device=device)
            opt.step(closure)
            losses.append(closure._loss) 
        log_lambdas = log_lam_param.detach()

    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Use 'adam' or 'lbfgs'.")

    # ---- final refit on all data ----
    w, _ = solve_primal(Xs, Y, log_lambdas, max_iter=max_iter_inner * 2, tol=_tol_base)

    if not batched:
        w = w.squeeze(-1)
    
    ## Finally: we output w as a list of per-subspace tensors, matching the input Xs structure.
    splits = [X.shape[1] for X in Xs]
    w = w.split(splits, dim=0)

    return w, log_lambdas, losses
