"""Numpy linear-softmax policy + linear value baseline for retrieval RL (advanced-RAG A6.2d).

The CI-floor learner backend: a plain **linear-softmax** (a.k.a. multinomial-logit / max-entropy)
policy over the A6.2b discrete action space, differentiable in **closed form** so REINFORCE and
behavior cloning need no autodiff (numpy only — torch stays in the isolated ``[rl]`` extra, A6.2e).

**Model.** For state ``s`` (dim ``d``) and action ``a ∈ {0..K-1}``, with each action's weight row
and bias folded into one augmented matrix ``W ∈ R^{K×(d+1)}`` and ``x̃ = [s; 1]``::

    logits  z = W x̃            π(a | s) = softmax(z)_a = exp(z_a) / Σ_k exp(z_k)

**Action masking.** Illegal actions (a discovered slot with no entity) get a ``−inf`` logit ⇒
probability 0, so the policy never proposes them and the gradient wrt their rows is 0. STOP (at
index 0) is always legal, so at least one logit is finite and the masked softmax is well defined.

**Gradient (closed form — why no autodiff is needed).** The softmax score identity gives::

    ∂ log π(a | s) / ∂ z_k      = 1[k = a] − π(k | s)
    ∂ log π(a | s) / ∂ W[k, :]  = (1[k = a] − π(k | s)) x̃      (chain rule, z_k = W[k,:]·x̃)

i.e. ``grad_log_prob(a, s) = outer(onehot(a) − π(·|s), x̃)``. This one quantity is both the REINFORCE
score (weighted by returns) and the negated cross-entropy gradient used by behavior cloning.
"""

from __future__ import annotations

import numpy as np


class LinearSoftmaxPolicy:
    """Linear-softmax policy over a fixed discrete action space (numpy, closed-form gradient).

    Parameters live in a single augmented weight matrix ``W`` of shape ``(n_actions, d + 1)`` (the
    trailing column is the per-action bias; ``x̃ = [s; 1]``). The action-space *order* is
    load-bearing — logits index into it — so it must match the A6.2b space the policy was built for.
    Holds no RNG: ``act`` takes an explicit ``np.random.Generator`` (determinism invariant; the
    trainer owns seeding).
    """

    def __init__(
        self,
        d: int,
        n_actions: int,
        *,
        seed: int = 0,
        init_scale: float = 0.0,
        bias: bool = True,
    ) -> None:
        if d <= 0 or n_actions <= 0:
            raise ValueError("d and n_actions must be positive")
        self.d = d
        self.n_actions = n_actions
        self.bias = bias
        self._p = d + (1 if bias else 0)  # augmented input width (bias folded in)
        rng = np.random.default_rng(seed)
        # init_scale=0 ⇒ zero weights ⇒ uniform-over-legal initial policy (the usual REINFORCE
        # start); a positive scale breaks symmetry with a seeded Gaussian (used for grad checks).
        self.W: np.ndarray = (
            init_scale * rng.standard_normal((n_actions, self._p))
            if init_scale > 0
            else np.zeros((n_actions, self._p))
        )

    # ---- forward -------------------------------------------------------------------
    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Append the constant bias feature: ``x -> [x; 1]`` (identity when ``bias=False``)."""
        xf: np.ndarray = np.asarray(x, dtype=np.float64)
        if self.bias:
            augmented: np.ndarray = np.concatenate([xf, np.ones(1, dtype=np.float64)])
            return augmented
        return xf

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Raw (unmasked) action logits ``z = W x̃``, shape ``(n_actions,)``."""
        z: np.ndarray = self.W @ self._augment(x)
        return z

    def _masked_logits(self, x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        """Logits with illegal actions forced to ``−inf`` (``mask=None`` ⇒ all legal)."""
        z = self.logits(x)
        if mask is not None:
            z = np.where(np.asarray(mask, dtype=bool), z, -np.inf)
        return z

    def action_probs(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
        """Masked softmax probabilities over actions (exact 0 on illegal actions; sums to 1).

        Numerically stable: subtract the max *finite* logit before exponentiating. At least one
        action (STOP) is always legal, so a finite max exists.
        """
        z = self._masked_logits(x, mask)
        finite = np.isfinite(z)
        m = np.max(z[finite])
        ez = np.exp(z - m)
        ez[~finite] = 0.0  # −inf − m = −inf ⇒ exp = 0 already; make it exact (no NaN leakage)
        probs: np.ndarray = ez / float(ez.sum())
        return probs

    def log_prob(self, a: int, x: np.ndarray, *, mask: np.ndarray | None = None) -> float:
        """``log π(a | s)`` via a stable log-sum-exp over the *legal* logits."""
        z = self._masked_logits(x, mask)
        finite = np.isfinite(z)
        m = np.max(z[finite])
        lse = m + np.log(np.sum(np.exp(z[finite] - m)))
        return float(z[a] - lse)

    # ---- sampling / greedy ---------------------------------------------------------
    def act(
        self, x: np.ndarray, *, rng: np.random.Generator, mask: np.ndarray | None = None
    ) -> tuple[int, float]:
        """Sample an action ``a ~ π(·|s)`` and return ``(a, log π(a|s))`` (masked-safe)."""
        p = self.action_probs(x, mask=mask)
        a = int(rng.choice(self.n_actions, p=p))
        return a, self.log_prob(a, x, mask=mask)

    def greedy_action(self, x: np.ndarray, *, mask: np.ndarray | None = None) -> int:
        """Deterministic ``argmax_a π(a|s)`` over legal actions (for evaluation)."""
        return int(np.argmax(self._masked_logits(x, mask)))

    # ---- gradient ------------------------------------------------------------------
    def grad_log_prob(
        self, a: int, x: np.ndarray, *, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """``∂ log π(a|s) / ∂W`` = ``outer(onehot(a) − π(·|s), x̃)``, shape ``(n_actions, d+1)``.

        The REINFORCE score and (negated) BC cross-entropy gradient. Illegal-action rows are 0
        (their probability is 0 and they are never the taken action ``a``).
        """
        xa = self._augment(x)
        p = self.action_probs(x, mask=mask)
        onehot = np.zeros(self.n_actions, dtype=np.float64)
        onehot[a] = 1.0
        return np.outer(onehot - p, xa)


class LinearValueBaseline:
    """State-only linear value baseline ``b(s) = θ·[s; 1]`` for REINFORCE variance reduction.

    A baseline that depends on the state but not the action leaves the policy-gradient estimator
    unbiased while cutting its variance. Fit by ridge least squares on ``(state, return)`` pairs
    (closed form ⇒ deterministic, no learning rate); the bias term is left unregularized.
    """

    def __init__(self, d: int, *, bias: bool = True, ridge: float = 1e-3) -> None:
        if d <= 0:
            raise ValueError("d must be positive")
        self.d = d
        self.bias = bias
        self.ridge = ridge
        self.theta = np.zeros(d + (1 if bias else 0), dtype=np.float64)

    def _design(self, x: np.ndarray) -> np.ndarray:
        """Design matrix: rows ``[s; 1]`` (bias column appended when ``bias=True``)."""
        rows = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self.bias:
            return np.hstack([rows, np.ones((rows.shape[0], 1))])
        return rows

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted baseline value(s) for a batch of states, shape ``(n,)``."""
        return self._design(x) @ self.theta

    def fit(self, x: np.ndarray, returns: np.ndarray) -> LinearValueBaseline:
        """Ridge least-squares fit of ``θ`` to ``(states, returns)`` (bias unregularized)."""
        design = self._design(x)
        y = np.asarray(returns, dtype=np.float64)
        n_feat = design.shape[1]
        reg = self.ridge * np.eye(n_feat)
        if self.bias:
            reg[-1, -1] = 0.0  # do not shrink the intercept toward 0
        self.theta = np.linalg.solve(design.T @ design + reg, design.T @ y)
        return self
