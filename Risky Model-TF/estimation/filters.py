"""estimation.filters
I came across many reporting errors when I run test for filters.
The reason is:
- Anything called inside `tfp.mcmc.sample_chain` must be graph-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def effective_sample_size(logw: tf.Tensor) -> tf.Tensor:
    """Compute ESS from log-weights.

    Args:
      logw: (N,) or (C,N) log-weights

    Returns:
      ESS: () or (C,)
    """
    w = tf.nn.softmax(logw, axis=-1)
    return 1.0 / tf.reduce_sum(tf.square(w), axis=-1)


def systematic_resample(logw: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Systematic resampling indices (graph-safe).

    Supports:
      logw: (N,) -> idx: (N,)
      logw: (C,N) -> idx: (C,N)

    Note: batched case folds chain index into seed.
    """
    logw = tf.convert_to_tensor(logw, tf.float32)
    seed = tf.convert_to_tensor(seed, tf.int32)

    if logw.shape.rank == 1:
        w = tf.nn.softmax(logw, axis=-1)
        N = tf.shape(w)[0]
        cdf = tf.cumsum(w)
        u0 = tf.random.stateless_uniform((), seed=seed, minval=0.0, maxval=1.0)
        js = (u0 + tf.cast(tf.range(N), tf.float32)) / tf.cast(N, tf.float32)
        idx = tf.searchsorted(cdf, js, side="left")
        return tf.cast(idx, tf.int32)

    # Batched (C,N)
    C = tf.shape(logw)[0]
    N = tf.shape(logw)[1]

    def _one_chain(c: tf.Tensor) -> tf.Tensor:
        seed_c = tf.random.experimental.stateless_fold_in(seed, c)
        w = tf.nn.softmax(logw[c], axis=-1)
        cdf = tf.cumsum(w)
        u0 = tf.random.stateless_uniform((), seed=seed_c, minval=0.0, maxval=1.0)
        js = (u0 + tf.cast(tf.range(N), tf.float32)) / tf.cast(N, tf.float32)
        idx = tf.searchsorted(cdf, js, side="left")
        return tf.cast(idx, tf.int32)

    return tf.vectorized_map(_one_chain, tf.range(C, dtype=tf.int32))


@dataclass
class PFDiagnostics:
    """Diagnostics stored as Tensors (graph-safe)."""

    ess_min: tf.Tensor  # (C,)
    ess_mean: tf.Tensor  # (C,)
    num_resamples: tf.Tensor  # (C,)


def _as_chain_vectors(rho: tf.Tensor, sigma_eps: tf.Tensor, sigma_obs: tf.Tensor):
    """Force params into (C,) vectors in a graph-safe way."""
    rho = tf.reshape(tf.convert_to_tensor(rho, tf.float32), (-1,))
    sigma_eps = tf.reshape(tf.convert_to_tensor(sigma_eps, tf.float32), (-1,))
    sigma_obs = tf.reshape(tf.convert_to_tensor(sigma_obs, tf.float32), (-1,))
    C = tf.shape(rho)[0]
    return C, rho, sigma_eps, sigma_obs


def bootstrap_pf_loglik_lnz(
    y_lnz: tf.Tensor,
    rho: tf.Tensor,
    sigma_eps: tf.Tensor,
    sigma_obs: tf.Tensor,
    num_particles: int,
    seed: int,
    resample_threshold: float = 0.5,
) -> Tuple[tf.Tensor, PFDiagnostics]:
    """Bootstrap PF log-likelihood for ln z with systematic resampling.

    State: x_t = ln z_t
    Transition: x_{t+1} = rho * x_t + eps, eps ~ N(0, sigma_eps)
    Obs: y_t = x_t + nu, nu ~ N(0, sigma_obs)

    Returns:
      loglik: scalar (if C==1) or (C,)
      diags: PFDiagnostics (always (C,) tensors)
    """
    y_lnz = tf.convert_to_tensor(y_lnz, tf.float32)
    T = tf.shape(y_lnz)[0]
    N = tf.cast(num_particles, tf.int32)

    C, rho_b, sigma_eps_b, sigma_obs_b = _as_chain_vectors(rho, sigma_eps, sigma_obs)
    squeeze_out = tf.equal(C, 1)

    base_seed = tf.constant([seed, seed + 12345], tf.int32)

    def _noise(
        t: tf.Tensor, salt: tf.Tensor, std: tf.Tensor, mean: tf.Tensor
    ) -> tf.Tensor:
        """Return (C,N) stateless normal draws."""

        def _one_chain(c: tf.Tensor) -> tf.Tensor:
            seed_c = tf.random.experimental.stateless_fold_in(base_seed, c)
            seed_ct = tf.random.experimental.stateless_fold_in(seed_c, t * 1000 + salt)
            return tf.random.stateless_normal(
                (N,), seed=seed_ct, mean=mean[c], stddev=std[c]
            )

        return tf.vectorized_map(_one_chain, tf.range(C, dtype=tf.int32))

    # Prior for x0: diffuse around first observation
    mean0 = tf.fill((C,), tf.cast(y_lnz[0], tf.float32))
    std0 = tf.ones((C,), tf.float32)
    x = _noise(
        tf.constant(0, tf.int32), tf.constant(11, tf.int32), std0, mean0
    )  # (C,N)

    # Robust broadcasting (NO slicing like sigma_obs_b[:, None])
    sigma_obs_mat = tf.reshape(
        sigma_obs_b, (-1, 1)
    )  # (C,1) even if scalar at trace time
    scale0 = tf.broadcast_to(sigma_obs_mat, tf.shape(x))  # (C,N)
    logw = tfd.Normal(loc=x, scale=scale0).log_prob(y_lnz[0])  # (C,N)

    loglik0 = tf.reduce_logsumexp(logw, axis=-1) - tf.math.log(
        tf.cast(N, tf.float32)
    )  # (C,)
    logw = logw - tf.reduce_logsumexp(logw, axis=-1, keepdims=True)

    t0 = tf.constant(1, tf.int32)
    ess_min0 = tf.fill((C,), tf.cast(N, tf.float32))
    ess_sum0 = tf.zeros((C,), tf.float32)
    ess_count0 = tf.zeros((C,), tf.float32)
    resamples0 = tf.zeros((C,), tf.int32)

    def cond(t, x, logw, loglik, ess_min, ess_sum, ess_count, resamples):
        return tf.less(t, T)

    def body(t, x, logw, loglik, ess_min, ess_sum, ess_count, resamples):
        ess_t = effective_sample_size(logw)  # (C,)
        ess_min = tf.minimum(ess_min, ess_t)
        ess_sum = ess_sum + ess_t
        ess_count = ess_count + 1.0

        # resample if ESS low
        do_resample = tf.less(
            ess_t, tf.cast(resample_threshold, tf.float32) * tf.cast(N, tf.float32)
        )  # (C,)
        idx = systematic_resample(logw, seed=base_seed + tf.stack([t, 0]))  # (C,N)

        def _resample_one_chain(args):
            _, x_c, logw_c, idx_c, do_c = args
            x_new = tf.cond(do_c, lambda: tf.gather(x_c, idx_c), lambda: x_c)
            logw_new = tf.cond(do_c, lambda: tf.zeros_like(logw_c), lambda: logw_c)
            r_inc = tf.cond(
                do_c, lambda: tf.constant(1, tf.int32), lambda: tf.constant(0, tf.int32)
            )
            return x_new, logw_new, r_inc

        x_new, logw_new, r_inc = tf.map_fn(
            _resample_one_chain,
            (tf.range(C, dtype=tf.int32), x, logw, idx, do_resample),
            fn_output_signature=(tf.float32, tf.float32, tf.int32),
        )
        x = x_new
        logw = logw_new
        resamples = resamples + r_inc

        # propagate
        mean_eps = tf.zeros((C,), tf.float32)
        x = rho_b[:, None] * x + _noise(
            t, tf.constant(21, tf.int32), sigma_eps_b, mean_eps
        )

        # weight update
        scale_t = tf.broadcast_to(sigma_obs_mat, tf.shape(x))  # (C,N)
        logw = logw + tfd.Normal(loc=x, scale=scale_t).log_prob(y_lnz[t])

        # loglik increment
        loglik = (
            loglik
            + tf.reduce_logsumexp(logw, axis=-1)
            - tf.math.log(tf.cast(N, tf.float32))
        )

        # normalize
        logw = logw - tf.reduce_logsumexp(logw, axis=-1, keepdims=True)
        return t + 1, x, logw, loglik, ess_min, ess_sum, ess_count, resamples

    _, _, _, loglikT, ess_minT, ess_sumT, ess_countT, resamplesT = tf.while_loop(
        cond,
        body,
        loop_vars=(t0, x, logw, loglik0, ess_min0, ess_sum0, ess_count0, resamples0),
        shape_invariants=(
            t0.get_shape(),
            tf.TensorShape([None, None]),  # x
            tf.TensorShape([None, None]),  # logw
            tf.TensorShape([None]),  # loglik
            tf.TensorShape([None]),  # ess_min
            tf.TensorShape([None]),  # ess_sum
            tf.TensorShape([None]),  # ess_count
            tf.TensorShape([None]),  # resamples
        ),
        parallel_iterations=1,
    )

    ess_meanT = tf.where(
        tf.greater(ess_countT, 0.0),
        ess_sumT / ess_countT,
        tf.fill((C,), tf.cast(N, tf.float32)),
    )
    diags = PFDiagnostics(
        ess_min=ess_minT, ess_mean=ess_meanT, num_resamples=resamplesT
    )

    loglik_out = tf.cond(squeeze_out, lambda: loglikT[0], lambda: loglikT)
    return loglik_out, diags


def importance_sampler_loglik_lnz(
    y_lnz: tf.Tensor,
    rho: tf.Tensor,
    sigma_eps: tf.Tensor,
    sigma_obs: tf.Tensor,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    """Differentiable sequential importance sampler log-likelihood (no resampling).

    Returns:
      loglik: scalar (if C==1) or (C,)
    """
    y_lnz = tf.convert_to_tensor(y_lnz, tf.float32)
    T = tf.shape(y_lnz)[0]
    N = tf.cast(num_particles, tf.int32)

    C, rho_b, sigma_eps_b, sigma_obs_b = _as_chain_vectors(rho, sigma_eps, sigma_obs)
    squeeze_out = tf.equal(C, 1)

    base_seed = tf.constant([seed, seed + 7], tf.int32)

    def _noise(
        t: tf.Tensor, salt: tf.Tensor, std: tf.Tensor, mean: tf.Tensor
    ) -> tf.Tensor:
        def _one_chain(c: tf.Tensor) -> tf.Tensor:
            seed_c = tf.random.experimental.stateless_fold_in(base_seed, c)
            seed_ct = tf.random.experimental.stateless_fold_in(seed_c, t * 1000 + salt)
            return tf.random.stateless_normal(
                (N,), seed=seed_ct, mean=mean[c], stddev=std[c]
            )

        return tf.vectorized_map(_one_chain, tf.range(C, dtype=tf.int32))

    mean0 = tf.fill((C,), tf.cast(y_lnz[0], tf.float32))
    std0 = tf.ones((C,), tf.float32)
    x = _noise(
        tf.constant(0, tf.int32), tf.constant(11, tf.int32), std0, mean0
    )  # (C,N)

    sigma_obs_mat = tf.reshape(sigma_obs_b, (-1, 1))  # (C,1)
    scale0 = tf.broadcast_to(sigma_obs_mat, tf.shape(x))
    logw = tfd.Normal(loc=x, scale=scale0).log_prob(y_lnz[0])
    loglik0 = tf.reduce_logsumexp(logw, axis=-1) - tf.math.log(tf.cast(N, tf.float32))
    logw = logw - tf.reduce_logsumexp(logw, axis=-1, keepdims=True)

    t0 = tf.constant(1, tf.int32)

    def cond(t, x, logw, loglik):
        return tf.less(t, T)

    def body(t, x, logw, loglik):
        mean_eps = tf.zeros((C,), tf.float32)
        x = rho_b[:, None] * x + _noise(
            t, tf.constant(21, tf.int32), sigma_eps_b, mean_eps
        )
        scale_t = tf.broadcast_to(sigma_obs_mat, tf.shape(x))
        logw = logw + tfd.Normal(loc=x, scale=scale_t).log_prob(y_lnz[t])
        loglik = (
            loglik
            + tf.reduce_logsumexp(logw, axis=-1)
            - tf.math.log(tf.cast(N, tf.float32))
        )
        logw = logw - tf.reduce_logsumexp(logw, axis=-1, keepdims=True)
        return t + 1, x, logw, loglik

    _, _, _, loglikT = tf.while_loop(
        cond,
        body,
        loop_vars=(t0, x, logw, loglik0),
        shape_invariants=(
            t0.get_shape(),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None]),
        ),
        parallel_iterations=1,
    )

    return tf.cond(squeeze_out, lambda: loglikT[0], lambda: loglikT)
