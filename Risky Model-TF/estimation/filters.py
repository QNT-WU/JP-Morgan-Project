"""Forward-filter utilities for Bayesian estimation.

This module implements the deterministic finite-state forward recursion used in
Bayesian estimation for the risky-debt model. The latent productivity state is
approximated on a fixed Rouwenhorst grid and the likelihood is evaluated by a
scaled Hamilton filter.

The implementation is TensorFlow-first so it can be plugged into TensorFlow
Probability samplers while remaining numerically stable on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ForwardFilterResult:
    """Container for the scaled forward-filter output.

    Attributes:
        filtered_probs: Filtered state probabilities with shape ``[T, Nz]``.
        scale_factors: Per-period scaling constants ``c_t`` with shape ``[T]``.
        log_likelihood: Total log likelihood ``sum_t log(c_t)``.
    """

    filtered_probs: tf.Tensor
    scale_factors: tf.Tensor
    log_likelihood: tf.Tensor


class FiniteStateForwardFilter:
    r"""Scaled forward filter for a fixed finite-state Markov chain.

    The recursion is

    .. math::
        \alpha_t(j) = \ell_t(j) \sum_i \xi_{t-1}(i) P_{ij},
        \qquad
        c_t = \sum_j \alpha_t(j),
        \qquad
        \xi_t(j) = \alpha_t(j) / c_t.

    where ``log_emissions[t, j]`` stores :math:`\log \ell_t(j)`.

    TensorFlow 2.16+ is stricter about what a ``tf.function`` may return. In
    particular, Python dataclasses are not accepted as compiled return values.
    To keep the compiled recursion while preserving a convenient result object,
    the public :meth:`run` method wraps a tensor-only private implementation.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize the filter.

        Args:
            eps: Positive floor for numerical stability.
        """
        self.eps = tf.constant(eps, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def _run_tensors(
        self,
        log_emissions: tf.Tensor,
        init_probs: tf.Tensor,
        transition_matrix: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run the compiled forward recursion and return tensors only.

        Args:
            log_emissions: Tensor with shape ``[T, Nz]``.
            init_probs: Initial latent-state probabilities with shape ``[Nz]``.
            transition_matrix: Markov transition matrix with shape ``[Nz, Nz]``.

        Returns:
            A tuple ``(filtered_probs, scale_factors, log_likelihood)``.
        """
        log_emissions = tf.convert_to_tensor(log_emissions, dtype=tf.float32)
        init_probs = tf.convert_to_tensor(init_probs, dtype=tf.float32)
        transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float32)

        t_len = tf.shape(log_emissions)[0]

        init_probs = init_probs / tf.maximum(tf.reduce_sum(init_probs), self.eps)
        emission0 = tf.exp(log_emissions[0])
        alpha0 = init_probs * emission0
        c0 = tf.maximum(tf.reduce_sum(alpha0), self.eps)
        xi0 = alpha0 / c0

        filtered_ta = tf.TensorArray(dtype=tf.float32, size=t_len, clear_after_read=False)
        scale_ta = tf.TensorArray(dtype=tf.float32, size=t_len, clear_after_read=False)
        filtered_ta = filtered_ta.write(0, xi0)
        scale_ta = scale_ta.write(0, c0)

        def body(t: tf.Tensor, prev_xi: tf.Tensor, filt_ta: tf.TensorArray, scl_ta: tf.TensorArray):
            """One filtering recursion updating the latent-state log weights."""
            pred = tf.linalg.matvec(transition_matrix, prev_xi, transpose_a=True)
            pred = tf.maximum(pred, self.eps)
            emission_t = tf.exp(log_emissions[t])
            alpha_t = pred * emission_t
            c_t = tf.maximum(tf.reduce_sum(alpha_t), self.eps)
            xi_t = alpha_t / c_t
            filt_ta = filt_ta.write(t, xi_t)
            scl_ta = scl_ta.write(t, c_t)
            return t + 1, xi_t, filt_ta, scl_ta

        _, _, filtered_ta, scale_ta = tf.while_loop(
            cond=lambda t, *_: t < t_len,
            body=body,
            loop_vars=(tf.constant(1, dtype=tf.int32), xi0, filtered_ta, scale_ta),
            parallel_iterations=1,
        )

        filtered = filtered_ta.stack()
        scales = scale_ta.stack()
        loglik = tf.reduce_sum(tf.math.log(tf.maximum(scales, self.eps)))
        return filtered, scales, loglik

    def run(
        self,
        log_emissions: tf.Tensor,
        init_probs: tf.Tensor,
        transition_matrix: tf.Tensor,
    ) -> ForwardFilterResult:
        """Run the scaled forward recursion.

        Args:
            log_emissions: Tensor with shape ``[T, Nz]``.
            init_probs: Initial latent-state probabilities with shape ``[Nz]``.
            transition_matrix: Markov transition matrix with shape ``[Nz, Nz]``.

        Returns:
            ``ForwardFilterResult`` with filtered probabilities, scaling factors,
            and the total log likelihood.
        """
        filtered, scales, loglik = self._run_tensors(
            log_emissions=log_emissions,
            init_probs=init_probs,
            transition_matrix=transition_matrix,
        )
        return ForwardFilterResult(
            filtered_probs=filtered,
            scale_factors=scales,
            log_likelihood=loglik,
        )


def stationary_distribution(transition_matrix: tf.Tensor) -> tf.Tensor:
    """Compute the stationary distribution of a Markov matrix.

    Args:
        transition_matrix: Square row-stochastic matrix with shape ``[Nz, Nz]``.

    Returns:
        Stationary probabilities with shape ``[Nz]``.
    """
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float32)
    eigvals, eigvecs = tf.linalg.eig(tf.transpose(transition_matrix))
    idx = tf.argmin(tf.abs(eigvals - tf.cast(1.0 + 0.0j, eigvals.dtype)))
    vec = tf.math.real(eigvecs[:, idx])
    vec = tf.maximum(vec, 0.0)
    total = tf.reduce_sum(vec)

    def _uniform() -> tf.Tensor:
        nz = tf.shape(transition_matrix)[0]
        return tf.ones((nz,), dtype=tf.float32) / tf.cast(nz, tf.float32)

    def _normalized() -> tf.Tensor:
        return vec / total

    return tf.cond(total <= 0.0, _uniform, _normalized)
