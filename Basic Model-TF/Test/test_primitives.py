# Test/test_primitives.py

import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams
from basic_mailer.primitives import beta_from_r, reward_basic, shock_next_z


def test_beta_from_r_basic():
    r = 0.04
    b = beta_from_r(r)
    assert abs(b - (1.0 / 1.04)) < 1e-12


def test_reward_basic_finite_and_shapes():
    mp = ModelParams()
    k = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    z = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    k_next = tf.constant([1.1, 2.1, 2.9], dtype=tf.float32)

    r = reward_basic(k, z, k_next, mp)
    assert r.shape == (3,)
    tf.debugging.assert_all_finite(r, "reward_basic returned non-finite values")


def test_reward_basic_increases_with_z_holding_other_fixed():
    mp = ModelParams()
    k = tf.constant([2.0], dtype=tf.float32)
    k_next = tf.constant([2.0], dtype=tf.float32)

    z1 = tf.constant([0.8], dtype=tf.float32)
    z2 = tf.constant([1.2], dtype=tf.float32)

    r1 = float(reward_basic(k, z1, k_next, mp).numpy()[0])
    r2 = float(reward_basic(k, z2, k_next, mp).numpy()[0])

    assert r2 > r1


def test_shock_next_z_positive_and_finite():
    mp = ModelParams()
    z = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32)

    z_next = shock_next_z(z, mp.rho, mp.sigma_eps)
    assert z_next.shape == (3,)
    tf.debugging.assert_all_finite(z_next, "shock_next_z returned non-finite values")
    assert tf.reduce_all(z_next > 0.0).numpy()
