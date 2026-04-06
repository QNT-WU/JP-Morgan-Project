"""Smoke tests for objective-loss implementations."""

# Test/test_objectives_smoke.py
import numpy as np
import tensorflow as tf

from risky_debt.config import (
    ModelParams,
    NetParams,
    TrainParams,
    Obj3Params,
    Obj1Params,
    Obj2Params,
)
from risky_debt.networks import PolicyNet, ValueNet, VtildeNet, PricingNet
from risky_debt.objectives import obj1_loss, obj2_batch_loss, obj3_batch_loss


def _tiny_train_params():
    # Use only the args that matter for objectives; rely on defaults for the rest.
    """Build a tiny training-parameter configuration for smoke tests."""
    return TrainParams(
        seed=1,
        T_train=10,
        N_paths_train=8,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )


def test_obj1_loss_finite_and_gradients():
    """Verify obj1 loss finite and gradients."""
    mp = ModelParams()
    tp = _tiny_train_params()
    op1 = Obj1Params(nu_zp=1.0)

    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    pol = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    # build
    _ = pol(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    with tf.GradientTape(persistent=True) as tape:
        loss, train_reward, zp_loss = obj1_loss(pol, qnet, mp, tp, op1)

    grads_p = tape.gradient(loss, pol.trainable_variables)
    grads_q = tape.gradient(loss, qnet.trainable_variables)
    del tape

    tf.debugging.assert_all_finite(loss, "Obj1 loss is non-finite")
    tf.debugging.assert_all_finite(train_reward, "Obj1 train_reward is non-finite")
    tf.debugging.assert_all_finite(zp_loss, "Obj1 zp_loss is non-finite")

    assert any(g is not None for g in grads_p)
    assert any(g is not None for g in grads_q)


def test_obj2_loss_finite_and_gradients():
    """Verify obj2 loss finite and gradients."""
    mp = ModelParams()
    tp = _tiny_train_params()
    op2 = Obj2Params(nu_def=1.0, nu_bell=1.0, nu_foc=1.0, nu_zp=1.0)

    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nval = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nvt = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    pol = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
    val = ValueNet(nval)
    vtilde = VtildeNet(nvt)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    _ = pol(tf.zeros((1, 3), dtype=tf.float32))
    _ = val(tf.zeros((1, 3), dtype=tf.float32))
    _ = vtilde(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    k = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)
    b = tf.constant(np.random.uniform(-0.3, 0.3, size=(32,)), dtype=tf.float32)
    z = tf.constant(np.random.uniform(0.7, 1.3, size=(32,)), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        loss = obj2_batch_loss(pol, val, vtilde, qnet, mp, tp, op2, k, b, z)

    grads_p = tape.gradient(loss, pol.trainable_variables)
    grads_v = tape.gradient(loss, val.trainable_variables)
    grads_vt = tape.gradient(loss, vtilde.trainable_variables)
    grads_q = tape.gradient(loss, qnet.trainable_variables)
    del tape

    tf.debugging.assert_all_finite(loss, "Obj2 loss is non-finite")

    assert any(g is not None for g in grads_p)
    assert any(g is not None for g in grads_v)
    assert any(g is not None for g in grads_vt)
    assert any(g is not None for g in grads_q)


def test_obj3_loss_finite_and_gradients():
    """Verify obj3 loss finite and gradients."""
    mp = ModelParams()
    tp = _tiny_train_params()
    op3 = Obj3Params(nu_def=1.0, nu_zp=1.0)

    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nval = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nq = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")

    pol = PolicyNet(npol, mp.k_min, mp.b_min, mp.b_max)
    val = ValueNet(nval)
    qnet = PricingNet(nq, mp.q_min, mp.q_max)

    _ = pol(tf.zeros((1, 3), dtype=tf.float32))
    _ = val(tf.zeros((1, 3), dtype=tf.float32))
    _ = qnet(tf.zeros((1, 3), dtype=tf.float32))

    k = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)
    b = tf.constant(np.random.uniform(-0.3, 0.3, size=(32,)), dtype=tf.float32)
    z = tf.constant(np.random.uniform(0.7, 1.3, size=(32,)), dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        loss = obj3_batch_loss(pol, val, qnet, mp, tp, op3, k, b, z)

    grads_p = tape.gradient(loss, pol.trainable_variables)
    grads_v = tape.gradient(loss, val.trainable_variables)
    grads_q = tape.gradient(loss, qnet.trainable_variables)
    del tape

    tf.debugging.assert_all_finite(loss, "Obj3 loss is non-finite")
    assert any(g is not None for g in grads_p)
    assert any(g is not None for g in grads_v)
    assert any(g is not None for g in grads_q)
