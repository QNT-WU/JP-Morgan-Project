"""Smoke tests for TensorFlow objective losses."""

## Test/test_objectives_smoke.py
# “Smoke test” means: does it even run and backprop?
# It verifies two things for each objective loss:
# The loss is finite (no NaN/Inf)
# The loss produces gradients w.r.t. the model parameters(at least one gradient tensor is not None)
import numpy as np
import tensorflow as tf

from basic_mailer.config import ModelParams, NetParams, TrainParams, Obj3Params
from basic_mailer.networks import PolicyNet, ValueNet
from basic_mailer.objectives import obj1_loss, obj2_batch_loss, obj3_batch_loss


def test_obj1_loss_finite_and_gradients():
    """Verify Objective 1 produces a finite loss and gradients."""
    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    # small network: 1 hidden layer, 8 units → fast test
    tp = TrainParams(
        seed=1,
        T_train=10,
        N_paths_train=8,
        epochs=1,
        steps_per_epoch=1,
        batch_size=16,
    )
    # Creates policy network to differentiate through.
    pol = PolicyNet(npol, mp.k_min, mp.k_max)

    _ = pol(tf.zeros((1, 2), dtype=tf.float32))

    # Compute loss under GradientTape
    with tf.GradientTape() as tape:
        loss, train_reward = obj1_loss(pol, mp, tp)
    # Compute gradients
    # This asks TF:
    # “Give me ∂loss/∂θ for each trainable variable θ in PolicyNet.”
    # If the network has no variables yet (common with subclassed models before first call/build),
    # then pol.trainable_variables can be empty or not connected, and gradients will be None.
    grads = tape.gradient(loss, pol.trainable_variables)

    # Finite checks
    # Ensures no NaN/Inf.
    tf.debugging.assert_all_finite(loss, "Obj1 loss is non-finite")
    tf.debugging.assert_all_finite(train_reward, "Obj1 train_reward is non-finite")
    # Gradient existence check
    # Intended meaning: “Obj1 loss is differentiable and connected to policy parameters.”
    assert any(g is not None for g in grads)


def test_obj2_loss_finite_and_gradients():
    """Verify Objective 2 produces a finite loss and gradients."""
    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    pol = PolicyNet(npol, mp.k_min, mp.k_max)

    _ = pol(tf.zeros((1, 2), dtype=tf.float32))

    k = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)
    z = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)

    # Obj2 loss is
    # L2​=E[f(ε1​)f(ε2​)]
    # where f calls policy twice (k1 = φ(k,z), k2 = φ(k1,z'))
    # This should absolutely depend on policy weights → should give gradients
    with tf.GradientTape() as tape:
        loss = obj2_batch_loss(pol, mp, k, z)
    grads = tape.gradient(loss, pol.trainable_variables)

    tf.debugging.assert_all_finite(loss, "Obj2 loss is non-finite")
    assert any(g is not None for g in grads)


def test_obj3_loss_finite_and_gradients():
    """Verify Objective 3 produces a finite loss and gradients."""
    mp = ModelParams()
    npol = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    nval = NetParams(hidden_units=8, hidden_layers=1, activation="tanh")
    pol = PolicyNet(npol, mp.k_min, mp.k_max)
    val = ValueNet(nval)

    _ = pol(tf.zeros((1, 2), dtype=tf.float32))
    _ = val(tf.zeros((1, 2), dtype=tf.float32))

    op3 = Obj3Params(nu=1.0)

    k = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)
    z = tf.constant(np.random.uniform(0.5, 2.0, size=(32,)), dtype=tf.float32)

    # Persistent tape
    # Because you will request gradients twice:
    # once wrt policy variables
    # once wrt value variables
    # A non-persistent tape can only be used once
    with tf.GradientTape(persistent=True) as tape:
        loss = obj3_batch_loss(pol, val, mp, op3, k, z)

    grads_p = tape.gradient(loss, pol.trainable_variables)
    grads_v = tape.gradient(loss, val.trainable_variables)
    del tape

    tf.debugging.assert_all_finite(loss, "Obj3 loss is non-finite")
    assert any(g is not None for g in grads_p)
    assert any(g is not None for g in grads_v)
