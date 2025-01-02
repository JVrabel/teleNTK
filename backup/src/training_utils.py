"""
Contains training utilities for both finite network and NTK training
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Dict, Any
from functools import partial
from neural_tangents import predict

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute cross entropy loss."""
    log_softmax = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(log_softmax * labels, axis=-1))

def accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute accuracy."""
    predicted_class = jnp.argmax(logits, axis=-1)
    true_class = jnp.argmax(labels, axis=-1)
    return jnp.mean(predicted_class == true_class)

@jax.jit
def finite_train_step(
    params: Dict,
    opt_state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    apply_fn: Callable,
    optimizer: Any
) -> Tuple[Dict, Any, jnp.ndarray, jnp.ndarray]:
    """Perform one training step for finite width network."""
    inputs, labels = batch
    
    def loss_fn(params):
        logits = apply_fn(params, inputs)
        return cross_entropy_loss(logits, labels)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = jax.tree_map(lambda p, u: p + u, params, updates)
    
    # Compute accuracy
    logits = apply_fn(params, inputs)
    acc = accuracy(logits, labels)
    
    return params, opt_state, loss, acc

@partial(jax.jit, static_argnums=(2,))
def ntk_train_step(
    kernel_fn: Callable,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    learning_rate: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Perform one training step using NTK dynamics."""
    inputs, labels = batch
    
    # Compute NTK prediction
    predictions = predict.gradient_descent_mse(
        kernel_fn=kernel_fn,
        train_data=inputs,
        train_labels=labels,
        learning_rate=learning_rate,
        steps=1
    )
    
    # Compute metrics
    loss = cross_entropy_loss(predictions, labels)
    acc = accuracy(predictions, labels)
    
    return loss, acc

@jax.jit
def eval_step(
    params: Dict,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    apply_fn: Callable
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluation step for finite width network."""
    inputs, labels = batch
    logits = apply_fn(params, inputs)
    loss = cross_entropy_loss(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc 