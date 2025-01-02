"""
Contains training loops for both finite width networks and NTK training
"""
from typing import Dict, Tuple, Callable, Iterator, Any
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
import time
from .training_utils import finite_train_step, ntk_train_step, eval_step

class TrainingHistory:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.epoch_times = []

def finite_train_step(params, opt_state, batch, apply_fn, optimizer):
    """Training step for finite width network."""
    @jax.jit(static_argnums=(3, 4))  # Move decorator here
    def train_step(params, opt_state, batch, apply_fn, optimizer):
        def loss_fn(params, batch):
            inputs, targets = batch
            logits = apply_fn(params, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        logits = apply_fn(params, batch[0])
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch[1])
        
        return params, opt_state, loss, acc
    
    return train_step(params, opt_state, batch, apply_fn, optimizer)

def train_finite_network(
    params,
    optimizer,
    opt_state,
    train_iterator,
    val_iterator,
    apply_fn,
    num_epochs,
    steps_per_epoch,
    eval_steps
):
    history = TrainingHistory()
    
    # Define training step without decorator first
    def train_step_fn(params, opt_state, batch, apply_fn, optimizer):
        def loss_fn(params, batch):
            inputs, targets = batch
            logits = apply_fn(params, inputs)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        logits = apply_fn(new_params, batch[0])
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch[1])
        
        return new_params, new_opt_state, loss, acc

    # JIT compile the function
    train_step = jax.jit(train_step_fn, static_argnums=(3, 4))

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        epoch_loss = []
        epoch_acc = []
        
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = next(train_iterator)
            params, opt_state, loss, acc = train_step(params, opt_state, batch, apply_fn, optimizer)
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            
        history.train_loss.append(float(jnp.mean(jnp.array(epoch_loss))))
        history.train_acc.append(float(jnp.mean(jnp.array(epoch_acc))))
        history.epoch_times.append(time.time() - start_time)
        
        # Validation
        val_loss = []
        val_acc = []
        
        for _ in range(eval_steps):
            batch = next(val_iterator)
            loss, acc = eval_step(params, batch, apply_fn)
            val_loss.append(loss)
            val_acc.append(acc)
            
        history.val_loss.append(float(jnp.mean(jnp.array(val_loss))))
        history.val_acc.append(float(jnp.mean(jnp.array(val_acc))))
        
    return history

def train_ntk(
    kernel_fn: Callable,
    train_iterator: Iterator,
    val_iterator: Iterator,
    num_epochs: int,
    steps_per_epoch: int,
    eval_steps: int,
    learning_rate: float
) -> TrainingHistory:
    """Training loop for NTK."""
    history = TrainingHistory()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss = 0.0
        train_acc = 0.0
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = next(train_iterator)
            loss, acc = ntk_train_step(kernel_fn, batch, learning_rate)
            train_loss += loss
            train_acc += acc
            
        train_loss /= steps_per_epoch
        train_acc /= steps_per_epoch
        
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        for _ in range(eval_steps):
            batch = next(val_iterator)
            loss, acc = ntk_train_step(kernel_fn, batch, learning_rate)
            val_loss += loss
            val_acc += acc
            
        val_loss /= eval_steps
        val_acc /= eval_steps
        
        epoch_time = time.time() - start_time
        
        # Update history
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
    return history 