"""
Contains training loops for both finite width networks and NTK training
"""
from typing import Dict, Tuple, Callable, Iterator, Any
import jax.numpy as jnp
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

def train_finite_network(
    params: Dict,
    optimizer: Any,
    train_iterator: Iterator,
    val_iterator: Iterator,
    apply_fn: Callable,
    num_epochs: int,
    steps_per_epoch: int,
    eval_steps: int
) -> TrainingHistory:
    """Training loop for finite width network."""
    history = TrainingHistory()
    opt_state = optimizer.init(params)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss = 0.0
        train_acc = 0.0
        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = next(train_iterator)
            params, opt_state, loss, acc = finite_train_step(
                params, opt_state, batch, apply_fn, optimizer
            )
            train_loss += loss
            train_acc += acc
            
        train_loss /= steps_per_epoch
        train_acc /= steps_per_epoch
        
        # Validation
        val_loss = 0.0
        val_acc = 0.0
        for _ in range(eval_steps):
            batch = next(val_iterator)
            loss, acc = eval_step(params, batch, apply_fn)
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