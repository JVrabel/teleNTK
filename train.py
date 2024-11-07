"""
Main training script for comparing finite networks and NTK training
"""
import os
import json
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import time
from datetime import datetime

from src.data_setup import create_data_iterators
from src.model_builder import create_model
from src.engine import train_finite_network, train_ntk

# Set training hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 50  # Adjust based on dataset size
EVAL_STEPS = 10
LEARNING_RATE = 0.1
WIDTH = 32

# Create experiment directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join("experiments", timestamp)
os.makedirs(experiment_dir, exist_ok=True)

# Save hyperparameters
hyperparams = {
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "eval_steps": EVAL_STEPS,
    "learning_rate": LEARNING_RATE,
    "width": WIDTH
}

with open(os.path.join(experiment_dir, "hyperparams.json"), "w") as f:
    json.dump(hyperparams, f, indent=2)

# Create data iterators
train_iterator, test_iterator, num_classes = create_data_iterators(
    batch_size=BATCH_SIZE
)

# Initialize model for finite network training
print("Initializing finite network...")
apply_fn_finite, params = create_model(
    input_shape=(28, 28, 1),
    width=WIDTH,
    num_classes=num_classes,
    mode='finite'
)

# Initialize optimizer
optimizer = optimizers.adam(LEARNING_RATE)

# Train finite network
print("\nTraining finite network...")
finite_history = train_finite_network(
    params=params,
    optimizer=optimizer,
    train_iterator=train_iterator,
    val_iterator=test_iterator,
    apply_fn=apply_fn_finite,
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    eval_steps=EVAL_STEPS
)

# Save finite network results
finite_results = {
    "train_loss": [float(x) for x in finite_history.train_loss],
    "train_acc": [float(x) for x in finite_history.train_acc],
    "val_loss": [float(x) for x in finite_history.val_loss],
    "val_acc": [float(x) for x in finite_history.val_acc],
    "epoch_times": finite_history.epoch_times
}

with open(os.path.join(experiment_dir, "finite_results.json"), "w") as f:
    json.dump(finite_results, f, indent=2)

# Initialize model for NTK training
print("\nInitializing NTK...")
kernel_fn, _ = create_model(
    input_shape=(28, 28, 1),
    width=WIDTH,
    num_classes=num_classes,
    mode='ntk'
)

# Train NTK
print("\nTraining NTK...")
ntk_history = train_ntk(
    kernel_fn=kernel_fn,
    train_iterator=train_iterator,
    val_iterator=test_iterator,
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    eval_steps=EVAL_STEPS,
    learning_rate=LEARNING_RATE
)

# Save NTK results
ntk_results = {
    "train_loss": [float(x) for x in ntk_history.train_loss],
    "train_acc": [float(x) for x in ntk_history.train_acc],
    "val_loss": [float(x) for x in ntk_history.val_loss],
    "val_acc": [float(x) for x in ntk_history.val_acc],
    "epoch_times": ntk_history.epoch_times
}

with open(os.path.join(experiment_dir, "ntk_results.json"), "w") as f:
    json.dump(ntk_results, f, indent=2)

print(f"\nExperiment results saved to: {experiment_dir}") 