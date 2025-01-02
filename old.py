import os
import time
import jax
import jax.numpy as np
from jax import random, grad, jit, vmap, value_and_grad, pmap
from jax.nn import log_softmax, softmax
from jax.example_libraries import optimizers
import neural_tangents as nt
from neural_tangents import stax
import mnist1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as onp
from functools import partial
import tensorflow as tf
from datetime import datetime
import json
from jax import pmap, vmap, local_device_count

# Configuration
EPOCHS = 1000
BATCH_SIZE = 1024
HIDDEN_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 
                220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
                450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
LEARNING_RATE = 1.0
TRIALS = 1
DEVICES = local_device_count()  # Get number of available GPUs

@jit
def loss(predictions, targets):
    """Compute cross-entropy loss."""
    return -np.mean(np.sum(targets * log_softmax(predictions), axis=1))

def make_batches(images, labels, batch_size):
    """Create batches from images and labels."""
    num_samples = images.shape[0]
    indices = onp.random.permutation(num_samples)
    for i in range(0, num_samples - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield images[batch_indices], labels[batch_indices]

@partial(jit, static_argnames=['opt_update', 'grad_loss'])
def train_step(params, opt_state, batch_images, batch_labels, opt_update, grad_loss):
    """Single training step, optimized for single GPU."""
    grads = grad_loss(params, batch_images, batch_labels)
    return opt_update(0, grads, opt_state)

def evaluate_accuracy(params, nn_apply, images, labels):
    """Calculate accuracy."""
    predictions = nn_apply(params, images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    return np.mean(predicted_classes == true_classes) * 100

@partial(jit, static_argnames=['nn_apply', 'f_lin'])
def compute_deltas(params, lin_params, images, nn_apply, f_lin):
    """Compute RMSE between neural network and its linearization."""
    nn_output = nn_apply(params, images)
    lin_output = f_lin(lin_params, images)
    return np.sqrt(np.mean((nn_output - lin_output) ** 2))

def evaluate_metrics(params, nn_apply, train_images, train_labels, test_images, test_labels):
    """Calculate all metrics for a given model."""
    # Training metrics
    train_output = nn_apply(params, train_images)
    train_loss = float(loss(train_output, train_labels))
    train_accuracy = float(evaluate_accuracy(params, nn_apply, train_images, train_labels))
    
    # Test metrics
    test_output = nn_apply(params, test_images)
    test_loss = float(loss(test_output, test_labels))
    test_accuracy = float(evaluate_accuracy(params, nn_apply, test_images, test_labels))
    
    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

def run_training_loop(num_epochs, opt_state, lin_opt_state, train_images, train_labels, 
                     test_images, test_labels, nn_apply, f_lin, opt_update, get_params, 
                     grad_loss, grad_lin_loss, save_path):
    """Run training loop and save results to JSON."""
    results = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Neural Network training step
        for batch_images, batch_labels in make_batches(train_images, train_labels, BATCH_SIZE):
            params = get_params(opt_state)
            opt_state = train_step(params, opt_state, batch_images, batch_labels, 
                                 opt_update, grad_loss)

        # Linearized model training step
        for batch_images, batch_labels in make_batches(train_images, train_labels, BATCH_SIZE):
            lin_params = get_params(lin_opt_state)
            lin_opt_state = train_step(lin_params, lin_opt_state, batch_images, batch_labels, 
                                     opt_update, grad_lin_loss)

        epoch_time = time.time() - start_time
        
        # Get final parameters for this epoch
        params = get_params(opt_state)
        lin_params = get_params(lin_opt_state)
        
        # Compute metrics for neural network
        nn_metrics = evaluate_metrics(params, nn_apply, 
                                    train_images, train_labels,
                                    test_images, test_labels)
        
        # Compute metrics for linearized model
        lin_metrics = evaluate_metrics(lin_params, f_lin,
                                     train_images, train_labels,
                                     test_images, test_labels)
        
        # Compute deltas
        delta_train = compute_deltas(params, lin_params, train_images, nn_apply, f_lin)
        delta_test = compute_deltas(params, lin_params, test_images, nn_apply, f_lin)
        
        # Store epoch results
        epoch_results = {
            "epoch": epoch,
            # Neural network metrics
            "train_loss": nn_metrics["train_loss"],
            "test_loss": nn_metrics["test_loss"],
            "train_accuracy": nn_metrics["train_accuracy"],
            "test_accuracy": nn_metrics["test_accuracy"],
            # Linearized model metrics
            "train_lin_loss": lin_metrics["train_loss"],
            "test_lin_loss": lin_metrics["test_loss"],
            "train_lin_accuracy": lin_metrics["train_accuracy"],
            "test_lin_accuracy": lin_metrics["test_accuracy"],
            # Deltas
            "delta_train": float(delta_train),
            "delta_test": float(delta_test),
            "epoch_time": epoch_time
        }
        results.append(epoch_results)
        
        # Save results after each epoch
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Epoch {epoch+1} | T: {epoch_time:0.2f} | "
              f"NN train/test: {nn_metrics['train_loss']:0.3f}/{nn_metrics['test_loss']:0.3f} | "
              f"Lin train/test: {lin_metrics['train_loss']:0.3f}/{lin_metrics['test_loss']:0.3f} | "
              f"NN acc: {nn_metrics['train_accuracy']:.2f}%/{nn_metrics['test_accuracy']:.2f}% | "
              f"Lin acc: {lin_metrics['train_accuracy']:.2f}%/{lin_metrics['test_accuracy']:.2f}%")
    
    return results

def load_fashion_mnist():
    """Load and preprocess Fashion MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Flatten and normalize
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Convert to one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    return np.array(x_train), y_train, np.array(x_test), y_test

def create_training_plots(results, hidden_size_dir):
    """Create comparison plots for all metrics."""
    epochs = [r['epoch'] for r in results]
    
    # Plot 1: Losses (Train and Test for both models)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['train_loss'] for r in results], 'b-', label='NN Train Loss')
    plt.plot(epochs, [r['test_loss'] for r in results], 'b--', label='NN Test Loss')
    plt.plot(epochs, [r['train_lin_loss'] for r in results], 'r-', label='Lin Train Loss')
    plt.plot(epochs, [r['test_lin_loss'] for r in results], 'r--', label='Lin Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(hidden_size_dir, 'losses.png'))
    plt.close()

    # Plot 2: Accuracies (Train and Test for both models)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['train_accuracy'] for r in results], 'b-', label='NN Train Acc')
    plt.plot(epochs, [r['test_accuracy'] for r in results], 'b--', label='NN Test Acc')
    plt.plot(epochs, [r['train_lin_accuracy'] for r in results], 'r-', label='Lin Train Acc')
    plt.plot(epochs, [r['test_lin_accuracy'] for r in results], 'r--', label='Lin Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(hidden_size_dir, 'accuracies.png'))
    plt.close()

    # Plot 3: Deltas
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['delta_train'] for r in results], 'g-', label='Train Delta')
    plt.plot(epochs, [r['delta_test'] for r in results], 'r-', label='Test Delta')
    plt.xlabel('Epoch')
    plt.ylabel('Delta (MSE)')
    plt.title('Network vs Linearization Deltas')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Often helpful for delta plots
    plt.tight_layout()
    plt.savefig(os.path.join(hidden_size_dir, 'deltas.png'))
    plt.close()

@partial(jit, static_argnames=['nn_apply'])
def grad_loss(params, x, y, nn_apply):
    return grad(lambda p: loss(vmap(nn_apply, in_axes=(None, 0))(p, x), y))(params)

@partial(jit, static_argnames=['f_lin'])
def grad_lin_loss(params, x, y, f_lin):
    return grad(lambda p: loss(f_lin(p, x), y))(params)

def main():
    # Create runs directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_run_dir = os.path.join('runs', f'run_{timestamp}')
    os.makedirs(base_run_dir, exist_ok=True)

    # Load Fashion MNIST data
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()

    for hidden_size in HIDDEN_SIZES:
        print(f"\nTraining with hidden size: {hidden_size}")
        
        # Create directory for this hidden size
        hidden_size_dir = os.path.join(base_run_dir, f'hidden_{hidden_size}')
        os.makedirs(hidden_size_dir, exist_ok=True)
        results_path = os.path.join(base_run_dir, f'hidden_{hidden_size}.json')

        # Initialize model with correct input shape (784 for Fashion MNIST)
        nn_init, nn_apply, _ = stax.serial(
            stax.Dense(hidden_size),
            stax.Relu(),
            stax.Dense(10)
        )

        # Initialize optimizer
        opt_init, opt_update, get_params = optimizers.momentum(LEARNING_RATE, 0.9)
        
        # Initialize parameters with correct input shape
        rng = random.PRNGKey(111)
        output_shape, params_init = nn_init(rng, input_shape=(-1, 784))
        
        # Create linearized function
        f_lin = nt.linearize(nn_apply, params_init)

        # Initialize optimizer states
        opt_state = opt_init(params_init)
        lin_opt_state = opt_init(params_init)

        # Training
        results = run_training_loop(
            EPOCHS, opt_state, lin_opt_state, 
            train_images, train_labels, 
            test_images, test_labels,
            nn_apply, f_lin, opt_update, get_params, 
            lambda p, x, y: grad_loss(p, x, y, nn_apply),  # Pass nn_apply here
            lambda p, x, y: grad_lin_loss(p, x, y, f_lin),  # Pass f_lin here
            results_path
        )

        # Create plots after training
        create_training_plots(results, hidden_size_dir)

if __name__ == "__main__":
    main()