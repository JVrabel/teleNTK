import os
import time
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, value_and_grad, jacfwd, jacrev
from jax.nn import log_softmax, softmax
from jax.example_libraries import optimizers
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt
import numpy as onp
from functools import partial
import tensorflow as tf
from datetime import datetime
import json

# Configuration
EPOCHS = 1000
BATCH_SIZE = 1024
HIDDEN_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 
                220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
                450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
LEARNING_RATE = 1.0
SCALE = 2.0

def create_networks(key, hidden_size, scale):
    """Create all three networks: standard, transformed, and NTK."""
    key1, key2 = random.split(key)
    
    # Standard network (vanilla initialization)
    nn_init, nn_apply, _ = stax.serial(
        stax.Dense(hidden_size),  # Vanilla stax.Dense without bias
        stax.Relu(),
        stax.Dense(10)
    )
    
    # Initialize parameters
    _, std_params = nn_init(key1, (-1, 784))  # Standard network params
    _, ntk_params = nn_init(key2, (-1, 784))  # Separate NTK params
    
    # Create transformed parameters from standard params
    W1, _ = std_params[0]  # Only weights, no biases
    W2, _ = std_params[2]  # Only weights, no biases
    
    # Scale both layers down by scale
    W1_scaled = W1 / scale
    W2_scaled = W2 / scale
    
    # Create transformed parameters
    trans_params = [(W1_scaled, None), (), (W2_scaled, None)]
    
    # Create NTK function
    f_ntk = nt.linearize(nn_apply, ntk_params)
    
    return std_params, trans_params, ntk_params, nn_apply, f_ntk

@jit
def loss(predictions, targets):
    """Compute cross-entropy loss."""
    return -jnp.mean(jnp.sum(targets * log_softmax(predictions), axis=1))

def make_batches(images, labels, batch_size):
    """Create batches from images and labels."""
    num_samples = images.shape[0]
    indices = onp.random.permutation(num_samples)
    for i in range(0, num_samples - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield images[batch_indices], labels[batch_indices]

@partial(jit, static_argnames=['nn_apply'])
def forward_pass(params, images, nn_apply, scale=None):
    """Forward pass with optional scaling for transformed network."""
    out = vmap(nn_apply, in_axes=(None, 0))(params, images)
    if scale is not None:
        out = out * scale * scale  # Multiply by scale^2 to compensate
    return out

@partial(jit, static_argnames=['apply_fn'])
def compute_gradients(params, batch_images, batch_labels, apply_fn, scale=None):
    """Compute gradients for a batch."""
    def loss_fn(p):
        predictions = forward_pass(p, batch_images, apply_fn, scale)
        return loss(predictions, batch_labels)
    return grad(loss_fn)(params)

def verify_initialization(std_params, trans_params, apply_fn, test_batch):
    """Verify that networks produce identical outputs at initialization."""
    std_out = forward_pass(std_params, test_batch, apply_fn)
    trans_out = forward_pass(trans_params, test_batch, apply_fn, SCALE)
    
    max_diff = jnp.max(jnp.abs(std_out - trans_out))
    print("\nInitialization check:")
    print(f"Standard output: {std_out[0][:3]}...")
    print(f"Transformed output: {trans_out[0][:3]}...")
    print(f"Maximum difference: {max_diff}")
    
    assert max_diff < 1e-6, f"Networks not initialized correctly! Max difference: {max_diff}"
    return max_diff

def compute_hessian(params, apply_fn, batch_x, batch_y, scale=None, max_batch=32):
    """Compute Hessian for a small batch to avoid memory issues."""
    batch_x = batch_x[:max_batch]
    batch_y = batch_y[:max_batch]
    
    def loss_fn(p):
        pred = forward_pass(p, batch_x, apply_fn, scale)
        return loss(pred, batch_y)
    
    W1, _ = params[0]
    reduced_params = W1[:10, :10]
    
    H = jacfwd(jacrev(lambda p: loss_fn([(p, params[0][1]), params[1], params[2]])))(reduced_params)
    H = (H + H.T) / 2
    return H

def analyze_hessian(H):
    """Analyze Hessian properties."""
    flat_H = jnp.concatenate([jnp.ravel(x) for x in jax.tree_leaves(H)])
    flat_H = flat_H.reshape(-1, flat_H.shape[0])
    eigvals = jnp.linalg.eigvalsh(flat_H)
    
    return {
        "max_eigenval": float(jnp.max(eigvals)),
        "min_eigenval": float(jnp.min(eigvals)),
        "condition_number": float(jnp.max(jnp.abs(eigvals)) / jnp.min(jnp.abs(eigvals[eigvals != 0]))),
        "trace": float(jnp.trace(flat_H))
    }

def evaluate_accuracy(params, apply_fn, images, labels, scale=None):
    """Calculate accuracy."""
    predictions = forward_pass(params, images, apply_fn, scale)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    return jnp.mean(predicted_classes == true_classes) * 100

def evaluate_metrics(params, apply_fn, images, labels, scale=None):
    """Calculate all metrics for a given model."""
    predictions = forward_pass(params, images, apply_fn, scale)
    current_loss = float(loss(predictions, labels))
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracy = float(jnp.mean(predicted_classes == true_classes) * 100)
    
    return {
        "train_loss": current_loss,
        "test_loss": current_loss,
        "train_accuracy": accuracy,
        "test_accuracy": accuracy
    }

def compute_deltas(params, ntk_params, images, apply_fn, f_ntk, scale=None):
    """Compute RMSE between neural network and its linearization."""
    nn_output = forward_pass(params, images, apply_fn, scale)
    ntk_output = f_ntk(ntk_params, images)
    return jnp.sqrt(jnp.mean((nn_output - ntk_output) ** 2))

def run_training_loop(num_epochs, std_opt_state, trans_opt_state, ntk_opt_state,
                     train_images, train_labels, test_images, test_labels,
                     nn_apply, f_ntk, opt_update, get_params, scale, save_path):
    """Run training loop for all three models."""
    results = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train standard network
        for std_batch_images, std_batch_labels in make_batches(train_images, train_labels, BATCH_SIZE):
            std_params = get_params(std_opt_state)
            std_grads = compute_gradients(std_params, std_batch_images, std_batch_labels, nn_apply)
            std_opt_state = opt_update(0, std_grads, std_opt_state)
        
        # Train transformed network
        for trans_batch_images, trans_batch_labels in make_batches(train_images, train_labels, BATCH_SIZE):
            trans_params = get_params(trans_opt_state)
            trans_grads = compute_gradients(trans_params, trans_batch_images, trans_batch_labels, nn_apply, scale)
            trans_opt_state = opt_update(0, trans_grads, trans_opt_state)
        
        # Train NTK
        for ntk_batch_images, ntk_batch_labels in make_batches(train_images, train_labels, BATCH_SIZE):
            ntk_params = get_params(ntk_opt_state)
            ntk_grads = compute_gradients(ntk_params, ntk_batch_images, ntk_batch_labels, f_ntk)
            ntk_opt_state = opt_update(0, ntk_grads, ntk_opt_state)
        
        # Get current parameters for evaluation
        final_std_params = get_params(std_opt_state)
        final_trans_params = get_params(trans_opt_state)
        final_ntk_params = get_params(ntk_opt_state)
        
        # Compute metrics
        metrics = {
            "epoch": epoch,
            # Standard network metrics
            "train_loss": float(evaluate_metrics(final_std_params, nn_apply, train_images, train_labels)["train_loss"]),
            "test_loss": float(evaluate_metrics(final_std_params, nn_apply, test_images, test_labels)["test_loss"]),
            "train_accuracy": float(evaluate_metrics(final_std_params, nn_apply, train_images, train_labels)["train_accuracy"]),
            "test_accuracy": float(evaluate_metrics(final_std_params, nn_apply, test_images, test_labels)["test_accuracy"]),
            # NTK metrics
            "train_lin_loss": float(evaluate_metrics(final_ntk_params, f_ntk, train_images, train_labels)["train_loss"]),
            "test_lin_loss": float(evaluate_metrics(final_ntk_params, f_ntk, test_images, test_labels)["test_loss"]),
            "train_lin_accuracy": float(evaluate_metrics(final_ntk_params, f_ntk, train_images, train_labels)["train_accuracy"]),
            "test_lin_accuracy": float(evaluate_metrics(final_ntk_params, f_ntk, test_images, test_labels)["test_accuracy"]),
            # Transformed network metrics
            "train_trans_loss": float(evaluate_metrics(final_trans_params, nn_apply, train_images, train_labels, scale)["train_loss"]),
            "test_trans_loss": float(evaluate_metrics(final_trans_params, nn_apply, test_images, test_labels, scale)["test_loss"]),
            "train_trans_accuracy": float(evaluate_metrics(final_trans_params, nn_apply, train_images, train_labels, scale)["train_accuracy"]),
            "test_trans_accuracy": float(evaluate_metrics(final_trans_params, nn_apply, test_images, test_labels, scale)["test_accuracy"]),
            # Deltas
            "delta_train": float(compute_deltas(final_std_params, final_ntk_params, train_images, nn_apply, f_ntk)),
            "delta_test": float(compute_deltas(final_std_params, final_ntk_params, test_images, nn_apply, f_ntk)),
            "delta_trans_train": float(compute_deltas(final_trans_params, final_ntk_params, train_images, nn_apply, f_ntk, scale)),
            "delta_trans_test": float(compute_deltas(final_trans_params, final_ntk_params, test_images, nn_apply, f_ntk, scale)),
            "epoch_time": time.time() - start_time
        }
        
        results.append(metrics)
        
        # Save results after each epoch
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | "
                  f"Std acc: {metrics['train_accuracy']:.2f}/{metrics['test_accuracy']:.2f} | "
                  f"Trans acc: {metrics['train_trans_accuracy']:.2f}/{metrics['test_trans_accuracy']:.2f} | "
                  f"NTK acc: {metrics['train_lin_accuracy']:.2f}/{metrics['test_lin_accuracy']:.2f}")
    
    return results

def load_fashion_mnist():
    """Load and preprocess Fashion MNIST dataset."""
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    x_train = jnp.array(x_train.reshape(-1, 784).astype('float32') / 255.0)
    x_test = jnp.array(x_test.reshape(-1, 784).astype('float32') / 255.0)
    
    y_train = jnp.array(jnp.eye(10)[y_train])
    y_test = jnp.array(jnp.eye(10)[y_test])
    
    print(f"Loaded {x_train.shape[0]} training samples and {x_test.shape[0]} test samples")
    print(f"Data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"            x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test

def create_training_plots(results, hidden_size_dir):
    """Create comparison plots for all metrics."""
    epochs = [r['epoch'] for r in results]
    
    # Accuracies plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['train_accuracy'] for r in results], 'b-', label='Standard Train')
    plt.plot(epochs, [r['test_accuracy'] for r in results], 'b--', label='Standard Test')
    plt.plot(epochs, [r['train_trans_accuracy'] for r in results], 'r-', label='Transformed Train')
    plt.plot(epochs, [r['test_trans_accuracy'] for r in results], 'r--', label='Transformed Test')
    plt.plot(epochs, [r['train_lin_accuracy'] for r in results], 'g-', label='NTK Train')
    plt.plot(epochs, [r['test_lin_accuracy'] for r in results], 'g--', label='NTK Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(hidden_size_dir, 'accuracies.png'))
    plt.close()

    # Losses plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['train_loss'] for r in results], 'b-', label='Standard Train')
    plt.plot(epochs, [r['test_loss'] for r in results], 'b--', label='Standard Test')
    plt.plot(epochs, [r['train_trans_loss'] for r in results], 'r-', label='Transformed Train')
    plt.plot(epochs, [r['test_trans_loss'] for r in results], 'r--', label='Transformed Test')
    plt.plot(epochs, [r['train_lin_loss'] for r in results], 'g-', label='NTK Train')
    plt.plot(epochs, [r['test_lin_loss'] for r in results], 'g--', label='NTK Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(hidden_size_dir, 'losses.png'))
    plt.close()

    # Deltas plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [r['delta_train'] for r in results], 'b-', label='Standard Train Delta')
    plt.plot(epochs, [r['delta_test'] for r in results], 'b--', label='Standard Test Delta')
    plt.plot(epochs, [r['delta_trans_train'] for r in results], 'r-', label='Transformed Train Delta')
    plt.plot(epochs, [r['delta_trans_test'] for r in results], 'r--', label='Transformed Test Delta')
    plt.xlabel('Epoch')
    plt.ylabel('Delta (RMSE)')
    plt.title('Network vs Linearization Deltas')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(hidden_size_dir, 'deltas.png'))
    plt.close()

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'runs/run_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    
    train_images, train_labels, test_images, test_labels = load_fashion_mnist()
    
    for hidden_size in HIDDEN_SIZES:
        print(f"\nTraining networks with hidden size: {hidden_size}")
        hidden_size_dir = os.path.join(base_dir, f'hidden_{hidden_size}')
        os.makedirs(hidden_size_dir, exist_ok=True)
        
        key = random.PRNGKey(0)
        std_params, trans_params, ntk_params, nn_apply, f_ntk = create_networks(key, hidden_size, SCALE)
        
        init_diff = verify_initialization(std_params, trans_params, nn_apply, train_images[:10])
        print(f"Initialization difference: {init_diff}")
        
        opt_init, opt_update, get_params = optimizers.momentum(LEARNING_RATE, 0.9)
        
        std_opt_state = opt_init(std_params)
        trans_opt_state = opt_init(trans_params)
        ntk_opt_state = opt_init(ntk_params)
        
        results = run_training_loop(
            EPOCHS, 
            std_opt_state,
            trans_opt_state,
            ntk_opt_state,
            train_images,
            train_labels,
            test_images,
            test_labels,
            nn_apply,
            f_ntk,
            opt_update,
            get_params,
            SCALE,
            os.path.join(hidden_size_dir, 'results.json')
        )
        
        with open(os.path.join(hidden_size_dir, 'results.json'), 'w') as f:
            json.dump({
                'training_results': results
            }, f, indent=2)
        
        create_training_plots(results, hidden_size_dir)

if __name__ == "__main__":
    main()
