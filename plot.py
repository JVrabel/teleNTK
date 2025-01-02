import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def load_results(run_folder, epoch):
    """
    Load results for all hidden sizes from a specific run folder.
    epoch: which epoch to extract (-1 for last epoch)
    """
    metrics_by_size = {}
    
    # Find all JSON files in the run folder
    for filename in os.listdir(run_folder):
        if filename.endswith('.json'):
            hidden_size = int(filename.split('_')[1].split('.')[0])  # Extract size from "hidden_100.json"
            
            with open(os.path.join(run_folder, filename), 'r') as f:
                results = json.load(f)
                metrics_by_size[hidden_size] = results[epoch]
    
    return metrics_by_size

def power_law(x, a, b):
    """Power law function: y = ax^b"""
    return a * np.power(x, b)

def fit_and_analyze_delta(hidden_sizes, deltas, ax, label, plot_references=False):
    """Fit different functions to delta and return the best fit"""
    x_data = np.array(hidden_sizes)
    y_data = np.array(deltas)
    
    # Skip first P% of data points for fitting
    P = 0
    cutoff_idx = int(len(x_data) * P)
    x_fit = x_data[cutoff_idx:]
    y_fit = y_data[cutoff_idx:]
    
    # Fit power law using later (1-P)*100% of data
    popt_power, _ = curve_fit(power_law, x_fit, y_fit, p0=[1, -1])
    
    # Calculate R-squared for power law fit
    y_power = power_law(x_fit, *popt_power)
    r2_power = 1 - np.sum((y_fit - y_power)**2) / np.sum((y_fit - np.mean(y_fit))**2)
    
    # Plot fits
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    ax.plot(x_smooth, power_law(x_smooth, *popt_power), '--', 
            label=f'{label} power law: {popt_power[0]:.2e}x^{popt_power[1]:.2f} (R²={r2_power:.3f})')
    
    # Plot reference curves only once
    if plot_references:
        # Add reference 1/x curve
        a_ref = np.mean(y_fit * x_fit)
        ax.plot(x_smooth, a_ref/x_smooth, ':', color='red', 
                label=f'Reference 1/x: {a_ref:.2e}/x')
        
        # Add reference 1/sqrt(x) curve
        a_sqrt = np.mean(y_fit * np.sqrt(x_fit))
        ax.plot(x_smooth, a_sqrt/np.sqrt(x_smooth), '-.', color='blue', 
                label=f'Reference 1/√x: {a_sqrt:.2e}/√x')
    
    return popt_power, r2_power

def create_comparison_plots(metrics_by_size, save_path=None):
    """Create comparison plots for all metrics across hidden sizes."""
    hidden_sizes = sorted(metrics_by_size.keys())
    
    # Extract metrics
    nn_train_loss = [metrics_by_size[size]['train_loss'] for size in hidden_sizes]
    nn_test_loss = [metrics_by_size[size]['test_loss'] for size in hidden_sizes]
    lin_train_loss = [metrics_by_size[size]['train_lin_loss'] for size in hidden_sizes]
    lin_test_loss = [metrics_by_size[size]['test_lin_loss'] for size in hidden_sizes]
    
    nn_train_acc = [metrics_by_size[size]['train_accuracy'] for size in hidden_sizes]
    nn_test_acc = [metrics_by_size[size]['test_accuracy'] for size in hidden_sizes]
    lin_train_acc = [metrics_by_size[size]['train_lin_accuracy'] for size in hidden_sizes]
    lin_test_acc = [metrics_by_size[size]['test_lin_accuracy'] for size in hidden_sizes]
    
    delta_train = [metrics_by_size[size]['delta_train'] for size in hidden_sizes]
    delta_test = [metrics_by_size[size]['delta_test'] for size in hidden_sizes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Metrics vs Hidden Size', fontsize=16)
    
    # Plot 1: NN Losses
    axes[0,0].plot(hidden_sizes, nn_train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0,0].plot(hidden_sizes, nn_test_loss, 'b--', label='Test Loss', linewidth=2)
    axes[0,0].set_title('Neural Network Losses')
    axes[0,0].set_xlabel('Hidden Size')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot 2: Linearized Losses
    axes[0,1].plot(hidden_sizes, lin_train_loss, 'r-', label='Train Loss', linewidth=2)
    axes[0,1].plot(hidden_sizes, lin_test_loss, 'r--', label='Test Loss', linewidth=2)
    axes[0,1].set_title('Linearized Model Losses')
    axes[0,1].set_xlabel('Hidden Size')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Plot 3: NN Accuracies
    axes[1,0].plot(hidden_sizes, nn_train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[1,0].plot(hidden_sizes, nn_test_acc, 'b--', label='Test Acc', linewidth=2)
    axes[1,0].set_title('Neural Network Accuracies')
    axes[1,0].set_xlabel('Hidden Size')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Plot 4: Linearized Accuracies
    axes[1,1].plot(hidden_sizes, lin_train_acc, 'r-', label='Train Acc', linewidth=2)
    axes[1,1].plot(hidden_sizes, lin_test_acc, 'r--', label='Test Acc', linewidth=2)
    axes[1,1].set_title('Linearized Model Accuracies')
    axes[1,1].set_xlabel('Hidden Size')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Plot 5: Deltas (log-log)
    axes[2,0].plot(hidden_sizes, delta_train, 'g-', label='Train Delta', linewidth=2)
    axes[2,0].plot(hidden_sizes, delta_test, 'g--', label='Test Delta', linewidth=2)
    
    # Fit and plot trend lines for log-log
    fit_and_analyze_delta(hidden_sizes, delta_train, axes[2,0], 'Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_test, axes[2,0], 'Test', plot_references=False)
    
    axes[2,0].set_title('Network vs Linearization Deltas (Log-Log)')
    axes[2,0].set_xlabel('Hidden Size')
    axes[2,0].set_ylabel('Delta (MSE)')
    axes[2,0].set_xscale('log')
    axes[2,0].set_yscale('log')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # Plot 6: Deltas (log-linear)
    axes[2,1].plot(hidden_sizes, delta_train, 'g-', label='Train Delta', linewidth=2)
    axes[2,1].plot(hidden_sizes, delta_test, 'g--', label='Test Delta', linewidth=2)
    
    # Fit and plot trend lines for log-linear
    fit_and_analyze_delta(hidden_sizes, delta_train, axes[2,1], 'Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_test, axes[2,1], 'Test', plot_references=False)
    
    axes[2,1].set_title('Network vs Linearization Deltas (Log-Linear)')
    axes[2,1].set_xlabel('Hidden Size')
    axes[2,1].set_ylabel('Delta (MSE)')
    axes[2,1].set_yscale('log')
    axes[2,1].legend()
    axes[2,1].grid(True)
    
    # Plot 7: Deltas (linear-linear)
    axes[3,0].plot(hidden_sizes, delta_train, 'g-', label='Train Delta', linewidth=2)
    axes[3,0].plot(hidden_sizes, delta_test, 'g--', label='Test Delta', linewidth=2)
    
    # Fit and plot trend lines for linear-linear
    fit_and_analyze_delta(hidden_sizes, delta_train, axes[3,0], 'Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_test, axes[3,0], 'Test', plot_references=False)
    
    axes[3,0].set_title('Network vs Linearization Deltas (Linear-Linear)')
    axes[3,0].set_xlabel('Hidden Size')
    axes[3,0].set_ylabel('Delta (MSE)')
    axes[3,0].legend()
    axes[3,0].grid(True)
    
    # Plot 8: Loss Differences (absolute values)
    loss_diff_train = np.abs(np.array(nn_train_loss) - np.array(lin_train_loss))
    loss_diff_test = np.abs(np.array(nn_test_loss) - np.array(lin_test_loss))
    axes[3,1].plot(hidden_sizes, loss_diff_train, 'm-', label='|Train Loss Diff|', linewidth=2)
    axes[3,1].plot(hidden_sizes, loss_diff_test, 'm--', label='|Test Loss Diff|', linewidth=2)
    axes[3,1].set_title('Absolute Loss Differences')
    axes[3,1].set_xlabel('Hidden Size')
    axes[3,1].set_ylabel('|Loss Difference|')
    axes[3,1].legend()
    axes[3,1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_params(hidden_size):
    """Calculate total number of parameters for the network architecture."""
    # For input -> hidden layer: (784 * hidden_size) weights + hidden_size biases
    # For hidden -> output layer: (hidden_size * 10) weights + 10 biases
    return (784 * hidden_size + hidden_size) + (hidden_size * 10 + 10)

def main():
    # Specify the run folder to analyze
    run_folder = "runs/run_20241110_195234"  # Replace with your actual run folder
    epoch = 999  # Use last epoch, or specify which epoch to analyze
    
    # Load results
    metrics = load_results(run_folder, epoch)
    
    # Create plots
    save_path = os.path.join(run_folder, 'newfinal_comparison_e1000.png')
    create_comparison_plots(metrics, save_path)
    
    # Print numerical summary
    print(f"\nNumerical summary for epoch {epoch}:")
    for size in sorted(metrics.keys()):
        n_params = calculate_params(size)
        print(f"\nHidden Size: {size} (Parameters: {n_params:,})")
        print(f"NN Train/Test Loss: {metrics[size]['train_loss']:.3f}/{metrics[size]['test_loss']:.3f}")
        print(f"Lin Train/Test Loss: {metrics[size]['train_lin_loss']:.3f}/{metrics[size]['test_lin_loss']:.3f}")
        print(f"NN Train/Test Acc: {metrics[size]['train_accuracy']:.1f}%/{metrics[size]['test_accuracy']:.1f}%")
        print(f"Lin Train/Test Acc: {metrics[size]['train_lin_accuracy']:.1f}%/{metrics[size]['test_lin_accuracy']:.1f}%")
        print(f"Deltas Train/Test: {metrics[size]['delta_train']:.3e}/{metrics[size]['delta_test']:.3e}")

if __name__ == "__main__":
    main()