import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def load_results(run_folder, epoch):
    """Load results for all hidden sizes from a specific run folder."""
    metrics_by_size = {}
    
    for dir_name in os.listdir(run_folder):
        if dir_name.startswith('hidden_'):
            hidden_size = int(dir_name.split('_')[1])
            results_file = os.path.join(run_folder, dir_name, 'results.json')
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    metrics_by_size[hidden_size] = data['training_results'][epoch]
    
    return metrics_by_size

def power_law(x, a, b):
    """Power law function: y = ax^b with overflow protection"""
    return a * np.exp(b * np.log(x))  # More numerically stable than np.power

def fit_and_analyze_delta(hidden_sizes, deltas, ax, label, plot_references=False):
    """Fit different functions to delta and return the best fit"""
    x_data = np.array(hidden_sizes)
    y_data = np.array(deltas)
    
    # Skip first P% of data points for fitting
    P = 0.05
    cutoff_idx = int(len(x_data) * P)
    x_fit = x_data[cutoff_idx:]
    y_fit = y_data[cutoff_idx:]
    
    # Fit power law
    popt_power, _ = curve_fit(power_law, x_fit, y_fit, p0=[1, -1])
    
    # Calculate R-squared for power law fit
    y_power = power_law(x_fit, *popt_power)
    r2_power = 1 - np.sum((y_fit - y_power)**2) / np.sum((y_fit - np.mean(y_fit))**2)
    
    # Plot fits
    x_smooth = np.linspace(min(x_data), max(x_data), 100)
    ax.plot(x_smooth, power_law(x_smooth, *popt_power), '--', 
            label=f'{label} power law: {popt_power[0]:.2e}x^{popt_power[1]:.2f} (R²={r2_power:.3f})')
    
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
    std_train_loss = [metrics_by_size[size]['train_loss'] for size in hidden_sizes]
    std_test_loss = [metrics_by_size[size]['test_loss'] for size in hidden_sizes]
    trans_train_loss = [metrics_by_size[size]['train_trans_loss'] for size in hidden_sizes]
    trans_test_loss = [metrics_by_size[size]['test_trans_loss'] for size in hidden_sizes]
    lin_train_loss = [metrics_by_size[size]['train_lin_loss'] for size in hidden_sizes]
    lin_test_loss = [metrics_by_size[size]['test_lin_loss'] for size in hidden_sizes]
    
    std_train_acc = [metrics_by_size[size]['train_accuracy'] for size in hidden_sizes]
    std_test_acc = [metrics_by_size[size]['test_accuracy'] for size in hidden_sizes]
    trans_train_acc = [metrics_by_size[size]['train_trans_accuracy'] for size in hidden_sizes]
    trans_test_acc = [metrics_by_size[size]['test_trans_accuracy'] for size in hidden_sizes]
    lin_train_acc = [metrics_by_size[size]['train_lin_accuracy'] for size in hidden_sizes]
    lin_test_acc = [metrics_by_size[size]['test_lin_accuracy'] for size in hidden_sizes]
    
    delta_std_train = [metrics_by_size[size]['delta_train'] for size in hidden_sizes]
    delta_std_test = [metrics_by_size[size]['delta_test'] for size in hidden_sizes]
    delta_trans_train = [metrics_by_size[size]['delta_trans_train'] for size in hidden_sizes]
    delta_trans_test = [metrics_by_size[size]['delta_trans_test'] for size in hidden_sizes]
    
    # Create figure with subplots (now 5x2 to accommodate transformed network)
    fig, axes = plt.subplots(5, 2, figsize=(15, 25))
    fig.suptitle('Metrics vs Hidden Size', fontsize=16)
    
    # Plot 1-2: Standard NN Losses and Accuracies
    axes[0,0].plot(hidden_sizes, std_train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0,0].plot(hidden_sizes, std_test_loss, 'b--', label='Test Loss', linewidth=2)
    axes[0,0].set_title('Standard Network Losses')
    axes[0,0].set_xlabel('Hidden Size')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot 3-4: Transformed NN Losses and Accuracies
    axes[1,0].plot(hidden_sizes, trans_train_loss, 'r-', label='Train Loss', linewidth=2)
    axes[1,0].plot(hidden_sizes, trans_test_loss, 'r--', label='Test Loss', linewidth=2)
    axes[1,0].set_title('Transformed Network Losses')
    axes[1,0].set_xlabel('Hidden Size')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Plot 5-6: NTK Losses and Accuracies
    axes[2,0].plot(hidden_sizes, lin_train_loss, 'g-', label='Train Loss', linewidth=2)
    axes[2,0].plot(hidden_sizes, lin_test_loss, 'g--', label='Test Loss', linewidth=2)
    axes[2,0].set_title('NTK Model Losses')
    axes[2,0].set_xlabel('Hidden Size')
    axes[2,0].set_ylabel('Loss')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # Accuracies in right column
    axes[0,1].plot(hidden_sizes, std_train_acc, 'b-', label='Train Acc', linewidth=2)
    axes[0,1].plot(hidden_sizes, std_test_acc, 'b--', label='Test Acc', linewidth=2)
    axes[0,1].set_title('Standard Network Accuracies')
    axes[0,1].set_xlabel('Hidden Size')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    axes[1,1].plot(hidden_sizes, trans_train_acc, 'r-', label='Train Acc', linewidth=2)
    axes[1,1].plot(hidden_sizes, trans_test_acc, 'r--', label='Test Acc', linewidth=2)
    axes[1,1].set_title('Transformed Network Accuracies')
    axes[1,1].set_xlabel('Hidden Size')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    axes[2,1].plot(hidden_sizes, lin_train_acc, 'g-', label='Train Acc', linewidth=2)
    axes[2,1].plot(hidden_sizes, lin_test_acc, 'g--', label='Test Acc', linewidth=2)
    axes[2,1].set_title('NTK Model Accuracies')
    axes[2,1].set_xlabel('Hidden Size')
    axes[2,1].set_ylabel('Accuracy (%)')
    axes[2,1].legend()
    axes[2,1].grid(True)
    
    # Plot 7-8: Deltas (log-log)
    axes[3,0].plot(hidden_sizes, delta_std_train, 'b-', label='Standard Train', linewidth=2)
    axes[3,0].plot(hidden_sizes, delta_std_test, 'b--', label='Standard Test', linewidth=2)
    axes[3,0].plot(hidden_sizes, delta_trans_train, 'r-', label='Transformed Train', linewidth=2)
    axes[3,0].plot(hidden_sizes, delta_trans_test, 'r--', label='Transformed Test', linewidth=2)
    
    # Fit and plot trend lines
    fit_and_analyze_delta(hidden_sizes, delta_std_train, axes[3,0], 'Std Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_std_test, axes[3,0], 'Std Test')
    fit_and_analyze_delta(hidden_sizes, delta_trans_train, axes[3,0], 'Trans Train')
    fit_and_analyze_delta(hidden_sizes, delta_trans_test, axes[3,0], 'Trans Test')
    
    axes[3,0].set_title('Network vs NTK Deltas (Log-Log)')
    axes[3,0].set_xlabel('Hidden Size')
    axes[3,0].set_ylabel('Delta (MSE)')
    axes[3,0].set_xscale('log')
    axes[3,0].set_yscale('log')
    axes[3,0].legend()
    axes[3,0].grid(True)
    
    # Plot 9-10: Loss Differences
    std_loss_diff_train = np.abs(np.array(std_train_loss) - np.array(lin_train_loss))
    std_loss_diff_test = np.abs(np.array(std_test_loss) - np.array(lin_test_loss))
    trans_loss_diff_train = np.abs(np.array(trans_train_loss) - np.array(lin_train_loss))
    trans_loss_diff_test = np.abs(np.array(trans_test_loss) - np.array(lin_test_loss))
    
    axes[4,0].plot(hidden_sizes, std_loss_diff_train, 'b-', label='Standard Train', linewidth=2)
    axes[4,0].plot(hidden_sizes, std_loss_diff_test, 'b--', label='Standard Test', linewidth=2)
    axes[4,0].plot(hidden_sizes, trans_loss_diff_train, 'r-', label='Transformed Train', linewidth=2)
    axes[4,0].plot(hidden_sizes, trans_loss_diff_test, 'r--', label='Transformed Test', linewidth=2)
    axes[4,0].set_title('Absolute Loss Differences')
    axes[4,0].set_xlabel('Hidden Size')
    axes[4,0].set_ylabel('|Loss Difference|')
    axes[4,0].legend()
    axes[4,0].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_scaling_plots(metrics_by_size, save_path=None):
    """Create scaling plots for both standard and transformed deltas."""
    hidden_sizes = sorted(metrics_by_size.keys())
    
    # Extract deltas
    delta_std_train = [metrics_by_size[size]['delta_train'] for size in hidden_sizes]
    delta_std_test = [metrics_by_size[size]['delta_test'] for size in hidden_sizes]
    delta_trans_train = [metrics_by_size[size]['delta_trans_train'] for size in hidden_sizes]
    delta_trans_test = [metrics_by_size[size]['delta_trans_test'] for size in hidden_sizes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Scaling Laws: Network vs NTK Deltas', fontsize=16)
    
    # Plot 1: Standard Deltas (log-log)
    axes[0,0].plot(hidden_sizes, delta_std_train, 'b-', label='Standard Train Delta', linewidth=2)
    axes[0,0].plot(hidden_sizes, delta_std_test, 'b--', label='Standard Test Delta', linewidth=2)
    fit_and_analyze_delta(hidden_sizes, delta_std_train, axes[0,0], 'Standard Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_std_test, axes[0,0], 'Standard Test')
    axes[0,0].set_title('Standard Network vs NTK (Log-Log)')
    axes[0,0].set_xlabel('Hidden Size')
    axes[0,0].set_ylabel('Delta (MSE)')
    axes[0,0].set_xscale('log')
    axes[0,0].set_yscale('log')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Plot 2: Transformed Deltas (log-log)
    axes[0,1].plot(hidden_sizes, delta_trans_train, 'r-', label='Transformed Train Delta', linewidth=2)
    axes[0,1].plot(hidden_sizes, delta_trans_test, 'r--', label='Transformed Test Delta', linewidth=2)
    fit_and_analyze_delta(hidden_sizes, delta_trans_train, axes[0,1], 'Transformed Train', plot_references=True)
    fit_and_analyze_delta(hidden_sizes, delta_trans_test, axes[0,1], 'Transformed Test')
    axes[0,1].set_title('Transformed Network vs NTK (Log-Log)')
    axes[0,1].set_xlabel('Hidden Size')
    axes[0,1].set_ylabel('Delta (MSE)')
    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Plot 3: Comparison of Train Deltas
    axes[1,0].plot(hidden_sizes, delta_std_train, 'b-', label='Standard Train Delta', linewidth=2)
    axes[1,0].plot(hidden_sizes, delta_trans_train, 'r-', label='Transformed Train Delta', linewidth=2)
    axes[1,0].set_title('Train Deltas Comparison (Log-Log)')
    axes[1,0].set_xlabel('Hidden Size')
    axes[1,0].set_ylabel('Delta (MSE)')
    axes[1,0].set_xscale('log')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Plot 4: Comparison of Test Deltas
    axes[1,1].plot(hidden_sizes, delta_std_test, 'b--', label='Standard Test Delta', linewidth=2)
    axes[1,1].plot(hidden_sizes, delta_trans_test, 'r--', label='Transformed Test Delta', linewidth=2)
    axes[1,1].set_title('Test Deltas Comparison (Log-Log)')
    axes[1,1].set_xlabel('Hidden Size')
    axes[1,1].set_ylabel('Delta (MSE)')
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    # Specify the run folder to analyze
    # run_folder = "runs/run_20241226_113533"  # Replace with your actual run folder
    run_folder = "runs/run_20241110_215433"  # Replace with your actual run folder
    epoch = -1  # Use last epoch
    
    # Load results
    metrics = load_results(run_folder, epoch)
    
    # Create both types of plots
    comparison_save_path = os.path.join(run_folder, 'comparison_plots.png')
    create_comparison_plots(metrics, comparison_save_path)  # Create all comparison plots
    
    scaling_save_path = os.path.join(run_folder, 'scaling_laws.png')
    create_scaling_plots(metrics, scaling_save_path)  # Create scaling law plots
    
    # Print numerical summary
    print(f"\nNumerical summary for epoch {epoch}:")
    for size in sorted(metrics.keys()):
        print(f"\nHidden Size: {size}")
        print(f"Standard Deltas Train/Test: {metrics[size]['delta_train']:.3e}/{metrics[size]['delta_test']:.3e}")
        print(f"Transformed Deltas Train/Test: {metrics[size]['delta_trans_train']:.3e}/{metrics[size]['delta_trans_test']:.3e}")

if __name__ == "__main__":
    main()