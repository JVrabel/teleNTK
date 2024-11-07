"""
Plotting utilities for analyzing training results
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(experiment_dir: str):
    """Load results from an experiment directory."""
    with open(os.path.join(experiment_dir, "finite_results.json"), "r") as f:
        finite_results = json.load(f)
    
    with open(os.path.join(experiment_dir, "ntk_results.json"), "r") as f:
        ntk_results = json.load(f)
        
    return finite_results, ntk_results

def plot_training_curves(experiment_dir: str, save_dir: str = None):
    """Plot training curves comparing finite network and NTK."""
    finite_results, ntk_results = load_experiment_results(experiment_dir)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training loss
    ax1.plot(finite_results["train_loss"], label="Finite Network")
    ax1.plot(ntk_results["train_loss"], label="NTK")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    # Plot validation loss
    ax2.plot(finite_results["val_loss"], label="Finite Network")
    ax2.plot(ntk_results["val_loss"], label="NTK")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    # Plot training accuracy
    ax3.plot(finite_results["train_acc"], label="Finite Network")
    ax3.plot(ntk_results["train_acc"], label="NTK")
    ax3.set_title("Training Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    
    # Plot validation accuracy
    ax4.plot(finite_results["val_acc"], label="Finite Network")
    ax4.plot(ntk_results["val_acc"], label="NTK")
    ax4.set_title("Validation Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_curves.png"))
    else:
        plt.show()
        
def plot_timing_comparison(experiment_dir: str, save_dir: str = None):
    """Plot timing comparison between finite network and NTK."""
    finite_results, ntk_results = load_experiment_results(experiment_dir)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(finite_results["epoch_times"], label="Finite Network")
    plt.plot(ntk_results["epoch_times"], label="NTK")
    plt.title("Training Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "timing_comparison.png"))
    else:
        plt.show() 