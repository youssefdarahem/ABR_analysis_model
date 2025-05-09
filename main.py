import torch
from abr_models.classifier import SimpleCnnModel
from abr_models.joint import JointModel
from abr_models.regressor import Regressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import argparse
import os

from utils.dataset_utils import DetectorDataset


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="ABR Signal Analysis with Wave V Detection")
    parser.add_argument("--model_path", type=str, default="weights/joint_model_weights.pth",
                        help="Path to the model weights file (default: weights/joint_model_weights.pth)")
    parser.add_argument("--data_path", type=str, default="sample/sample.csv",
                        help="Path to the sample data CSV file (default: sample/sample.csv)")
    return parser.parse_args()


def plot_signal(fig, ax, x, y, exist, loc, current_idx, total_samples):
    """
    Plot the ABR signal with predictions
    
    Args:
        fig: matplotlib figure
        ax: matplotlib axes
        x: input signal tensor
        y: ground truth tensor
        exist: prediction of peak existence
        loc: predicted peak location
        current_idx: current sample index
        total_samples: total number of samples
    """
    ax.clear()
    ax.plot(x[0, 0, :].numpy(), label="Reference Signal")
    ax.plot(x[0, 1, :].numpy(), label="ABR Signal")
    
    if y[0].item():
        ax.axvline(x=y[1].item(), color='r',
                    linestyle='--', label="Wave V location (True)")

    if exist.item():
        ax.axvline(x=loc.item(), color='b',
                    linestyle='--', label="Wave V location (Predicted)")
    
    ax.set_title(f"ABR Signal {current_idx+1}/{total_samples} - Peak Exists: {exist.item()}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()
    
    # Add usage instructions
    instruction_text = "Navigation: ← Previous | Next → | q: Quit"
    fig.text(0.5, 0.01, instruction_text, ha='center', fontsize=10)
    
    fig.canvas.draw()


def on_key_press(event, fig, data_samples, current_idx):
    """
    Handle key press events for signal navigation
    
    Args:
        event: key press event
        fig: matplotlib figure
        data_samples: list of data samples
        current_idx: current sample index reference [current_idx]
    """
    if event.key == 'right' or event.key == 'n':
        # Next signal
        current_idx[0] = (current_idx[0] + 1) % len(data_samples)
    elif event.key == 'left' or event.key == 'p':
        # Previous signal
        current_idx[0] = (current_idx[0] - 1) % len(data_samples)
    elif event.key == 'q' or event.key == 'escape':
        # Quit
        plt.close(fig)
        return
    
    # Get current sample data
    x, y, exist, loc = data_samples[current_idx[0]]
    
    # Update plot
    plot_signal(fig, fig.axes[0], x, y, exist, loc, current_idx[0], len(data_samples))


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Validate file paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights file not found at '{args.model_path}'")
        return
        
    if not os.path.exists(args.data_path):
        print(f"Error: Sample data file not found at '{args.data_path}'")
        return
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    sample_data = pd.read_csv(args.data_path)
    dataset = DetectorDataset(sample_data)
    
    # Load model
    print(f"Loading model weights from: {args.model_path}")
    model = JointModel(
        regressor=Regressor(input_dim=2, output_dim=1, hidden_dim=256),
        classifier=SimpleCnnModel(base_model=None)
    )
    model.load_state_dict(args.model_path)
    model.eval()
    
    # Process all samples
    data_samples = []
    with torch.no_grad():
        for i, (x, y, _) in enumerate(dataset):
            x_unsqueezed = x.unsqueeze(0)
            exist, loc = model(x_unsqueezed)
            
            # Store sample data
            data_samples.append((x_unsqueezed, y, exist, loc))
            
            print(f"Sample {i+1}/{len(dataset)}")
            print(f"Peak exists: {exist.item()}, Location: {loc.item()}, True: {y}")
    
    if not data_samples:
        print("No samples to display!")
        return
    
    # Setup interactive plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    current_idx = [0]  # Use list so it can be modified inside the callback function
    
    # Plot first sample
    x, y, exist, loc = data_samples[0]
    plot_signal(fig, ax, x, y, exist, loc, current_idx[0], len(data_samples))
    
    # Connect key press event handler
    fig.canvas.mpl_connect('key_press_event', 
                          lambda event: on_key_press(event, fig, data_samples, current_idx))
    
    # Show instructions
    print("\nABR Signal Viewer")
    print("=================")
    print("Navigation:")
    print("  - Right arrow or 'n': Next signal")
    print("  - Left arrow or 'p': Previous signal")
    print("  - 'q' or ESC: Close the viewer")
    
    plt.show(block=True)


if __name__ == "__main__":
    main()
