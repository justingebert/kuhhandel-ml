import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define colors
COLOR_ORANGE = '#E85C02'  # (232, 92, 2)
COLOR_CYAN = '#09AACD'    # (9, 170, 205)

# Set style
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_reward():
    # File path
    csv_path = os.path.join(os.path.dirname(__file__), 'PPO_1.csv')
    
    # Read the data
    df = pd.read_csv(csv_path)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Calculate rolling mean for a smoother line
    window_size = 5
    df['Smoothed'] = df['Value'].rolling(window=window_size, center=True).mean()
    
    # Plot raw data (lighter/transparent)
    ax.plot(df['Step'], df['Value'], color=COLOR_CYAN, alpha=1, label='Raw Data', linewidth=1)
    
    # Plot smoothed data (solid)
    ax.plot(df['Step'], df['Smoothed'], color=COLOR_ORANGE, label=f'Moving Average (n={window_size})', linewidth=2.5)
    
    # Customize the axes
    ax.set_title("Average Loss of Value Function per Episode", fontsize=16, pad=20, fontweight='bold', color='#333333')
    ax.set_xlabel("Training Steps", fontsize=12, labelpad=10)
    ax.set_ylabel("Average Loss", fontsize=12, labelpad=10)
    
    # Grid configuration
    ax.grid(True, linestyle='--', alpha=0.7, color='lightgray')
    
    # Legend
    ax.legend(frameon=True, framealpha=0.9, loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'value_loss_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    #plt.show()

if __name__ == "__main__":
    plot_reward()
