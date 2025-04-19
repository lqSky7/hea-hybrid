import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from statsmodels.graphics.tsaplots import plot_acf
import logging

# --- Configuration ---
DATA_FILE = "/Users/ca5/Desktop/qnn_fnl/dataset/dgmix_values.txt"
OUTPUT_DIR = "/Users/ca5/Desktop/qnn_fnl/verification/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Plot Styling ---
try:
    plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style
except OSError:
    logger.warning("Seaborn style 'seaborn-v0_8-darkgrid' not found. Using default.")
    plt.style.use('default')

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.labelsize': 16,     # X/Y label size
    'axes.titlesize': 18,     # Plot title size
    'xtick.labelsize': 12,    # X tick label size
    'ytick.labelsize': 12,    # Y tick label size
    'legend.fontsize': 12,    # Legend font size
    'figure.titlesize': 20,   # Figure title size
    'axes.labelweight': 'bold', # Make labels bold
    'axes.titleweight': 'bold', # Make title bold
    'axes.edgecolor': '.15',  # Darker axes edges
    'axes.labelcolor': '.15', # Darker label color
    'xtick.color': '.15',     # Darker tick color
    'ytick.color': '.15',     # Darker tick color
    'text.color': '.15'       # Darker text color
})
logger.info("Plot styling applied.")

# --- Load Data ---
try:
    # Add skiprows=1 to skip the header row
    dgmix_values = np.loadtxt(DATA_FILE, skiprows=1)
    logger.info(f"Loaded {len(dgmix_values)} dGmix values from {DATA_FILE}")
    if len(dgmix_values) == 0:
        logger.error("Data file is empty. Exiting.")
        exit()
except FileNotFoundError:
    logger.error(f"Error: Data file not found at {DATA_FILE}")
    exit()
except Exception as e:
    logger.error(f"Error loading data from {DATA_FILE}: {e}")
    exit()

# --- 1. Plot Data Distribution ---
try:
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.histplot(dgmix_values, kde=True, ax=ax1, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of dGmix Values')
    ax1.set_xlabel('dGmix Value')
    ax1.set_ylabel('Frequency')
    plt.tight_layout()
    dist_plot_path = os.path.join(OUTPUT_DIR, "dgmix_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close(fig1)
    logger.info(f"Saved distribution plot to {dist_plot_path}")
except Exception as e:
    logger.error(f"Failed to create distribution plot: {e}")


# --- 2. Analyze Relations ---
# a) Plot values vs. index
try:
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(dgmix_values, marker='o', linestyle='-', markersize=3, alpha=0.6, color='teal')
    ax2.set_title('dGmix Values vs. Sample Index')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('dGmix Value')
    plt.tight_layout()
    index_plot_path = os.path.join(OUTPUT_DIR, "dgmix_vs_index.png")
    plt.savefig(index_plot_path, dpi=300)
    plt.close(fig2)
    logger.info(f"Saved index plot to {index_plot_path}")
except Exception as e:
    logger.error(f"Failed to create index plot: {e}")

# b) Plot Autocorrelation Function (ACF)
try:
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    plot_acf(dgmix_values, ax=ax3, lags=min(40, len(dgmix_values)//2 - 1), title='Autocorrelation of dGmix Values', color='navy', vlines_kwargs={"colors": 'navy'})
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    # Customize ACF plot appearance further
    for item in ([ax3.title, ax3.xaxis.label, ax3.yaxis.label] +
                 ax3.get_xticklabels() + ax3.get_yticklabels()):
        item.set_color('.15')
        item.set_fontweight('bold' if item in [ax3.title, ax3.xaxis.label, ax3.yaxis.label] else 'normal')
    ax3.grid(True, alpha=0.5)
    plt.tight_layout()
    acf_plot_path = os.path.join(OUTPUT_DIR, "dgmix_autocorrelation.png")
    plt.savefig(acf_plot_path, dpi=300)
    plt.close(fig3)
    logger.info(f"Saved ACF plot to {acf_plot_path}")
except Exception as e:
    logger.error(f"Failed to create ACF plot: {e}")

logger.info("Analysis complete.")
