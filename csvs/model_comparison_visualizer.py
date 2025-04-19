import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import os
import traceback
import sys

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.figsize'] = (14, 10)

# Create output directory for graphs
os.makedirs('comparison_graphs', exist_ok=True)

# Define hybrid model gradient colors (pink-to-blue)
hybrid_cmap = LinearSegmentedColormap.from_list("hybrid_gradient", ["#FF69B4", "#1E90FF"])

# Load all CSV files
model_files = {
    'Dense': 'dense.csv',
    'Hybrid': 'hybrid.csv',
    'MLP': 'mlp.csv',
    'Linear': 'model_performance_summary_linear.csv',
    'TDNN': 'tdnn.csv'
}

model_data = {}
for model_name, file_name in model_files.items():
    try:
        df = pd.read_csv(file_name)
        # Convert to dictionary for easier manipulation
        model_dict = dict(zip(df['Metric'], df['Value']))
        # Convert string values to float where possible
        for key, value in model_dict.items():
            try:
                if isinstance(value, str) and '%' in value:
                    model_dict[key] = float(value.strip('%'))/100
                else:
                    model_dict[key] = float(value)
            except (ValueError, TypeError):
                pass
        model_data[model_name] = model_dict
    except Exception as e:
        print(f"Error loading {file_name}: {e}")

# Check if we successfully loaded any data
if not model_data:
    print("No valid data files found. Please check the file paths and formats.")
    sys.exit(1)

print(f"Successfully loaded data for {len(model_data)} models: {', '.join(model_data.keys())}")
print(f"Metrics available: {list(next(iter(model_data.values())).keys())}")

# --- VISUALIZATION FUNCTIONS ---

def get_model_color(model_name, position=None):
    """Get color for model with special gradient for hybrid model."""
    colors = {
        'Dense': '#FF6B6B',     # Red
        'Linear': '#4ECDC4',    # Teal
        'MLP': '#FFD166',       # Yellow
        'TDNN': '#6A0572',      # Purple
    }
    
    if model_name == 'Hybrid' and position is not None:
        # Return a color from the gradient
        return hybrid_cmap(position)
    
    return colors.get(model_name, '#333333')  # Default dark gray

def execute_viz_function(func_name, function):
    """Execute a visualization function with proper error handling."""
    print(f"Generating {func_name}...")
    try:
        function()
        print(f"Successfully generated {func_name}")
    except Exception as e:
        print(f"Error generating {func_name}: {str(e)}")
        traceback.print_exc()

def bar_comparison(metric, title, ylabel, filename):
    """Create a bar chart comparing models on a specific metric."""
    values = []
    labels = []
    colors = []
    
    for i, (model_name, metrics) in enumerate(model_data.items()):
        if metric in metrics:
            values.append(metrics[metric])
            labels.append(model_name)
            colors.append(get_model_color(model_name, 0.5 if model_name == 'Hybrid' else None))
    
    if not values:
        print(f"No data available for metric: {metric}")
        return
    
    plt.figure(figsize=(14, 10))
    bars = plt.bar(labels, values, color=colors, width=0.6)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(values),
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.title(title, fontweight='extra bold', fontsize=22, pad=20)
    plt.ylabel(ylabel, fontweight='extra bold', fontsize=18)
    plt.xlabel('Model', fontweight='extra bold', fontsize=18)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Special highlight for hybrid model
    if 'Hybrid' in labels:
        idx = labels.index('Hybrid')
        plt.axvspan(idx-0.4, idx+0.4, color='#E0FFFF', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f'comparison_graphs/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def radar_chart():
    """Create a radar chart comparing all models across multiple metrics."""
    # Common metrics across models
    metrics = ['MAE', 'RMSE', 'R2']
    
    # Get values for each model
    model_values = {}
    for model_name, metrics_dict in model_data.items():
        model_values[model_name] = [metrics_dict.get(m, 0) for m in metrics]
    
    # For visualization purposes, lower values are better for MAE and RMSE
    # but higher values are better for R2. Let's invert MAE and RMSE
    for model in model_values:
        # Invert MAE and RMSE (smaller is better, so we take reciprocal)
        if model_values[model][0] != 0:  # Avoid division by zero
            model_values[model][0] = 1 / model_values[model][0]
        if model_values[model][1] != 0:
            model_values[model][1] = 1 / model_values[model][1]
    
    # Normalize all values to 0-1 range for each metric
    normalized_values = {}
    for i, metric in enumerate(metrics):
        max_val = max(abs(model_values[model][i]) for model in model_values)
        if max_val == 0:
            max_val = 1  # Avoid division by zero
        
        for model in model_values:
            if model not in normalized_values:
                normalized_values[model] = []
            normalized_values[model].append(model_values[model][i] / max_val)
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.figure(figsize=(14, 14)), plt.subplot(111, polar=True)
    
    # Add metrics to the chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{m}\n(normalized)' for m in metrics], fontsize=16, fontweight='bold')
    
    # Set y-ticks
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=14)
    
    # Draw the radar chart for each model
    for i, (model, values) in enumerate(normalized_values.items()):
        values += values[:1]  # Close the loop
        
        if model == 'Hybrid':
            # Create gradient line for hybrid
            line, = ax.plot(angles, values, linewidth=3, linestyle='-', label=model)
            # Create a gradient color effect
            for j in range(len(angles)-1):
                ax.plot(angles[j:j+2], values[j:j+2], 
                         color=hybrid_cmap(j/(len(angles)-2)), 
                         linewidth=4)
            # Fill with light gradient
            ax.fill(angles, values, alpha=0.25, color='#AEDDFF')
        else:
            color = get_model_color(model)
            ax.plot(angles, values, linewidth=3, linestyle='-', label=model, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, 
               fontsize=14, facecolor='white', edgecolor='gray')
    plt.title('Model Performance Comparison (Radar Chart)', size=22, y=1.08, fontweight='extra bold')
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def loss_comparison_3d():
    """Create a 3D plot comparing training and validation loss."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(len(model_data))
    training_losses = []
    validation_losses = []
    model_names = []
    
    for model_name, metrics in model_data.items():
        if 'Training Loss (Final)' in metrics and 'Validation Loss (Final)' in metrics:
            training_losses.append(metrics['Training Loss (Final)'])
            validation_losses.append(metrics['Validation Loss (Final)'])
            model_names.append(model_name)
    
    # Create bar positions
    x_pos = np.arange(len(model_names))
    y_pos = [0] * len(model_names)
    z_pos = [0] * len(model_names)
    
    # Width and depth of bars
    dx = 0.6
    dy = 0.4
    
    # Training loss bars (position 0)
    for i, (x, y, z, model) in enumerate(zip(x_pos, y_pos, z_pos, model_names)):
        if model == 'Hybrid':
            # Create special gradient colored bar for hybrid
            cmap = hybrid_cmap
            color = cmap(0.3)  # Use position in gradient
        else:
            color = get_model_color(model)
            
        ax.bar3d(x, y, z, dx, dy, training_losses[i], color=color, alpha=0.8, 
                 shade=True, label=f"{model} (Training)" if i == 0 else "")
    
    # Validation loss bars (position 1)
    for i, (x, y, z, model) in enumerate(zip(x_pos, y_pos+1, z_pos, model_names)):
        if model == 'Hybrid':
            # Create special gradient colored bar for hybrid
            cmap = hybrid_cmap
            color = cmap(0.7)  # Use position in gradient
        else:
            color = get_model_color(model)
            # Add transparency to differentiate from training loss
            color = list(plt.cm.colors.to_rgba(color))
            color[3] = 0.6  # Alpha
            
        ax.bar3d(x, y, z, dx, dy, validation_losses[i], color=color, alpha=0.8, 
                 shade=True, label=f"{model} (Validation)" if i == 0 else "")
    
    # Custom legend for training and validation
    handles = [
        plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.8),
        plt.Rectangle((0,0), 1, 1, color='gray', alpha=0.5)
    ]
    labels = ['Training Loss', 'Validation Loss']
    
    # Add another legend for models
    model_handles = []
    for model in model_names:
        color = get_model_color(model, 0.5 if model == 'Hybrid' else None)
        model_handles.append(plt.Rectangle((0,0), 1, 1, color=color))
    
    # Combine legends
    ax.legend(handles + model_handles, labels + model_names, 
              loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=3, fontsize=14)
    
    # Set labels and title
    ax.set_xlabel('Model', fontweight='extra bold', fontsize=18, labelpad=20)
    ax.set_ylabel('Loss Type', fontweight='extra bold', fontsize=18, labelpad=20)
    ax.set_zlabel('Loss Value', fontweight='extra bold', fontsize=18, labelpad=20)
    ax.set_title('Training vs Validation Loss Comparison', fontweight='extra bold', fontsize=22, pad=20)
    
    # Set ticks
    ax.set_xticks(x_pos + dx/2)
    ax.set_xticklabels(model_names, fontsize=14)
    ax.set_yticks([0.2, 1.2])
    ax.set_yticklabels(['Training', 'Validation'], fontsize=14)
    
    # Improve perspective
    ax.view_init(elev=30, azim=140)
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/loss_comparison_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def performance_heatmap():
    """Create a heatmap of model performances across metrics."""
    metrics_to_plot = ['MAE', 'RMSE', 'R2', 'Training Loss (Final)', 'Validation Loss (Final)']
    
    # Prepare data for heatmap
    data = []
    model_names = []
    
    for model_name, metrics in model_data.items():
        model_names.append(model_name)
        model_row = []
        for metric in metrics_to_plot:
            model_row.append(metrics.get(metric, np.nan))
        data.append(model_row)
    
    # Convert to numpy array for heatmap
    data_array = np.array(data)
    
    # For visualization purposes, we want lower values for MAE, RMSE, losses to show as "better" (higher in the heatmap)
    # But higher values for R2 to show as "better". Let's set up the normalization accordingly.
    for i, metric in enumerate(metrics_to_plot):
        if metric != 'R2':  # For metrics where lower is better
            # Check for NaN values
            valid_indices = ~np.isnan(data_array[:, i])
            if np.any(valid_indices):
                max_val = np.max(data_array[valid_indices, i])
                min_val = np.min(data_array[valid_indices, i])
                if max_val != min_val:  # Avoid division by zero
                    data_array[valid_indices, i] = 1 - (data_array[valid_indices, i] - min_val) / (max_val - min_val)
    
    # Create heatmap
    plt.figure(figsize=(16, 10))
    
    # Custom colormap with blue-white-red gradient
    cmap = plt.cm.RdYlGn

    # Create heatmap with custom annotations
    ax = sns.heatmap(data_array, annot=True, cmap=cmap, fmt='.4f', 
                    xticklabels=metrics_to_plot, yticklabels=model_names, 
                    linewidths=1, linecolor='white', cbar=True)
    
    # Highlight the hybrid model row with a pink-blue gradient background
    if 'Hybrid' in model_names:
        hybrid_idx = model_names.index('Hybrid')
        for i in range(len(metrics_to_plot)):
            ax.add_patch(plt.Rectangle((i, hybrid_idx), 1, 1, fill=False, 
                                      edgecolor='blue', lw=3, clip_on=False))
    
    plt.title('Model Performance Metrics Heatmap', fontweight='extra bold', fontsize=22, pad=20)
    plt.xlabel('Metrics', fontweight='extra bold', fontsize=18)
    plt.ylabel('Models', fontweight='extra bold', fontsize=18)
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def bubble_chart():
    """Create a bubble chart showing the relationship between metrics with size representing R2."""
    if not all(metric in model_data[model] for model in model_data for metric in ['MAE', 'RMSE', 'R2']):
        print("Missing required metrics for bubble chart")
        return
        
    plt.figure(figsize=(14, 10))
    
    for model_name, metrics in model_data.items():
        x = metrics['MAE']
        y = metrics['RMSE']
        size = metrics['R2'] * 1000  # Scale R2 for better visibility
        
        if model_name == 'Hybrid':
            # Create a gradient-filled bubble for the hybrid model
            plt.scatter(x, y, s=size, alpha=0.7, label=model_name,
                        color=get_model_color(model_name, 0.5),
                        edgecolor='black', linewidth=2)
            
            # Add a highlight ring around the hybrid bubble
            plt.scatter(x, y, s=size+100, alpha=0.2, color='#ADD8E6', edgecolor='none')
        else:
            plt.scatter(x, y, s=size, alpha=0.7, label=model_name,
                        color=get_model_color(model_name),
                        edgecolor='black', linewidth=1)
    
    # Add model name labels to each bubble
    for model_name, metrics in model_data.items():
        plt.annotate(model_name, 
                    xy=(metrics['MAE'], metrics['RMSE']),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold')
    
    plt.title('Model Performance Bubble Chart', fontweight='extra bold', fontsize=22)
    plt.xlabel('Mean Absolute Error (MAE)', fontweight='extra bold', fontsize=18)
    plt.ylabel('Root Mean Square Error (RMSE)', fontweight='extra bold', fontsize=18)
    
    # Add a legend explaining the bubble size
    sizes = [0.85, 0.90, 0.95, 0.99]
    labels = [f'R² = {s}' for s in sizes]
    
    # Create proxy artists for the legend
    handles = [plt.scatter([], [], s=s*1000, color='gray', alpha=0.7, edgecolor='black') for s in sizes]
    plt.legend(handles, labels, title="Bubble Size Legend", 
              title_fontsize=14, fontsize=12, 
              loc='upper right', frameon=True,
              scatterpoints=1)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Equal aspect ratio gives true scaling
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/bubble_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_surface():
    """Create a 3D surface plot showing the relationship between MAE, RMSE, and R2."""
    if not all('MAE' in model_data[model] and 'RMSE' in model_data[model] and 'R2' in model_data[model] 
               for model in model_data):
        print("Missing required metrics for 3D surface plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract metrics data
    mae_values = [model_data[model]['MAE'] for model in model_data]
    rmse_values = [model_data[model]['RMSE'] for model in model_data]
    r2_values = [model_data[model]['R2'] for model in model_data]
    
    # Create a meshgrid for the surface plot
    mae_range = np.linspace(min(mae_values)*0.9, max(mae_values)*1.1, 20)
    rmse_range = np.linspace(min(rmse_values)*0.9, max(rmse_values)*1.1, 20)
    
    X, Y = np.meshgrid(mae_range, rmse_range)
    
    # Create a theoretical surface: higher R2 is achieved with lower MAE and RMSE
    # This is a simplification for visualization
    Z = 1 - (X / (2*max(mae_values))) - (Y / (2*max(rmse_values)))
    Z = np.clip(Z, 0, 1)  # Clip R2 values between 0 and 1
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')
    
    # Plot actual data points
    for i, model in enumerate(model_data):
        if model == 'Hybrid':
            # Special marker and color for hybrid model
            ax.scatter(mae_values[i], rmse_values[i], r2_values[i], 
                       c=[get_model_color(model, 0.5)], s=200, marker='*', 
                       label=model, edgecolor='black', linewidth=1.5, zorder=10)
            
            # Add vertical line to surface for better visualization
            ax.plot([mae_values[i], mae_values[i]], 
                    [rmse_values[i], rmse_values[i]], 
                    [0, r2_values[i]], 'r--', alpha=0.5)
        else:
            ax.scatter(mae_values[i], rmse_values[i], r2_values[i], 
                       c=[get_model_color(model)], s=100, marker='o', 
                       label=model, edgecolor='black')
            
            # Add vertical line to surface
            ax.plot([mae_values[i], mae_values[i]], 
                    [rmse_values[i], rmse_values[i]], 
                    [0, r2_values[i]], 'k--', alpha=0.3)
    
    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Theoretical R² Value')
    
    # Setting labels and title
    ax.set_xlabel('Mean Absolute Error (MAE)', fontweight='extra bold', fontsize=18, labelpad=15)
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontweight='extra bold', fontsize=18, labelpad=15) 
    ax.set_zlabel('R² Score', fontweight='extra bold', fontsize=18, labelpad=15)
    ax.set_title('3D Performance Surface: MAE vs RMSE vs R²', fontweight='extra bold', fontsize=22, pad=20)
    
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(model_data), fontsize=14)
    
    # Set the viewing angle
    ax.view_init(elev=30, azim=135)
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/3d_performance_surface.png', dpi=300, bbox_inches='tight')
    plt.close()

def r2_benchmark_plot():
    """Create a plot ranking the models by their R2 scores with thresholds."""
    if not all('R2' in model_data[model] for model in model_data):
        print("Missing R2 data for R2 benchmark plot")
        return
        
    # Get models and R2 scores
    models = []
    r2_scores = []
    
    for model_name, metrics in model_data.items():
        models.append(model_name)
        r2_scores.append(metrics['R2'])
    
    # Sort data by R2 score
    sorted_indices = np.argsort(r2_scores)
    models = [models[i] for i in sorted_indices]
    r2_scores = [r2_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(models, r2_scores, height=0.6)
    
    # Color bars based on model name
    for i, bar in enumerate(bars):
        if models[i] == 'Hybrid':
            # Create gradient fill for hybrid model
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            
            # Create gradient using multiple thin bars
            segments = 50
            for j in range(segments):
                seg_width = width / segments
                seg_x = x + j * seg_width
                color = hybrid_cmap(j / segments)
                plt.barh([models[i]], [seg_width], height=height, left=seg_x, color=color)
                
            # Add black border around the hybrid bar
            plt.barh([models[i]], [width], height=height, left=x, 
                     color='none', edgecolor='black', linewidth=2)
        else:
            bar.set_color(get_model_color(models[i]))
    
    # Add benchmark zones
    plt.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Good (R² ≥ 0.9)')
    plt.axvline(x=0.95, color='green', linestyle='--', alpha=0.7, label='Excellent (R² ≥ 0.95)')
    plt.axvline(x=0.99, color='blue', linestyle='--', alpha=0.7, label='Outstanding (R² ≥ 0.99)')
    
    # Add value labels
    for i, v in enumerate(r2_scores):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold', fontsize=14)
    
    plt.xlim(0.8, 1.05)
    plt.title('R² Score Benchmark Comparison', fontweight='extra bold', fontsize=22)
    plt.xlabel('R² Score', fontweight='extra bold', fontsize=18)
    plt.ylabel('Model', fontweight='extra bold', fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=14)
    
    # Add colored background zones
    plt.axvspan(0, 0.9, alpha=0.1, color='gray', zorder=0)
    plt.axvspan(0.9, 0.95, alpha=0.1, color='red', zorder=0)
    plt.axvspan(0.95, 0.99, alpha=0.1, color='green', zorder=0)
    plt.axvspan(0.99, 1.0, alpha=0.1, color='blue', zorder=0)
    
    plt.tight_layout()
    plt.savefig('comparison_graphs/r2_benchmark.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_dashboard():
    """Create a combined dashboard with multiple plots."""
    plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 3)
    
    # Extract all metrics
    metrics = {}
    models = list(model_data.keys())
    
    # Get all available metrics
    all_metrics = set()
    for model, data in model_data.items():
        all_metrics.update(data.keys())
    
    # Extract data for each metric
    for metric in all_metrics:
        metrics[metric] = []
        for model in models:
            if metric in model_data[model]:
                metrics[metric].append(model_data[model][metric])
            else:
                metrics[metric].append(np.nan)
    
    # Plot 1: MAE and RMSE Comparison (Top Left)
    ax1 = plt.subplot(gs[0, 0])
    width = 0.35
    ind = np.arange(len(models))
    
    mae_bars = ax1.bar(ind - width/2, metrics.get('MAE', [0]*len(models)), width, 
                         label='MAE', color='#FF9999')
    rmse_bars = ax1.bar(ind + width/2, metrics.get('RMSE', [0]*len(models)), width, 
                          label='RMSE', color='#99CCFF')
    
    # Highlight the hybrid model
    hybrid_idx = models.index('Hybrid') if 'Hybrid' in models else -1
    if hybrid_idx >= 0:
        ax1.axvspan(hybrid_idx-0.5, hybrid_idx+0.5, color='#FFECF5', alpha=0.3, zorder=0)
    
    ax1.set_ylabel('Error Value', fontweight='extra bold', fontsize=16)
    ax1.set_title('MAE & RMSE Comparison', fontweight='extra bold', fontsize=18)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: R2 Score (Top Center)
    ax2 = plt.subplot(gs[0, 1])
    colors = [get_model_color(model, 0.5 if model == 'Hybrid' else None) for model in models]
    
    bars = ax2.bar(models, metrics.get('R2', [0]*len(models)), color=colors)
    
    # Add a subtle background for each bar
    for i, bar in enumerate(bars):
        if models[i] == 'Hybrid':
            # Create gradient effect for hybrid
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            
            # Create gradient fill
            grad_colors = [hybrid_cmap(j/20) for j in range(20)]
            for j in range(20):
                seg_width = width / 20
                ax2.bar(models[i], height, width=seg_width, 
                         left=x + j*seg_width, color=grad_colors[j], 
                         edgecolor=None, alpha=0.7)
            
            # Add border
            ax2.bar(models[i], height, width=width, left=x, 
                     color='none', edgecolor='black', linewidth=2)
    
    ax2.set_title('R² Score Comparison', fontweight='extra bold', fontsize=18)
    ax2.set_ylim(0.8, 1.0)
    ax2.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    # Plot 3: Training vs Validation Loss (Top Right)
    ax3 = plt.subplot(gs[0, 2])
    
    x = np.arange(len(models))
    train_loss = []
    val_loss = []
    
    for model in models:
        train_loss.append(model_data[model].get('Training Loss (Final)', 0))
        val_loss.append(model_data[model].get('Validation Loss (Final)', 0))
    
    ax3.plot(x, train_loss, 'o-', color='#FF7F0E', label='Training Loss', linewidth=3, markersize=10)
    ax3.plot(x, val_loss, 's-', color='#1F77B4', label='Validation Loss', linewidth=3, markersize=10)
    
    # Highlight the hybrid model with vertical line
    if hybrid_idx >= 0:
        ax3.axvspan(hybrid_idx-0.2, hybrid_idx+0.2, color='#E6F3FF', alpha=0.5, zorder=0)
        
        # Make hybrid points special
        ax3.plot(hybrid_idx, train_loss[hybrid_idx], '*', color='red', markersize=20, zorder=5)
        ax3.plot(hybrid_idx, val_loss[hybrid_idx], '*', color='blue', markersize=20, zorder=5)
    
    ax3.set_title('Training vs Validation Loss', fontweight='extra bold', fontsize=18)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylabel('Loss Value', fontweight='extra bold', fontsize=16)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Error Distribution (Middle Left & Center Combined)
    ax4 = plt.subplot(gs[1, 0:2])
    
    # Show MAE, RMSE as stacked bars
    metrics_to_plot = ['MAE', 'RMSE']
    bottom = np.zeros(len(models))
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics:
            values = metrics[metric]
            ax4.bar(models, values, bottom=bottom, 
                     label=metric, alpha=0.7)
            bottom += np.array(values)
    
    # Highlight the hybrid model with a subtle background
    if hybrid_idx >= 0:
        ax4.axvspan(hybrid_idx-0.5, hybrid_idx+0.5, color='#F0F9FF', alpha=0.5, zorder=0)
    
    ax4.set_title('Error Metrics Distribution', fontweight='extra bold', fontsize=18)
    ax4.set_ylabel('Error Value', fontweight='extra bold', fontsize=16)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend(loc='upper right')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 5: Performance Ratio (Middle Right)
    ax5 = plt.subplot(gs[1, 2])
    
    # Calculate performance ratio (R2 / MAE) - higher is better
    performance_ratio = []
    for i, model in enumerate(models):
        if 'R2' in model_data[model] and 'MAE' in model_data[model] and model_data[model]['MAE'] != 0:
            ratio = model_data[model]['R2'] / model_data[model]['MAE']
            performance_ratio.append(ratio)
        else:
            performance_ratio.append(0)
    
    # Sort by performance ratio
    sorted_indices = np.argsort(performance_ratio)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ratios = [performance_ratio[i] for i in sorted_indices]
    
    # Create horizontal bars
    bars = ax5.barh(sorted_models, sorted_ratios)
    
    # Color bars
    for i, bar in enumerate(bars):
        model = sorted_models[i]
        if model == 'Hybrid':
            # Create gradient fill for hybrid
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            
            # Create gradient using multiple thin bars
            segments = 20
            for j in range(segments):
                seg_width = width / segments
                seg_x = x + j * seg_width
                color = hybrid_cmap(j / segments)
                ax5.barh([model], [seg_width], height=height, left=seg_x, color=color)
        else:
            bar.set_color(get_model_color(model))
    
    ax5.set_title('Performance Ratio (R² / MAE)', fontweight='extra bold', fontsize=18)
    ax5.set_xlabel('Ratio (higher is better)', fontweight='extra bold', fontsize=16)
    ax5.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot 6: Training vs Validation Loss Comparison (Bottom)
    ax6 = plt.subplot(gs[2, :])
    
    # Create a grouped bar chart
    width = 0.35
    ind = np.arange(len(models))
    
    train_bars = ax6.bar(ind - width/2, [model_data[m].get('Training Loss (Final)', 0) for m in models], 
                         width, label='Training Loss', color='#FF7F0E')
    
    val_bars = ax6.bar(ind + width/2, [model_data[m].get('Validation Loss (Final)', 0) for m in models], 
                       width, label='Validation Loss', color='#1F77B4')
    
    # Apply gradient to hybrid model bars
    if hybrid_idx >= 0:
        # Highlight area
        ax6.axvspan(hybrid_idx-0.5, hybrid_idx+0.5, color='#F0F9FF', alpha=0.3, zorder=0)
        
        # Gradient for hybrid training loss bar
        x = train_bars[hybrid_idx].get_x()
        width = train_bars[hybrid_idx].get_width()
        height = train_bars[hybrid_idx].get_height()
        
        # Remove original bar
        train_bars[hybrid_idx].set_alpha(0)
        
        # Create gradient
        for j in range(20):
            seg_width = width / 20
            color = hybrid_cmap(j/20)
            ax6.bar(ind[hybrid_idx] - width/2 + j*seg_width, height, width=seg_width, 
                     color=color, edgecolor=None, alpha=0.8)
        
        # Gradient for hybrid validation loss bar
        x = val_bars[hybrid_idx].get_x()
        width = val_bars[hybrid_idx].get_width()
        height = val_bars[hybrid_idx].get_height()
        
        # Remove original bar
        val_bars[hybrid_idx].set_alpha(0)
        
        # Create gradient
        for j in range(20):
            seg_width = width / 20
            color = hybrid_cmap(j/20)
            ax6.bar(ind[hybrid_idx] + width/2 + j*seg_width, height, width=seg_width, 
                     color=color, edgecolor=None, alpha=0.8)
    
    # Add value labels
    for i, bars in enumerate([train_bars, val_bars]):
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                         f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax6.set_title('Training vs Validation Loss Comparison', fontweight='extra bold', fontsize=18)
    ax6.set_xticks(ind)
    ax6.set_xticklabels(models)
    ax6.legend()
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    ax6.set_ylabel('Loss Value', fontweight='extra bold', fontsize=16)
    
    # Add overall title
    plt.suptitle('Comprehensive Model Performance Dashboard', fontsize=24, fontweight='extra bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('comparison_graphs/combined_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- EXECUTE VISUALIZATION FUNCTIONS ---
# Generate individual performance bar charts with error handling
execute_viz_function("MAE comparison", lambda: bar_comparison('MAE', 'Mean Absolute Error (MAE) Comparison', 'MAE (lower is better)', 'mae_comparison'))
execute_viz_function("RMSE comparison", lambda: bar_comparison('RMSE', 'Root Mean Square Error (RMSE) Comparison', 'RMSE (lower is better)', 'rmse_comparison'))
execute_viz_function("R2 comparison", lambda: bar_comparison('R2', 'R² Score Comparison', 'R² Score (higher is better)', 'r2_comparison'))

# Generate radar chart
execute_viz_function("Radar chart", radar_chart)

# Generate 3D loss comparison
execute_viz_function("3D loss comparison", loss_comparison_3d)

# Generate heatmap
execute_viz_function("Performance heatmap", performance_heatmap)

# Generate bubble chart
execute_viz_function("Bubble chart", bubble_chart)

# Generate 3D surface plot
execute_viz_function("3D surface plot", create_3d_surface)

# Generate R2 benchmark plot
execute_viz_function("R2 benchmark plot", r2_benchmark_plot)

# Generate combined dashboard
execute_viz_function("Combined dashboard", create_combined_dashboard)

print("Visualization process completed. Check the 'comparison_graphs' folder for generated visualizations.")