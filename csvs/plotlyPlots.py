import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import traceback
import sys
import plotly.io as pio

# Create output directory for graphs
os.makedirs('plotly_graphs', exist_ok=True)

# Set up a white‐background template with larger, black axis/text fonts
pio.templates["custom_white"] = pio.templates["plotly_white"]
pio.templates["custom_white"].layout.update(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(size=18, color="black", family="Arial Black"),
    xaxis=dict(
        title_font=dict(size=28, color="black", family="Arial Black"), 
        tickfont=dict(size=24, color="black", family="Arial Black")
    ),
    yaxis=dict(
        title_font=dict(size=28, color="black", family="Arial Black"), 
        tickfont=dict(size=24, color="black", family="Arial Black")
    ),
    scene=dict(
        xaxis=dict(
            title_font=dict(size=28, color="black", family="Arial Black"),
            tickfont=dict(size=24, color="black", family="Arial Black")
        ),
        yaxis=dict(
            title_font=dict(size=28, color="black", family="Arial Black"),
            tickfont=dict(size=24, color="black", family="Arial Black")
        ),
        zaxis=dict(
            title_font=dict(size=28, color="black", family="Arial Black"),
            tickfont=dict(size=24, color="black", family="Arial Black")
        )
    )
)
pio.templates.default = "custom_white"

# Define colors for each model
model_colors = {
    'Dense': '#FF6B6B',     # Red
    'Linear': '#4ECDC4',    # Teal
    'MLP': '#FFD166',       # Yellow
    'TDNN': '#6A0572',      # Purple
    'Hybrid': '#FF69B4'     # Pink (default, will use gradient)
}

# Define a gradient for hybrid model
hybrid_colors = ['#FF69B4', '#1E90FF']  # Pink to Blue gradient

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

# Helper function for gradient colors
def get_gradient_color(normalized_pos, gradient_colors):
    """Generate a color from a gradient based on position (0 to 1)"""
    pos = min(max(normalized_pos, 0), 1)  # Ensure position is between 0 and 1
    
    # For simplicity with just two colors
    start_color = gradient_colors[0]
    end_color = gradient_colors[1]
    
    # Parse hex colors to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    
    # Interpolate
    result_rgb = tuple(int(start_rgb[i] + pos * (end_rgb[i] - start_rgb[i])) for i in range(3))
    
    # Convert back to hex
    return f'rgb({result_rgb[0]}, {result_rgb[1]}, {result_rgb[2]})'

def get_model_color(model_name, position=None):
    """Get color for model with special gradient for hybrid model."""
    if model_name == 'Hybrid' and position is not None:
        return get_gradient_color(position, hybrid_colors)
    
    return model_colors.get(model_name, '#333333')  # Default dark gray

# Function to ensure consistent axis styling across all plots
def set_axis_styling(fig):
    """Apply consistent bold and large axis styling to any figure"""
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].update(
                title_font=dict(size=28, color="black", family="Arial Black"),
                tickfont=dict(size=24, color="black", family="Arial Black")
            )
    
    # Handle 3D plots
    if hasattr(fig.layout, 'scene'):
        for axis in ['xaxis', 'yaxis', 'zaxis']:
            if hasattr(fig.layout.scene, axis):
                getattr(fig.layout.scene, axis).update(
                    title_font=dict(size=28, color="black", family="Arial Black"),
                    tickfont=dict(size=24, color="black", family="Arial Black")
                )
    return fig

def execute_viz_function(func_name, function):
    """Execute a visualization function with proper error handling."""
    print(f"Generating {func_name}...")
    try:
        result = function()
        if isinstance(result, go.Figure):
            result = set_axis_styling(result)
            result.write_image(f'plotly_graphs/{func_name.replace(" ", "_")}.png')
            result.write_html(f'plotly_graphs/{func_name.replace(" ", "_")}.html')
        print(f"Successfully generated {func_name}")
    except Exception as e:
        print(f"Error generating {func_name}: {str(e)}")
        traceback.print_exc()

# --- VISUALIZATION FUNCTIONS ---

def bar_comparison(metric, title, ylabel, filename):
    """Create a bar chart comparing models on a specific metric."""
    values = []
    labels = []
    colors = []
    hybrid_idx = -1
    
    for i, (model_name, metrics) in enumerate(model_data.items()):
        if metric in metrics:
            values.append(metrics[metric])
            labels.append(model_name)
            colors.append(get_model_color(model_name))
            if model_name == 'Hybrid':
                hybrid_idx = i
    
    if not values:
        print(f"No data available for metric: {metric}")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for each model
    for i, (model, value) in enumerate(zip(labels, values)):
        if model == 'Hybrid':
            # Create a gradient for hybrid model
            fig.add_trace(go.Bar(
                x=[model], 
                y=[value],
                marker=dict(
                    color=get_gradient_color(0.5, hybrid_colors),
                    line=dict(color='black', width=2)
                ),
                name=model,
                text=[f"{value:.4f}"],
                textposition='outside',
                textfont=dict(size=14, color='black', family="Arial Black")
            ))
        else:
            fig.add_trace(go.Bar(
                x=[model], 
                y=[value],
                marker_color=colors[i],
                name=model,
                text=[f"{value:.4f}"],
                textposition='outside',
                textfont=dict(size=14, color='black', family="Arial Black")
            ))
    
    # Highlight the hybrid model with a rectangle if it exists
    if hybrid_idx >= 0:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=hybrid_idx - 0.4,
            y0=0,
            x1=hybrid_idx + 0.4,
            y1=1,
            fillcolor="lightcyan",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    
    # Update layout with styling
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, family="Arial Black"),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text='Model',
                font=dict(size=28, family="Arial Black")
            ),
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            title=dict(
                text=ylabel,
                font=dict(size=28, family="Arial Black")
            ),
            tickfont=dict(size=24),
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        showlegend=False,
        height=800,
        width=1200
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image(f'plotly_graphs/{filename}.png')
    fig.write_html(f'plotly_graphs/{filename}.html')
    
    return fig

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
    
    # Create radar chart
    fig = go.Figure()
    
    # Add a trace for each model
    for model, values in normalized_values.items():
        # Connect back to the start to close the polygon
        theta_values = metrics + [metrics[0]]
        r_values = values + [values[0]]
        
        if model == 'Hybrid':
            # Create a special gradient-like effect for hybrid
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=model,
                line=dict(color='black', width=3),
                fillcolor='rgba(255, 105, 180, 0.2)'  # Light pink for hybrid
            ))
        else:
            color = get_model_color(model)
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name=model,
                line=dict(color=color, width=2),
                fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'
            ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=24),
                gridcolor='lightgray'
            )
        ),
        title=dict(
            text='Model Performance Comparison (Radar Chart)',
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        legend=dict(
            font=dict(size=14),
            bordercolor="Black",
            borderwidth=1
        ),
        height=900,
        width=1200
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/radar_comparison.png')
    fig.write_html('plotly_graphs/radar_comparison.html')
    
    return fig

def loss_comparison_3d():
    """Create a 3D plot comparing training and validation loss."""
    x = []
    training_losses = []
    validation_losses = []
    model_names = []
    
    for model_name, metrics in model_data.items():
        if 'Training Loss (Final)' in metrics and 'Validation Loss (Final)' in metrics:
            training_losses.append(metrics['Training Loss (Final)'])
            validation_losses.append(metrics['Validation Loss (Final)'])
            model_names.append(model_name)
            x.append(model_name)
    
    # Create 3D bar plot
    fig = go.Figure()
    
    # Add training loss bars
    for i, (model, train_loss) in enumerate(zip(model_names, training_losses)):
        if model == 'Hybrid':
            # Gradient for hybrid model
            fig.add_trace(go.Bar3d(
                x=[i, i], y=[0, 0], z=[0, train_loss],
                marker=dict(color=get_gradient_color(0.3, hybrid_colors)),
                name=f"{model} (Training)",
                hovertemplate="<b>%{x}</b><br>Training Loss: %{z}<extra></extra>"
            ))
        else:
            color = get_model_color(model)
            fig.add_trace(go.Bar3d(
                x=[i, i], y=[0, 0], z=[0, train_loss],
                marker=dict(color=color),
                name=f"{model} (Training)",
                hovertemplate="<b>%{x}</b><br>Training Loss: %{z}<extra></extra>"
            ))
    
    # Add validation loss bars
    for i, (model, val_loss) in enumerate(zip(model_names, validation_losses)):
        if model == 'Hybrid':
            # Gradient for hybrid model
            fig.add_trace(go.Bar3d(
                x=[i, i], y=[1, 1], z=[0, val_loss],
                marker=dict(color=get_gradient_color(0.7, hybrid_colors)),
                name=f"{model} (Validation)",
                hovertemplate="<b>%{x}</b><br>Validation Loss: %{z}<extra></extra>"
            ))
        else:
            color = get_model_color(model)
            # Add some transparency
            color_rgba = f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}'
            fig.add_trace(go.Bar3d(
                x=[i, i], y=[1, 1], z=[0, val_loss],
                marker=dict(color=color_rgba),
                name=f"{model} (Validation)",
                hovertemplate="<b>%{x}</b><br>Validation Loss: %{z}<extra></extra>"
            ))
    
    # Update layout for better appearance
    fig.update_layout(
        title=dict(
            text="Training vs Validation Loss Comparison",
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text="Model", font=dict(size=28, family="Arial Black")),
                ticktext=model_names,
                tickvals=list(range(len(model_names))),
                tickfont=dict(size=24)
            ),
            yaxis=dict(
                title=dict(text="Loss Type", font=dict(size=28, family="Arial Black")),
                ticktext=["Training", "Validation"],
                tickvals=[0, 1],
                tickfont=dict(size=24)
            ),
            zaxis=dict(
                title=dict(text="Loss Value", font=dict(size=28, family="Arial Black")),
                tickfont=dict(size=24)
            )
        ),
        scene_camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1)
        ),
        height=800,
        width=1200
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/loss_comparison_3d.png')
    fig.write_html('plotly_graphs/loss_comparison_3d.html')
    
    return fig

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
    
    # For visualization purposes, we want lower values for MAE, RMSE, losses to show as "better"
    # But higher values for R2 to show as "better". Set up normalization accordingly.
    normalized_data = np.copy(data_array)
    for i, metric in enumerate(metrics_to_plot):
        if metric != 'R2':  # For metrics where lower is better
            # Check for NaN values
            valid_indices = ~np.isnan(data_array[:, i])
            if np.any(valid_indices):
                max_val = np.max(data_array[valid_indices, i])
                min_val = np.min(data_array[valid_indices, i])
                if max_val != min_val:  # Avoid division by zero
                    normalized_data[valid_indices, i] = 1 - (data_array[valid_indices, i] - min_val) / (max_val - min_val)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=normalized_data,
        x=metrics_to_plot,
        y=model_names,
        colorscale='RdYlGn',
        text=[[f"{val:.4f}" if not np.isnan(val) else "N/A" for val in row] for row in data_array],
        texttemplate="%{text}",
        textfont={"size":14}
    ))
    
    # Update layout for better appearance
    fig.update_layout(
        title=dict(
            text="Model Performance Metrics Heatmap",
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Metrics", font=dict(size=28, family="Arial Black")),
            tickangle=30,
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            title=dict(text="Models", font=dict(size=28, family="Arial Black")),
            tickfont=dict(size=24)
        ),
        height=800,
        width=1200
    )
    
    # Highlight the hybrid model row
    if 'Hybrid' in model_names:
        hybrid_idx = model_names.index('Hybrid')
        for i in range(len(metrics_to_plot)):
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=i-0.5,
                y0=hybrid_idx-0.5,
                x1=i+0.5,
                y1=hybrid_idx+0.5,
                line=dict(color="blue", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/performance_heatmap.png')
    fig.write_html('plotly_graphs/performance_heatmap.html')
    
    return fig

def bubble_chart():
    """Create a bubble chart showing the relationship between metrics with size representing R2."""
    if not all(metric in model_data[model] for model in model_data for metric in ['MAE', 'RMSE', 'R2']):
        print("Missing required metrics for bubble chart")
        return
    
    fig = go.Figure()
    
    # Add bubbles for each model
    for model_name, metrics in model_data.items():
        x = metrics['MAE']
        y = metrics['RMSE']
        size = metrics['R2'] * 100  # Scale R2 for better visibility
        
        if model_name == 'Hybrid':
            # Special handling for hybrid model
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=size,
                    color=get_gradient_color(0.5, hybrid_colors),
                    line=dict(color='black', width=2),
                    opacity=0.8
                ),
                name=model_name,
                text=[f"{model_name}<br>MAE: {x:.4f}<br>RMSE: {y:.4f}<br>R²: {metrics['R2']:.4f}"],
                hoverinfo="text"
            ))
            
            # Add a highlight ring around the hybrid bubble
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=size+20,
                    color='rgba(173, 216, 230, 0.2)',
                    line=dict(color='rgba(0,0,0,0)')
                ),
                name='',
                hoverinfo='skip',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=size,
                    color=get_model_color(model_name),
                    line=dict(color='black', width=1),
                    opacity=0.7
                ),
                name=model_name,
                text=[f"{model_name}<br>MAE: {x:.4f}<br>RMSE: {y:.4f}<br>R²: {metrics['R2']:.4f}"],
                hoverinfo="text"
            ))
    
    # Add model name labels to each bubble
    for model_name, metrics in model_data.items():
        fig.add_trace(go.Scatter(
            x=[metrics['MAE']],
            y=[metrics['RMSE']],
            mode='text',
            text=[model_name],
            textfont=dict(
                color='black',
                size=14,
                family='Arial Black'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add legend for bubble sizes
    sizes = [0.85, 0.90, 0.95, 0.99]
    for s in sizes:
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=s*100,
                color='gray',
                opacity=0.7,
                line=dict(color='black', width=1)
            ),
            name=f'R² = {s}'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Model Performance Bubble Chart',
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text='Mean Absolute Error (MAE)',
                font=dict(size=28, family="Arial Black")
            ),
            gridcolor='lightgray',
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            title=dict(
                text='Root Mean Square Error (RMSE)',
                font=dict(size=28, family="Arial Black")
            ),
            gridcolor='lightgray',
            scaleanchor="x",
            scaleratio=1,
            tickfont=dict(size=24)
        ),
        height=900,
        width=1200,
        legend=dict(
            title="Bubble Size Legend",
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/bubble_chart.png')
    fig.write_html('plotly_graphs/bubble_chart.html')
    
    return fig

def create_3d_surface():
    """Create a 3D surface plot showing the relationship between MAE, RMSE, and R2."""
    if not all('MAE' in model_data[model] and 'RMSE' in model_data[model] and 'R2' in model_data[model] 
               for model in model_data):
        print("Missing required metrics for 3D surface plot")
        return
    
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
    
    # Create figure
    fig = go.Figure()
    
    # Add theoretical surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.7,
        name='Theoretical Surface',
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Theoretical R² Value",
                font=dict(size=16, family="Arial Black")
            ),
            tickfont=dict(size=14)
        )
    ))
    
    # Add model points
    for i, model in enumerate(model_data):
        if model == 'Hybrid':
            # Special marker for hybrid model
            fig.add_trace(go.Scatter3d(
                x=[mae_values[i]],
                y=[rmse_values[i]],
                z=[r2_values[i]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=get_gradient_color(0.5, hybrid_colors),
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                name=model,
                text=[f"{model}<br>MAE: {mae_values[i]:.4f}<br>RMSE: {rmse_values[i]:.4f}<br>R²: {r2_values[i]:.4f}"],
                hoverinfo="text"
            ))
            
            # Add vertical line to surface
            fig.add_trace(go.Scatter3d(
                x=[mae_values[i], mae_values[i]],
                y=[rmse_values[i], rmse_values[i]],
                z=[0, r2_values[i]],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=[mae_values[i]],
                y=[rmse_values[i]],
                z=[r2_values[i]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=get_model_color(model),
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                name=model,
                text=[f"{model}<br>MAE: {mae_values[i]:.4f}<br>RMSE: {rmse_values[i]:.4f}<br>R²: {r2_values[i]:.4f}"],
                hoverinfo="text"
            ))
            
            # Add vertical line to surface
            fig.add_trace(go.Scatter3d(
                x=[mae_values[i], mae_values[i]],
                y=[rmse_values[i], rmse_values[i]],
                z=[0, r2_values[i]],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='3D Performance Surface: MAE vs RMSE vs R²',
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text='Mean Absolute Error (MAE)',
                    font=dict(size=28, family="Arial Black")
                ),
                gridcolor='lightgray',
                tickfont=dict(size=24)
            ),
            yaxis=dict(
                title=dict(
                    text='Root Mean Square Error (RMSE)',
                    font=dict(size=28, family="Arial Black")
                ),
                gridcolor='lightgray',
                tickfont=dict(size=24)
            ),
            zaxis=dict(
                title=dict(
                    text='R² Score',
                    font=dict(size=28, family="Arial Black")
                ),
                gridcolor='lightgray',
                tickfont=dict(size=24)
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1)
            )
        ),
        height=900,
        width=1200
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/3d_performance_surface.png')
    fig.write_html('plotly_graphs/3d_performance_surface.html')
    
    return fig

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
    
    fig = go.Figure()
    
    # Add bars for each model
    for i, (model, r2) in enumerate(zip(models, r2_scores)):
        if model == 'Hybrid':
            # Create gradient for hybrid
            fig.add_trace(go.Bar(
                y=[model],
                x=[r2],
                orientation='h',
                marker=dict(
                    color=get_gradient_color(0.5, hybrid_colors),
                    line=dict(color='black', width=2)
                ),
                name=model,
                text=[f"{r2:.4f}"],
                textposition='outside',
                textfont=dict(size=14, color='black', family="Arial Black")
            ))
        else:
            fig.add_trace(go.Bar(
                y=[model],
                x=[r2],
                orientation='h',
                marker_color=get_model_color(model),
                name=model,
                text=[f"{r2:.4f}"],
                textposition='outside',
                textfont=dict(size=14, color='black', family="Arial Black")
            ))
    
    # Add benchmark lines
    fig.add_shape(
        type='line',
        x0=0.9, y0=-0.5,
        x1=0.9, y1=len(models)-0.5,
        line=dict(color='red', dash='dash', width=2),
        name='Good (R² ≥ 0.9)'
    )
    
    fig.add_shape(
        type='line',
        x0=0.95, y0=-0.5,
        x1=0.95, y1=len(models)-0.5,
        line=dict(color='green', dash='dash', width=2),
        name='Excellent (R² ≥ 0.95)'
    )
    
    fig.add_shape(
        type='line',
        x0=0.99, y0=-0.5,
        x1=0.99, y1=len(models)-0.5,
        line=dict(color='blue', dash='dash', width=2),
        name='Outstanding (R² ≥ 0.99)'
    )
    
    # Add colored background zones
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=0, x1=0.9,
        y0=0, y1=1,
        fillcolor="gray",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=0.9, x1=0.95,
        y0=0, y1=1,
        fillcolor="red",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=0.95, x1=0.99,
        y0=0, y1=1,
        fillcolor="green",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    fig.add_shape(
        type="rect",
        xref="x", yref="paper",
        x0=0.99, x1=1.0,
        y0=0, y1=1,
        fillcolor="blue",
        opacity=0.1,
        layer="below",
        line_width=0
    )
    
    # Add annotations for benchmark categories
    fig.add_annotation(
        x=0.895, y=1.05,
        xref="x", yref="paper",
        text="Good",
        showarrow=False,
        font=dict(size=14, color="red"),
        align="right"
    )
    
    fig.add_annotation(
        x=0.945, y=1.05,
        xref="x", yref="paper",
        text="Excellent",
        showarrow=False,
        font=dict(size=14, color="green"),
        align="right"
    )
    
    fig.add_annotation(
        x=0.985, y=1.05,
        xref="x", yref="paper",
        text="Outstanding",
        showarrow=False,
        font=dict(size=14, color="blue"),
        align="right"
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='R² Score Benchmark Comparison',
            font=dict(size=22, family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text='R² Score',
                font=dict(size=28, family="Arial Black")
            ),
            range=[0.8, 1.05],
            gridcolor='lightgray',
            tickfont=dict(size=24)
        ),
        yaxis=dict(
            title=dict(
                text='Model',
                font=dict(size=28, family="Arial Black")
            ),
            tickfont=dict(size=24)
        ),
        height=800,
        width=1200,
        showlegend=False
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/r2_benchmark.png')
    fig.write_html('plotly_graphs/r2_benchmark.html')
    
    return fig

def create_combined_dashboard():
    """Create a combined dashboard with multiple plots."""
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
    
    # Find hybrid model index
    hybrid_idx = models.index('Hybrid') if 'Hybrid' in models else -1
    
    # Create subplot grid
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'MAE & RMSE Comparison', 'R² Score Comparison', 'Training vs Validation Loss',
            'Error Metrics Distribution', 'Performance Ratio (R² / MAE)', '',
            'Training vs Validation Loss Comparison', '', ''
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar", "colspan": 2}, {"type": "bar"}, None],
            [{"type": "bar", "colspan": 3}, None, None]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Plot 1: MAE and RMSE Comparison (Top Left)
    for i, model in enumerate(models):
        if 'MAE' in model_data[model]:
            fig.add_trace(
                go.Bar(
                    x=[model], 
                    y=[model_data[model]['MAE']], 
                    name='MAE',
                    marker_color='#FF9999',
                    text=[f"{model_data[model]['MAE']:.4f}"],
                    textposition='outside',
                    showlegend=i==0
                ),
                row=1, col=1
            )
        
        if 'RMSE' in model_data[model]:
            fig.add_trace(
                go.Bar(
                    x=[model], 
                    y=[model_data[model]['RMSE']], 
                    name='RMSE',
                    marker_color='#99CCFF',
                    text=[f"{model_data[model]['RMSE']:.4f}"],
                    textposition='outside',
                    showlegend=i==0
                ),
                row=1, col=1
            )
    
    # Plot 2: R2 Score (Top Center)
    for i, model in enumerate(models):
        if 'R2' in model_data[model]:
            if model == 'Hybrid':
                fig.add_trace(
                    go.Bar(
                        x=[model],
                        y=[model_data[model]['R2']],
                        name=model,
                        marker=dict(
                            color=get_gradient_color(0.5, hybrid_colors),
                            line=dict(color='black', width=2)
                        ),
                        text=[f"{model_data[model]['R2']:.4f}"],
                        textposition='outside',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Bar(
                        x=[model],
                        y=[model_data[model]['R2']],
                        name=model,
                        marker_color=get_model_color(model),
                        text=[f"{model_data[model]['R2']:.4f}"],
                        textposition='outside',
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # Plot 3: Training vs Validation Loss (Top Right)
    train_loss = []
    val_loss = []
    
    for model in models:
        train_loss.append(model_data[model].get('Training Loss (Final)', 0))
        val_loss.append(model_data[model].get('Validation Loss (Final)', 0))
    
    fig.add_trace(
        go.Scatter(
            x=models,
            y=train_loss,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#FF7F0E', width=3),
            marker=dict(size=10)
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=models,
            y=val_loss,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#1F77B4', width=3),
            marker=dict(size=10, symbol='square')
        ),
        row=1, col=3
    )
    
    # Highlight hybrid points with special markers
    if hybrid_idx >= 0:
        fig.add_trace(
            go.Scatter(
                x=[models[hybrid_idx]],
                y=[train_loss[hybrid_idx]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='red'
                ),
                showlegend=False
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Scatter(
                x=[models[hybrid_idx]],
                y=[val_loss[hybrid_idx]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='blue'
                ),
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Plot 4: Error Distribution (Middle Left & Center Combined)
    metrics_to_plot = ['MAE', 'RMSE']
    colors = ['#FF9999', '#99CCFF']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics:
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=metrics[metric],
                    name=metric,
                    marker_color=colors[i],
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # Plot 5: Performance Ratio (Middle Right)
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
    
    # Add bars
    for i, (model, ratio) in enumerate(zip(sorted_models, sorted_ratios)):
        if model == 'Hybrid':
            fig.add_trace(
                go.Bar(
                    y=[model],
                    x=[ratio],
                    orientation='h',
                    marker=dict(
                        color=get_gradient_color(0.5, hybrid_colors),
                        line=dict(color='black', width=2)
                    ),
                    name=model,
                    showlegend=False,
                    text=[f"{ratio:.4f}"],
                    textposition='outside'
                ),
                row=2, col=3
            )
        else:
            fig.add_trace(
                go.Bar(
                    y=[model],
                    x=[ratio],
                    orientation='h',
                    marker_color=get_model_color(model),
                    name=model,
                    showlegend=False,
                    text=[f"{ratio:.4f}"],
                    textposition='outside'
                ),
                row=2, col=3
            )
    
    # Plot 6: Training vs Validation Loss Comparison (Bottom)
    for i, model in enumerate(models):
        train_val_loss = [
            model_data[model].get('Training Loss (Final)', 0),
            model_data[model].get('Validation Loss (Final)', 0)
        ]
        
        if model == 'Hybrid':
            fig.add_trace(
                go.Bar(
                    x=[f"{model} (Training)", f"{model} (Validation)"],
                    y=train_val_loss,
                    marker=dict(
                        color=[
                            get_gradient_color(0.3, hybrid_colors),
                            get_gradient_color(0.7, hybrid_colors)
                        ],
                        line=dict(color='black', width=2)
                    ),
                    name=model,
                    text=train_val_loss,
                    textposition='outside',
                    showlegend=False
                ),
                row=3, col=1
            )
        else:
            color = get_model_color(model)
            fig.add_trace(
                go.Bar(
                    x=[f"{model} (Training)", f"{model} (Validation)"],
                    y=train_val_loss,
                    marker_color=color,
                    name=model,
                    text=train_val_loss,
                    textposition='outside',
                    showlegend=False
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Comprehensive Model Performance Dashboard',
            font=dict(size=24, family="Arial Black"),
            x=0.5,
            y=0.98
        ),
        height=1800,
        width=1600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
    )
    
    # Update axes titles with bold formatting
    fig.update_xaxes(
        title_text="Model",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Error Value",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Model",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text="R² Score",
        title_font=dict(size=28, family="Arial Black"),
        range=[0.8, 1.0],
        tickfont=dict(size=24),
        row=1, col=2
    )
    
    fig.update_xaxes(
        title_text="Model",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=1, col=3
    )
    
    fig.update_yaxes(
        title_text="Loss Value",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=1, col=3
    )
    
    fig.update_xaxes(
        title_text="Model",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Error Value",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=2, col=1
    )
    
    fig.update_xaxes(
        title_text="Ratio (higher is better)",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=2, col=3
    )
    
    fig.update_yaxes(
        title_text="Model",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=2, col=3
    )
    
    fig.update_xaxes(
        title_text="Model and Loss Type",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=3, col=1
    )
    
    fig.update_yaxes(
        title_text="Loss Value",
        title_font=dict(size=28, family="Arial Black"),
        tickfont=dict(size=24),
        row=3, col=1
    )
    
    # Apply consistent styling
    fig = set_axis_styling(fig)
    
    # Save the figure
    fig.write_image('plotly_graphs/combined_dashboard.png')
    fig.write_html('plotly_graphs/combined_dashboard.html')
    
    return fig

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

print("Visualization process completed. Check the 'plotly_graphs' folder for generated visualizations.")
