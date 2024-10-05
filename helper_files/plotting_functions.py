import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from processing.tf_idf_functions import categorize_direction, calculate_directions, categorize_distance


def get_opponent_df(cp_grid):
    opponent_df = cp_grid[['receiver_x',
    'receiver_y',
    'possession_num',
    'possession_throw',
    'game_quarter',
    'quarter_point',
    'score_diff',
    'times']]
    opponent_df.loc[:, 'receiver_x'] = -opponent_df.loc[:, 'receiver_x']
    # discs turned over in either endzone are brought to the front line
    opponent_df.loc[:, 'receiver_y'] = (120 - opponent_df.loc[:, 'receiver_y']).clip(lower=20, upper=100)
    # the possession number increments, throw counter resets and score differential inverts
    if 'possession_num' in opponent_df.columns:
        opponent_df.loc[:, 'possession_num'] += 1

    if 'possession_throw' in opponent_df.columns:
        opponent_df.loc[:, 'possession_throw'] = 1

    if 'score_diff' in opponent_df.columns:
        opponent_df.loc[:, 'score_diff'] = -opponent_df.loc[:, 'score_diff']
    opponent_df = opponent_df.rename({'receiver_x':'thrower_x', 'receiver_y':'thrower_y'}, axis=1)
    return opponent_df

def generate_fv_grid(x_min=-26.66, x_max=26.67, y_min=0, y_max=120, grid_width=50, grid_height=120, include_interaction=True, default_columns=None):
    if default_columns is None:
        default_columns = {}

    x_values = np.linspace(x_min, x_max, grid_width)
    y_values = np.linspace(y_min, y_max, grid_height)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=['thrower_x', 'thrower_y'])
    
    if include_interaction:
        grid_df['x_squared'] = grid_df['thrower_x'] ** 2
        grid_df['y_squared'] = grid_df['thrower_y'] ** 2
        grid_df['xy_interaction'] = grid_df['thrower_x'] * grid_df['thrower_y']
        grid_df['throw_distance'] = np.sqrt(grid_df['thrower_x']**2 + grid_df['thrower_y']**2)

    for col, default_value in default_columns.items():
        grid_df[col] = default_value
    
    return grid_df

def plot_heatmap(model, grid_df, grid_width, grid_height, title='', scaler=None, sigma=None, ax=None, highlight_point=None, contour_every=0.1):
    X = grid_df if scaler is None else scaler.transform(grid_df)
    if model is not None:
        predictions = model.predict_proba(X)[:, 1]
        predicted_probabilities = predictions.reshape(grid_height, grid_width)
    else:
        predicted_probabilities = X
        
    if sigma is not None:
        predicted_probabilities = gaussian_filter(predicted_probabilities, sigma=sigma)
    
    # Use the provided axis if available, otherwise use the current axis
    if ax is None:
        ax = plt.gca()
    
    # Plot the heatmap on the given axis
    sns.heatmap(predicted_probabilities, cmap="YlGnBu", cbar_kws={'label': f'{title}'}, alpha=0.8, ax=ax)
    
    # Add contour lines
    contours = np.arange(0, 1.1, contour_every)
    contour_plot = ax.contour(predicted_probabilities, levels=contours, colors='black', linestyles='--', linewidths=1, extent=[0, grid_width, 0, grid_height])
    ax.clabel(contour_plot, inline=True, fontsize=8, fmt="%.1f")
    
    # Add field lines
    ax.hlines(y=20, xmin=0, xmax=grid_width, color='black', linestyle='-', linewidth=1)
    ax.hlines(y=100, xmin=0, xmax=grid_width, color='black', linestyle='-', linewidth=1)
    ax.hlines(y=0.1, xmin=0, xmax=grid_width, color='black', linestyle='-', linewidth=1)
    ax.hlines(y=120, xmin=0, xmax=grid_width, color='black', linestyle='-', linewidth=1)
    ax.vlines(x=0, ymin=0, ymax=120, color='black', linestyle='-', linewidth=1)
    ax.vlines(x=49.8, ymin=0, ymax=120, color='black', linestyle='-', linewidth=1)
    
    if highlight_point is not None:
        highlight_x, highlight_y = highlight_point
        ax.plot(highlight_x, highlight_y, 'ko', markersize=6, label="Highlighted Point")
    

    # Set plot labels and title
    ax.set_title(f'{title}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_ylim([0, 120])

    # Set custom ticks for the plot
    num_yticks = 13
    ax.set_yticks(np.linspace(0, 120, num=num_yticks))
    ax.set_yticklabels(np.round(np.linspace(0, 120, num=num_yticks).astype(int), 1))
    ax.set_xticks(np.linspace(0, grid_width, num=10))
    ax.set_xticklabels(np.round(np.linspace(-26.66, 26.67, num=10), 1).astype(int))
    
    ax.set_aspect('equal')

    return predicted_probabilities


def generate_cp_grid(thrower_x, thrower_y, receiver_x_range=(-26.66, 26.67), receiver_y_range=(0, 120), 
                           default_columns=None):
    if default_columns is None:
        default_columns = {
            'times': 0,
            'score_diff': 0,
            'quarter_point': 0,
            'game_quarter': 1,
            'possession_throw': 0,
            'possession_num': 0
        }
    
    receiver_x_values = np.linspace(receiver_x_range[0], receiver_x_range[1], num=50)
    receiver_y_values = np.linspace(receiver_y_range[0], receiver_y_range[1], num=120)
    
    receiver_x, receiver_y = np.meshgrid(receiver_x_values, receiver_y_values)
    
    grid_points = np.c_[receiver_x.ravel(), receiver_y.ravel()]
    
    grid_df = pd.DataFrame(grid_points, columns=['receiver_x', 'receiver_y'])
    grid_df['thrower_x'] = thrower_x
    grid_df['thrower_y'] = thrower_y
    grid_df['throw_distance'] = np.sqrt((grid_df['receiver_x'] - thrower_x) ** 2 + 
                                         (grid_df['receiver_y'] - thrower_y) ** 2)

    # Calculate direction and categorize it
    grid_df = calculate_directions(grid_df)
    grid_df['direction_forward'] = (grid_df['direction'].apply(categorize_direction) == 'forward').astype(int)
    grid_df['direction_sideways'] = (grid_df['direction'].apply(categorize_direction) == 'sideways').astype(int)
    grid_df['direction_backward'] = (grid_df['direction'].apply(categorize_direction) == 'backward').astype(int)


    # Categorize distance
    grid_df = categorize_distance(grid_df)
    grid_df['distance_short'] = grid_df['distance_category'] == 'short'
    grid_df['distance_medium'] = grid_df['distance_category'] == 'medium'
    grid_df['distance_long'] = grid_df['distance_category'] == 'long'
    grid_df = grid_df.drop('distance_category', axis=1)
    # Set default values for other columns
    grid_df['possession_num'] = default_columns['possession_num']
    grid_df['possession_throw'] = default_columns['possession_throw']
    grid_df['game_quarter'] = default_columns['game_quarter']
    grid_df['quarter_point'] = default_columns['quarter_point']
    grid_df['score_diff'] = default_columns['score_diff']
    grid_df['times'] = default_columns['times']
    
    return grid_df