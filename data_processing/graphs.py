import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def performance_select(df, label):
    # Initialize the row to be selected
    acc_row = df

    # Select the row with the highest testing accuracy and lowest training loss
    # and match the other columns
    highest_accuracy_row = acc_row.loc[acc_row['testing_accuracy'].idxmax()]
    lowest_loss_row = acc_row.loc[acc_row['training_loss'].idxmin()]

    # Get the columns to match
    columns_to_match = [col for col in acc_row.columns if col not in [label, 'testing_accuracy', 'training_loss']]

    # Select the rows that match the highest accuracy and lowest loss
    acc_df = acc_row[
        acc_row[columns_to_match].eq(highest_accuracy_row[columns_to_match]).all(axis=1)
    ]
    loss_df = acc_row[
        acc_row[columns_to_match].eq(lowest_loss_row[columns_to_match]).all(axis=1)
    ]


    return acc_df, loss_df

def efficiency_select(df, label):
    # Initialize the row to be selected
    flops_row = df
    # Select the row with the lowest total FLOPS
    lowest_flops_row = flops_row.loc[flops_row['flops'].idxmax()]

    # Get the columns to match
    columns_to_match = [col for col in flops_row.columns if col not in [label, 'flops', 'training_loss', 'testing_accuracy']]

    # Select the rows that match the lowest flops
    if label not in ['training_loss', 'testing_accuracy']:  
        flops_df = flops_row[
            flops_row[columns_to_match].eq(lowest_flops_row[columns_to_match]).all(axis=1)
        ]
    else:
        if label == 'training_loss':
            flops_df = flops_row.drop(columns=['testing_accuracy'])
        elif label == 'testing_accuracy':
            flops_df = flops_row.drop(columns=['training_loss'])

    return flops_df

def plot_graphs(df, x, y, file_name, output_dir, mark_xlabels=True, point=None):
    '''
    Generates a graph from the given DataFrame and saves it to the specified directory.
    df : DataFrame to plot
    x : x-axis column name
    y : y-axis column name
    title : Title of the graph
    file_name : Name of the file to save the graph
    output_dir : Directory to save the graph

    '''

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x_data = df[x]
    y_data = df[y]

    # Get retrieve the values from the columns that are held constant
    df = df.drop(columns=[x])  # Drop the columns that are not needed
    if 'testing_accuracy' in df.columns and 'training_loss' in df.columns:
        df = df.drop(columns=['testing_accuracy', 'training_loss'])

    # Check if the DataFrame contains any of the SNN FLOPS columns
    snn_flops_columns = ['flops']
    has_snn_flops_columns = any(col in df.columns for col in snn_flops_columns)

    # Drop the CNN FLOPS columns if they exist
    if has_snn_flops_columns:
        df = df.drop(columns=snn_flops_columns)
    
    constant_values = df.iloc[0].to_dict()  # Get the first row as a dictionary
    constant_values_str = ', '.join([f"{key}: {value}" for key, value in constant_values.items()])  # Create a string representation of the constant values
    constant_values_str = constant_values_str.replace('_', ' ')  # Replace '_' with ' '

    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices] 
 
    # Example: Generate a simple line graph
    plt.figure(figsize=(5.7, 3.5))
    if point is None:
        if mark_xlabels:
            line = plt.plot(range(len(x_data)), y_data, marker='o', label=constant_values_str)  # Use range(len(x_data)) for consistent spacing
        else:
            line = plt.plot((x_data), y_data, marker='o', label=constant_values_str)  # Use range(len(x_data)) for consistent spacing

    if point is not None:
        point_x = point[x]
        point_y = point[y]
        plt.scatter(x_data, y_data, color='blue', label=constant_values_str)  # Use the same color for the line and points
        plt.scatter(point_x.to_numpy(), point_y.to_numpy(), color='red', label='Baseline', zorder=5)  # Add the point with a label

    x_label = 'FLOPS' if x == 'flops' else \
        'Leakage Voltage' if x=='leak_mem' else \
        'Epochs' if x == 'num_epochs' else \
        'Learning Rate' if x == 'lr' else \
        'Timesteps' if x == 'num_steps' else \
        x.replace('_', ' ').title()
    y_label = y.replace('_', ' ').title()

    if mark_xlabels:
        plt.xticks(ticks=range(len(x_data)), labels=x_data)  # Ensure xticks match x_data
        plt.xlabel(x_label)


 # Manually set x-axis ticks

    plt.ylabel(y_label)
    plt.title(f'{y_label} vs. {x_label}')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=1, frameon=True)  # Place legend at the top
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}.png'), bbox_inches='tight')  # Remove extra whitespace
    plt.close()