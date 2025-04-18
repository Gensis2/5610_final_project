import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def load_data(file_path):
    return pd.read_csv(file_path)

def accuracy_select(df, label):
    # Initialize the row to be selected
    acc_row = df

    # Select the row with the highest testing accuracy and lowest training loss
    # and match the other columns
    highest_accuracy_row = acc_row.loc[acc_row['testing_acc'].idxmax()]
    lowest_loss_row = acc_row.loc[acc_row['training_loss'].idxmin()]

    # Get the columns to match
    columns_to_match = [col for col in acc_row.columns if col not in [label, 'testing_acc', 'training_loss']]

    # Select the rows that match the highest accuracy and lowest loss
    acc_df = acc_row[
        acc_row[columns_to_match].eq(highest_accuracy_row[columns_to_match]).all(axis=1)
    ]
    loss_df = acc_row[
        acc_row[columns_to_match].eq(lowest_loss_row[columns_to_match]).all(axis=1)
    ]

    return acc_df, loss_df

# def flops_select(df, label):
#     # Initialize the row to be selected
#     flops_row = df

#     # Select the row with the lowest total FLOPS
#     lowest_flops_row = flops_row.loc[flops_row['total_flops'].idxmin()]

#     # Get the columns to match
#     columns_to_match = [col for col in flops_row.columns if col not in [label, 'total_flops', 'conv_flops', 'conv_bn_flops', 'pool_flops', 'fc_flops', 'fc_bn_flops']]

#     # Select the rows that match the lowest flops
#     flops_df = flops_row[
#         flops_row[columns_to_match].eq(lowest_flops_row[columns_to_match]).all(axis=1)
#     ]

#     return flops_df

def snn_flops_select(df, label):
    # Initialize the row to be selected
    flops_row = df

    # Select the row with the lowest total FLOPS
    lowest_flops_row = flops_row.loc[flops_row['total_flops'].idxmax()]

    # Get the columns to match
    columns_to_match = [col for col in flops_row.columns if col not in [label, 'total_flops', 'conv_neuron_flops', 'conv_flops', 'conv_bn_flops', 'pool_flops',
       'fc_neuron_flops', 'fc_flops', 'fc_bn_flops']]

    # Select the rows that match the lowest flops
    flops_df = flops_row[
        flops_row[columns_to_match].eq(lowest_flops_row[columns_to_match]).all(axis=1)
    ]

    return flops_df

def generate_graphs(df, x, y, file_name, output_dir):
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
    if 'testing_acc' in df.columns and 'training_loss' in df.columns:
        df = df.drop(columns=['testing_acc', 'training_loss'])

    # Check if the DataFrame contains any of the SNN FLOPS columns
    snn_flops_columns = ['total_flops', 'conv_neuron_flops', 'conv_flops', 'conv_bn_flops', 'pool_flops',
       'fc_neuron_flops', 'fc_flops', 'fc_bn_flops', 'normalized_total_flops']
    has_snn_flops_columns = any(col in df.columns for col in snn_flops_columns)

    # Drop the CNN FLOPS columns if they exist
    if has_snn_flops_columns:
        df = df.drop(columns=snn_flops_columns)
    
    constant_values = df.iloc[0].to_dict()  # Get the first row as a dictionary
    constant_values_str = ', '.join([f"{key}: {value}" for key, value in constant_values.items()])  # Create a string representation of the constant values
    constant_values_str = constant_values_str.replace('_', ' ')  # Replace '_' with ' '

    # Example: Generate a simple line graph
    plt.figure(figsize=(10, 5))
    line = plt.plot(range(len(x_data)), y_data, marker='o', label=constant_values_str)  # Use range(len(x_data)) for consistent spacing

    match x:
        case 'num_epochs':
            x_label = 'Epochs'

        case 'batch_size':
            x_label = 'Batch Size'

        case 'lr':
            x_label = 'Learning Rate'

        case 'leak_mem':
            x_label = 'Leakage Memory'
        
        case 'num_steps':
            x_label = 'Number of Time Steps'

    match y:
        case 'testing_acc':
            y_label = 'Testing Accuracy'

        case 'training_loss':
            y_label = 'Training Loss'

        case 'normalized_total_flops':
            y_label = 'Normalized FLOPS'

    plt.xticks(ticks=range(len(x_data)), labels=x_data, rotation=45)  # Ensure xticks match x_data
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs. {x_label}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='graphs', help='Directory to save the generated graphs')

    args = parser.parse_args()

    cnn_result_data = load_data('data_processing\cnn_results.csv')
    # snn_result_data = load_data('data_processing\snn_results.csv')
    cnn_flops_data = load_data('data_processing\cnn_flops.csv')
    snn_flops_data = load_data('data_processing\snn_flops.csv')

    labels = cnn_result_data.columns.tolist()
    for label in labels:
        if label not in ['testing_acc', 'training_loss']:
            acc_df, loss_df = accuracy_select(cnn_result_data, label)  # Pass the loaded data to the function

            generate_graphs(acc_df, f'{label}', 'testing_acc', f'{label}_testing_acc_cnn', args.output)  # Pass the loaded data to the function
            generate_graphs(loss_df, f'{label}', 'training_loss', f'{label}_training_loss_cnn', args.output)  # Pass the loaded data to the function

    labels = snn_flops_data.columns.tolist()
    for label in labels:
        if label not in ['total_flops', 'conv_neuron_flops', 'conv_flops', 'conv_bn_flops', 'pool_flops',
       'fc_neuron_flops', 'fc_flops', 'fc_bn_flops', 'batch_size', 'num_epochs']:
            
            # Select the row with the lowest total FLOPS and match the other columns
            snn_total_flops_df = snn_flops_select(snn_flops_data, label)  # Pass the loaded data to the function
            snn_total_flops_df = snn_total_flops_df.sort_values(by=label, ascending=True)  # Sort the dataframe by label in ascending order

            # add a new column to store normalized total_flops value from snn by cnn_total_flops_df
            # first need to get the total_flops value from cnn_total_flops_df using 
            # the batch_size and num_epochs values from snn_total_flops_df
            # this is done by filtering the cnn_flops_data dataframe using the batch_size and num_epochs values from snn_total_flops_df
            cnn_total_flops_df = cnn_flops_data[(cnn_flops_data['batch_size'] == snn_total_flops_df['batch_size'].values[0]) & (cnn_flops_data['num_epochs'] == snn_total_flops_df['num_epochs'].values[0])]

            # check if the cnn_total_flops_df is empty
            if cnn_total_flops_df.empty:
                print(f'No matching cnn_total_flops_df found for batch_size: {snn_total_flops_df["batch_size"].values[0]} and num_epochs: {snn_total_flops_df["num_epochs"].values[0]}')
                continue

            # get the total_flops value from cnn_total_flops_df
            cnn_total_flops = cnn_total_flops_df['total_flops'].values[0]
            
            # add new column to snn_total_flops_df
            snn_total_flops_df['normalized_total_flops'] = snn_total_flops_df['total_flops'] / cnn_total_flops
            snn_total_flops_df = snn_total_flops_df.sort_values(by=label, ascending=True)

            generate_graphs(snn_total_flops_df, label, 'normalized_total_flops', f'{label}_normalized_total_flops_snn', args.output)  # Pass the loaded data to the function






