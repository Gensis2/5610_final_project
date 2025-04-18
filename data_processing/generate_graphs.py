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

    # Example: Generate a simple line graph
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, y_data)

    match x:
        case 'num_epochs':
            x_label = 'Epochs'

        case 'batch_size':
            x_label = 'Batch Size'

        case 'lr':
            x_label = 'Learning Rate'

    match y:
        case 'testing_acc':
            y_label = 'Testing Accuracy'

        case 'training_loss':
            y_label = 'Training Loss'

    plt.xticks(ticks=x_data, labels=x_data, rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'{file_name}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='cnn_results.csv', help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='graphs', help='Directory to save the generated graphs')

    args = parser.parse_args()

    data = load_data(args.input)

    labels = data.columns.tolist()
    for label in labels:
        if label not in ['testing_acc', 'training_loss']:
            acc_df, loss_df = accuracy_select(data, label)  # Pass the loaded data to the function

            generate_graphs(acc_df, f'{label}', 'testing_acc', f'{label}_testing_acc', args.output)  # Pass the loaded data to the function
            generate_graphs(loss_df, f'{label}', 'training_loss', f'{label}_training_loss', args.output)  # Pass the loaded data to the function








