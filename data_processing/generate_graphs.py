import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def load_data(file_path):
    return pd.read_csv(file_path)

# Find best lr and epoch while varying batch size
def find_best_lr_and_epoch(data):
    # Implement logic to find the best learning rate and epoch based on the data

    # Obtain a list of all batch sizes
    batch_sizes = data['batch_size'].unique()
    print("Batch Sizes: ", batch_sizes)

    # Obtain a list of all learning rates
    learning_rates = data['lr'].unique()

    # Obtain a list of all epochs
    epochs = data['num_epochs'].unique()

    # Initialize variables to store the best learning rate and epoch
    best_lr = None
    best_epoch = None
        
    # Organize data by same lr and epoch, varying batch size
    for lr in learning_rates:
        for epoch in epochs:
            subset = data[(data['lr'] == lr) & (data['num_epochs'] == epoch)]
            if not subset.empty:
                # Calculate the average accuracy for the current learning rate and epoch
                print(subset)
                avg_accuracy = subset['testing_acc'].mean()
                print(f"Learning Rate: {lr}, Epoch: {epoch}, Average Accuracy: {avg_accuracy}")
                # Update the best learning rate and epoch if necessary
                if best_lr is None or avg_accuracy > best_accuracy:
                    best_lr = lr
                    best_epoch = epoch
                    best_accuracy = avg_accuracy
                    print(f"New Best Learning Rate: {best_lr}, Epoch: {best_epoch}, Accuracy: {best_accuracy}")
    
    return best_lr, best_epoch

def generate_graphs(x, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Example: Generate a simple line graph
    plt.figure(figsize=(10, 5))
    plt.plot(data['x'], data['y'])
    plt.title('Sample Graph')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(os.path.join(output_dir, 'sample_graph.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='cnn_results.csv',required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='graphs', help='Directory to save the generated graphs')

    args = parser.parse_args()

    data = load_data(args.input)

    best_lr, best_epoch = find_best_lr_and_epoch(data)

    
    

