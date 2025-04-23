import pandas as pd

df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_acc'])
for batch_size in ['32', '64', '128', '256']:
    for lr in ['0.2', '0.1', '0.05', '0.01']:
        for num_epochs in ['5', '10', '25', '50']:
            path = f'batch_size/{batch_size}/lr/{lr}/num_epochs/{num_epochs}/snn_results.csv'
            epoch_df = pd.read_csv(path)
            df = pd.concat([df, epoch_df], ignore_index=True)
            df = df.drop_duplicates()  # Remove duplicate rows

df.to_csv('data_processing/snn_results.csv', index=False)