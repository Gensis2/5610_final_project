import pandas as pd

def accuracy_select(df, label):
    acc_row = df
    highest_accuracy_row = acc_row.loc[acc_row['testing_acc'].idxmax()]
    lowest_loss_row = acc_row.loc[acc_row['training_loss'].idxmin()]
    columns_to_match = [col for col in acc_row.columns if col not in [label, 'testing_acc', 'training_loss']]
    acc_df = acc_row[
        acc_row[columns_to_match].eq(highest_accuracy_row[columns_to_match]).all(axis=1)
    ]
    loss_df = acc_row[
        acc_row[columns_to_match].eq(lowest_loss_row[columns_to_match]).all(axis=1)
    ]

    return acc_df, loss_df
df = pd.read_csv('data_processing/cnn_results.csv')
acc_df, loss_df = accuracy_select(df, 'batch_size')
print(acc_df)
print(loss_df)