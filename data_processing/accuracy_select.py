import pandas as pd

def accuracy_select(df, label):
    acc_row = df
    highest_accuracy_row = acc_row.loc[acc_row['testing_acc'].idxmax()]
    lowest_loss_row = acc_row.loc[acc_row['training_loss'].idxmin()]
    acc_df = acc_row[
        (acc_row['lr'] == highest_accuracy_row['lr']) & 
        (acc_row['num_epochs'] == highest_accuracy_row['num_epochs'])
    ]
    loss_df = acc_row[
        (acc_row['lr'] == lowest_loss_row['lr']) & 
        (acc_row['num_epochs'] == lowest_loss_row['num_epochs'])
    ]

    return acc_df, loss_df