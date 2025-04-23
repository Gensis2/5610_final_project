import training.snn_train as snn_train
import training.cnn_train as cnn_train
import pandas as pd
import os

# Set hyperparameters for the case study
batch_size_list = [32, 64, 128, 256]
lr_list = [0.2, 0.1, 0.05, 0.01]
num_epochs_list = [5, 10, 25, 50]
num_steps_list = [5, 10, 20, 40]
leak_mem_list = [0.99, 0.985, 0.98, 0.975]

img_size = 28
num_classes = 10
case_study = 'accuracy'

# Initialize paths
df_path = 'data/csv/'
snn_path = df_path + 'snn/'
cnn_path = df_path + 'cnn/'

# initialize dataframes
# loop through hyperparameters in case study in baseline and SNN

df = pd.DataFrame(columns=['batch_size', 'lr', 'num_epochs', 'training_loss', 'testing_accuracy'])
label_df = pd.DataFrame(columns=['batch_size', 'lr', 'num_epochs', 'training_loss', 'testing_accuracy'])
for batch_size in batch_size_list:
    for lr in lr_list:
        print(f"Running with batch_size={batch_size}, lr={lr}")
        new_df = cnn_train.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs_list[-1], epoch_list=num_epochs_list, img_size=img_size, num_classes=num_classes, case_study=case_study, df=label_df)
        df = pd.concat([df, new_df], ignore_index=True)

df.to_csv(f'{cnn_path}performance_case_study.csv', index=False)

df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy'])
label_df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy'])
for batch_size in batch_size_list:
    for lr in lr_list:
        for num_steps in num_steps_list:
            for leak_mem in leak_mem_list:
                print(f"Running with batch_size={batch_size}, lr={lr}, num_steps={num_steps}, leak_mem={leak_mem}")
                new_df = snn_train.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs_list[-1], epoch_list=num_epochs_list, img_size=img_size, num_classes=num_classes, num_steps=num_steps, leak_mem=leak_mem, case_study=case_study, df=label_df)

                df = pd.concat([df, new_df], ignore_index=True)
                # Takes a long time to run, checkpoint every run in case of inerruption.
            df.to_csv(f'{snn_path}performance_case_study_checkpoint.csv', index=False)

os.rmdir(f'{snn_path}performance_case_study_checkpoint.csv')
df.to_csv(f'{snn_path}performance_case_study.csv', index=False)