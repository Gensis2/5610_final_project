import training.snn_inference as snn_inference
import training.cnn_inference as cnn_inference
import pandas as pd
import os

# Set hyperparameters for the case study
batch_size = 32
lr = 0.1
num_epochs = 25

img_size = 28
num_classes = 10
case_study = 'all'

# Initialize paths
df_path = 'data/csv/'
snn_path = df_path + 'snn/'
cnn_path = df_path + 'cnn/'

# initialize dataframes
# single run of baseline model for given batch size, lr, and epochs with best accuracy

cnn_df = pd.DataFrame(columns=['batch_size', 'lr', 'num_epochs', 'training_loss', 'testing_accuracy', 'flops'])
cnn_df = cnn_inference.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, img_size=img_size, num_classes=num_classes, case_study=case_study, df=cnn_df)

batch_size = 32
lr = 0.2
num_epochs = 50
num_steps_list = [5, 10, 20, 40]
leak_mem_list = [0.99, 0.985, 0.98, 0.975]

# loop through snn model for num_steps and leak_mem with static batch size, lr, and epochs of best accuracy
snn_df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy', 'flops'])
label_df = pd.DataFrame(columns=['batch_size', 'lr', 'leak_mem', 'num_steps', 'num_epochs', 'training_loss', 'testing_accuracy', 'flops'])
for num_steps in num_steps_list:
    for leak_mem in leak_mem_list:
        print(f"Running with num_steps={num_steps}, leak_mem={leak_mem}")
        new_snn_df = snn_inference.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, img_size=img_size, num_classes=num_classes, num_steps=num_steps, leak_mem=leak_mem, case_study=case_study, df=label_df)

        snn_df = pd.concat([snn_df, new_snn_df], ignore_index=True)

# normalize flops to baseline model
snn_df['flops'] = snn_df['flops'] / cnn_df['flops'].iloc[0]
cnn_df['flops'] = 1
snn_df.to_csv(f'{snn_path}efficiency_case_study.csv', index=False)
cnn_df.to_csv(f'{cnn_path}efficiency_case_study.csv', index=False)