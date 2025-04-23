import training.snn_train as snn_train
import training.cnn_train as cnn_train
import pandas as pd
import os

# initialize paths
base_path = 'data/csv/'
cnn_path = base_path + 'cnn/'
snn_path = base_path + 'snn/'
if not os.path.exists(cnn_path):
    os.makedirs(cnn_path)
if not os.path.exists(snn_path):
    os.makedirs(snn_path)

# set hyperparameters to match highest accuracy from the case study for baseline
batch_size = 32
lr = 0.1
num_epochs = 25
epoch_list = [25]
img_size = 28
num_classes = 10
case_study = 'all'

# Baseline evaluation metrics
cnn_df = pd.DataFrame(columns=['batch_size', 'lr', 'num_epochs', 'training_loss', 'testing_accuracy', 'flops'])
cnn_df = cnn_train.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, epoch_list = epoch_list, img_size=img_size, num_classes=num_classes, case_study=case_study, df=cnn_df)

# set hyperparameters to match highest accuracy from the case study for snn
batch_size = 32
lr = 0.2
num_epochs = 50
epoch_list = [50]
num_steps = 10
leak_mem = 0.99
img_size = 28
num_classes = 10
case_study = 'all'

# SNN evaluation metrics
snn_df = pd.DataFrame(columns=['batch_size', 'lr', 'num_epochs', 'leak_mem', 'num_steps', 'training_loss', 'testing_accuracy', 'flops'])
snn_df = snn_train.run(batch_size=batch_size, lr=lr, num_epochs=num_epochs, epoch_list = epoch_list, leak_mem=leak_mem, num_steps=num_steps, img_size=img_size, num_classes=num_classes, case_study=case_study, df=snn_df)

# Normalize FLOPS
snn_df['flops'] = snn_df['flops'].iloc[0] / cnn_df['flops'].iloc[0]
cnn_df['flops'] = 1

# Save dataframes
snn_df.to_csv(cnn_path + 'evalutation.csv', index=False)
cnn_df.to_csv(snn_path + 'evaluation.csv', index=False)