import pandas as pd
from data_processing.graphs import performance_select, efficiency_select, plot_graphs

cnn_path = 'data/figs/cnn/'
snn_path = 'data/figs/snn/'
normalized_path = 'data/figs/normalized_flops/'

cnn_performance_data = pd.read_csv('data/csv/cnn/performance_case_study.csv')
snn_performance_data = pd.read_csv('data/csv/snn/performance_case_study.csv')
cnn_efficiency_data = pd.read_csv('data/csv/cnn/efficiency_case_study.csv')
snn_efficiency_data = pd.read_csv('data/csv/snn/efficiency_case_study.csv')

# Create CNN case study figures
labels = cnn_performance_data.columns.tolist()
for label in labels:
    if label not in ['testing_accuracy', 'training_loss']:
        acc_df, loss_df = performance_select(cnn_performance_data, label) 

        plot_graphs(acc_df, f'{label}', 'testing_accuracy', f'{label}_testing_acc_cnn', cnn_path + 'testing/')  
        plot_graphs(loss_df, f'{label}', 'training_loss', f'{label}_training_loss_cnn', cnn_path + 'training/') 

# Create SNN case study figures
labels = snn_performance_data.columns.tolist()
for label in labels:
    if label not in ['testing_accuracy', 'training_loss']:
        acc_df, loss_df = performance_select(snn_performance_data, label)

        plot_graphs(acc_df, f'{label}', 'testing_accuracy', f'{label}_testing_acc_snn', snn_path + 'testing/') 
        plot_graphs(loss_df, f'{label}', 'training_loss', f'{label}_training_loss_snn', snn_path + 'training/')

# Create normalized FLOPs figures
labels = snn_efficiency_data.columns.tolist()
for label in labels:
    if label not in ['flops', 'batch_size', 'lr', 'num_epochs']:
        snn_total_flops_df = efficiency_select(snn_efficiency_data, label) 
        cnn_total_flops_df = efficiency_select(cnn_efficiency_data, label)

        if label in ['training_loss', 'testing_accuracy']:
            plot_graphs(snn_total_flops_df, label, 'flops', f'{label}_normalized_FLOPs', normalized_path, mark_xlabels=False, point=cnn_total_flops_df)
        else:
            plot_graphs(snn_total_flops_df, f'{label}', 'flops', f'{label}_normalized_FLOPs', normalized_path)
