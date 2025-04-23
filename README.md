Spiking Neural Networks
Daniel Price & Brandon Nguyen
April 23, 2025

5610 Intoduction to Machine Learning Final Project

Model code can be found in models/
snn_conv was heavily directed by [BNTT](https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time)

All the figures in data/figs are updated to the paper version, and are not consistent with what was presented in class on April 21.
While they are technically different, the only difference is a slight compacting of hyperparameters and increase of batch size to generate figures faster. As noted in the powerpoint, it takes
nearly a day (24H) to run the SNN model for the entire case study, even on a 4090.
The current figures produce the same trends as the ones used in the powerpoint, and thus conclusions stay the same.

To rerun our code, follow the steps below.

All needed dependencies are listed in requirements.txt. to install, run

pip install -r requirements.txt

If you want to install pytorch with cuda (highly recommend to run the case study) please follow the instructions from [PyTorch](https://pytorch.org/get-started/locally/)

To run the performance (accuracy and loss) and efficiency (FLOPs) case studys, run
python -m training.performance_case_study
python -m training.efficiency_case_study

evalutaion is a single run for both models, and was used to generate the table in the powerpoint. To run it, run
python -m training.evaluation

Once all data has been generated, you can run
python -m data_processing.generate_figures

This will generate all the figures in the final paper.