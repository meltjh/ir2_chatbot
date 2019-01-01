# ir2_chatbot
Information Retrieval 2 Project 10-1: Joint Knowledge Selection and Dialogue Response Generation.

## Code usage
This repository contains the code which is used to create the results as mentioned in the paper. (1) train.py trains a model given the arguments which are shown when using --help. Checkpoints are stored in the checkpoints directory; each checkpoint could be several hundreds of megabytes large. (2) checkpoints/plot_losses.py plots all checkpoints that are saved in that directory. (3) checkpoint_to_eval.py with the similar arguments as train.py, will generate predictions of that specific experiment based on the data/tokenized_input.txt sentences. These predictions are saved in the evaluation/result_data directory. This directory contains all generated response sentences and 
(4) evaluation/evaluate.py uses those predictions to compare with data/ tokenized_target.txt and calculate the BLEU and ROUGE scores and plot them.

## Data used in the paper
The checkpoints are not in this repository due to the large size of the files. The responses generated that are used in the paper are stored in this repository. The data for this project can be obtained from https://github.com/nikitacs16/Holl-E by clicking on Download. The data that is used is from experiment_data/bidaf. Copy the folders so that the structure is as following: ir2_chatbot/data/experiment_data/bidaf/mixed_short and ir2_chatbot/data/experiment_data/bidaf/oracle_short.
