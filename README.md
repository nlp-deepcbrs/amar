# AMAR: Ask Me Any Rating

Code for the paper "Ask Me Any Rating: A Content-based Recommender System based on Recurrent Neural Networks".

## Description
In this work we propose *Ask Me Any Rating (AMAR)*, a novel content-based recommender system based on deep neural networks which is able to produce top-N recommendations leveraging user and item embeddings which are learnt from textual information describing the items. 

A comprehensive experimental evaluation conducted on state of-the-art datasets such as *MovieLens 1M* and *DBbook* showed a significant improvement over all the baselines taken into account.

## Requirements

- Lua
- Torch
  - nn
  - rnn
  - cudnn
  - cunn
  - cutorch
  - optim
  - pl

## Usage

1. Retrieve item descriptions and save them in ".txt" files. Each file should be named as the item identifier in the used dataset (e.g., item 1 has a description file named 1.txt)
2. Create JSON configuration file for the training file that you want to use and start the training using *Torch* specifying the configuration filename with the `-config` parameter
3. Evaluate the trained model using the `run_amar_experiments.lua` program specifying the configuration filename with the `-config` parameter

## Configuration files
Training and evaluation configuration files are in JSON format and are composed by specific fields. They are used in order to modify model parameters and to specify the supplementary files used to train the models or to evaluate the models.

For instance, the training configuration file for the `train_amar_rnn.lua` file is composed by the following fields:
    - items: path of item descriptions
    - genres: filename of item genres (optional)
    - models_mapping: dictionary which associates training sets to models
    - rnn_unit: RNN unit identifier used in rnn package
    - optim_method: optimization method identifier used in optim package
    - training_params: parameters of the optimization method
    - batch_size: number of training examples in a batch
    - num_epochs: number of training epochs
    - save_after: save model each save_after epochs
See the specific training file to understand which are the required parameters.

In addition, the evaluation configuration file for the `run_amar_experiments.lua` file is in JSON format and is composed by the following fields:
    - items: path of items descriptions
    - genres: filename of items genres
    - models_mapping: dictionary which associates test files to models
    - predictions: generated predictions filename
    - batch_size: number of examples in a batch
    - topn: list of cutoff values

## Authors

All the following authors have equally contributed to this project (listed in alphabetical order by surname):

- Claudio Greco ([github](https://github.com/claudiogreco))
- Alessandro Suglia ([github](https://github.com/aleSuglia))
