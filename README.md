# The codes and datasets of 'Adversarial Signed Graph Learning with Differential Privacy'
### Manuscript are available in https://arxiv.org/abs/2512.00307


### Key Document Description
1. dataset_configs.json:  The hyperparameters of each dataset
2. main_edge_sign_prediction.py: Main code for edge sign prediction tasks
3. main_node_clustering.py: Main code for node clustering tasks
4. discriminator.py: The discriminator module of ASGL
5. generator.py: The generator module of ASGL
6. rdp_accountant.py: Privacy budget calculation
7. data file: Storing each processed datasets



### Arguments
```
Model hyperparameters:
--dataset                 Dataset name
--n_emb                   Embedding dimension size
--lr                      Learning rate
--window_size             Context window size for pair generation
--lambda_gen          	  L2 regularization weight for generator
--lambda_dis          	  L2 regularization weight for discriminator
--n_epochs                Number of training epochs
--n_epochs_gen            Generator training iterations per epoch
--n_epochs_dis            Discriminator training iterations per epoch
--n_sample_gen            Number of generating edges per node
--batch_size_gen          Generator batch size
--batch_size_dis          Discriminator batch size
--n_node_subsets          Number of node subsets for large graphs

Privacy parameters:
--noise_stddev            Noise standard deviation for RDP
--clip_value              Gradient clipping value for RDP
--epsilon                 Privacy budget epsilon
--delta                   Probability of privacy protection failure
--RDP                     Enable Renyi Differential Privacy

Experimental flags:
--partial_node_flag       Use subset of nodes for large graphs (such as Epinions dataset)

```

### Basic Usage
```
1. Edge sign prediction task:
python src/main_edge_sign_prediction.py --dataset Bitcoin_Alpha --epsilon 3

2. Node clustering task:
python src/main_node_clustering.py --dataset Bitcoin_Alpha --epsilon 3
```

### Requirements
The code has been tested running under Python 3.6.13. The required packages are as follows:

- ```tensorflow == 1.8.0```
- ```numpy == 1.19.5```
- ```pandas == 0.24.2```
- ```scikit-learn == 0.19.1```
- ```tqdm == 4.23.4```

