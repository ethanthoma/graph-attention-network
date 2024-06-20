<h3 align="center">
    Graph Attention Network
</h3>

This is an implementation of the [GAT paper](https://arxiv.org/abs/1710.10903v3) 
in [tinygrad](https://docs.tinygrad.org/).

The `./app/train.py` script uses the CORA dataset. 

The model gets 80.39% accuracy on the test set (2/3rds of the dataset) after 
1000 epochs.

Full training time was 51:11 minutes on a surface pro 9, CPU only.

Checkpoints for every 100 epochs are found in `./models/` along with the final 
model.
