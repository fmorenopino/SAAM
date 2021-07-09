# Spectral Attention Autoregressive Model (SAAM)

[![DOI](https://zenodo.org/badge/377461790.svg)](https://zenodo.org/badge/latestdoi/377461790)

This repository contains the Pytorch implementation of the Spectral Attention Autoregressive Model (SAAM) proposed in the paper 'Deep Autoregressive Models with Spectral Attention'.

## Structure.

The repository is divided into three parts:

- <code>SAAM_LSTM_Embedding</code> contains an implementation of the SAAM model based in [DeepAR model](https://www.sciencedirect.com/science/article/pii/S0169207019301888). In this implementation, a LSTM is used to perform the embedding described in the paper.
- <code>SAAM_Transformer_Embedding</code> contains an implementation of the SAAM model based in [ConvTrans model](https://papers.nips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html). In this implementation, a decoder-only mode Transformer is used to perform the embedding described in the paper.
- <code>Synthetic_Datasets_Exps</code> contains the code to train and evaluate SAAM (using a LSTM for the embedding) in the synthetic data.

## Usage

Notice that for using the LSTM or Transformer based model you will have to go to the corresponding folder. The arguments are slightly different for both models, you can check them in detail in <code>SAAM_LSTM_Embedding/train_SAAM_LSTM_Emb.py</code> and <code>SAAM_Transformer_Embedding/train_SAAM_Transformer_Emb.py</code>. Some examples of how to run both models on the solar dataset are:

```
# Example for training SAAM performing the embedding through a LSTM on the solar dataset:
python train_FAAM_ct.py --dataset solar --model-name solar --cuda-device 0 --sampling

# Example for training SAAM performing the embedding through a ConvTrans on the solar dataset:
CUDA_VISIBLE_DEVICES=0 python main.py --path data/solar.csv --outdir solar --dataset solar --enc_len 168 --dec_len 24 --batch-size 128
```

Some important parameters are: 

- <code>--dataset</code>: the available options are <code>ele</code>, <code>traffic</code>, <code>m4</code>, <code>wind</code> and <code>solar</code>.
- <code>--sampling</code>: whetever to compute p-risk metrics.
- <code>--path</code>: path to the dataset location.

## Synthetic dataset example notebook.

A example notebook is contained in <code>Synthetic_Datasets_Exps/Synthetic Dataset Experiment.ipynb</code>. Running this notebook, predictions on the synthetic dataset with a pre-trained model will be visible.

## Datasets.

Datasets used in the paper are publicly available. They can be download from the following sources:

- Electricity dataset: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#.
- Traffic dataset: https://archive.ics.uci.edu/ml/datasets/PEMS-SF.
- Solar dataset: https://www.nrel.gov/grid/solar-power-data.html.
- Wind dataset: https://www.kaggle.com/sohier/30-years-of-european-wind-generation.
- M4 Hourly dataset: https://www.kaggle.com/yogesh94/m4-forecasting-competition-dataset.
- For the synthetic dataset, 'Synthetic_Datasets_Exps/dataloaders/dataloader_sin_cos.py' can be checked. 

## Dependencies

The code has been tested with the following dependencies.

```
python 3.8.5
torch 1.6.0
matplotlib 3.3.2
numpy 1.19.2
pandas 1.1.5
scikit-learn 0.23.2
```

Notice that Pytorch introduced several modifications on the FFT functions on the version 1.7.0 (https://pytorch.org/blog/the-torch.fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pyTorch/). The model was developed with Pytorch 1.6.0, hence running it in a newer version will cause errors. Code will be updated asap to the new Pytorch specifications.

## Contributors
[Fernando Moreno-Pino](http://www.tsc.uc3m.es/~fmoreno/), [Pablo M. Olmos](http://www.tsc.uc3m.es/~olmos/), and [Antonio Artés-Rodríguez](http://www.tsc.uc3m.es/~antonio/antonio_artes/Home.html).

For further information: <a href="mailto:fmoreno@tsc.uc3m.es">fmoreno@tsc.uc3m.es</a>
