# QUIC Network Traffic Classification

This project focuses on QUIC network traffic classification using deep learning.

## Dataset

The main dataset used for training is the UCDavis QUIC dataset:

https://www.kaggle.com/datasets/guillaumefraysse/ucdavisquic

## Files

`AICS 2026.py`  
Training script using the UCDavis QUIC dataset.

`ucdavis_model.pt`  
Trained model obtained from the UCDavis QUIC dataset.

`CESNET_TL`  
Transfer learning script that uses the UCDavis-trained model and applies transfer learning on the CESNET dataset.

## Features

The model uses three types of traffic features:

- Statistical features
- Sequential features
- Burst-level features

The experiments evaluate individual feature types as well as their combinations.

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Inference latency

## Dataset Source

UCDavis QUIC Dataset  
Guillaume Fraysse  
https://www.kaggle.com/datasets/guillaumefraysse/ucdavisquic
