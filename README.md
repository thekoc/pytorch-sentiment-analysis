# Sentiment Analysis Using RNN and CNN

Models from this repository with modifications: https://github.com/bentrevett/pytorch-sentiment-analysis
Dataset from: https://alt.qcri.org/semeval2017/task4/

## Requirements
- Python 3.7
- PyTorch 1.8.1
- torchtext 0.9.0

## Usage

### Train
Run `train_cnn` in `main.py` to train a CNN model. Parameters are saved in `cnn-model.pt`.
Run `train_rnn` in `main.py` to train a CNN model. Parameters are saved in `rnn-model.pt`.


### Predict

Run `predict_cnn` in `main.py`. This requires the `cnn-model.pt` file to exist.

