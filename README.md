# Sentiment Analysis Using RNN and CNN

Models from this repository with modifications: https://github.com/bentrevett/pytorch-sentiment-analysis
Dataset from: https://alt.qcri.org/semeval2017/task4/

## Requirements
- Python 3.7
- PyTorch 1.8.1
- torchtext 0.9.0

## Usage

### Command line
Run `python main.py train rnn` to train the rnn model.
Run `python main.py train lstm` to train the lstm model.
Run `python main.py train cnn` to train the cnn model.

*After* the model has been trained,

Run `python main.py predict <model_name>` to predict a sentence to see if it is of negative emotion.
Replace `<model_name>` with `rnn`, `lstm`, or `cnn`.

The returned value represents the probability of the sentence being negative.

### Train
Run `train_cnn` in `main.py` to train a CNN model. Parameters are saved in `cnn-model.pt`.
Run `train_rnn` in `main.py` to train a CNN model. Parameters are saved in `rnn-model.pt`.
Run `train_lstm` in `main.py` to train a CNN model. Parameters are saved in `lstm-model.pt`.


### Predict

Run `predict_<model_name>` in `main.py`. This requires the `<model+name>-model.pt` file to be existed.

## Implementation
The model structure can be found under
```
class CNN:
  ...
 
class RNN:
  ...
  
 class LSTM:
  ...
```

The word embedding method used is [GloVe](https://nlp.stanford.edu/projects/glove/).
