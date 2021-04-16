import random
import sys

import torch
import torchtext
from torch.utils.data import random_split, DataLoader
from torch import nn
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import dropout, relu, max_pool1d

vector = torchtext.vocab.GloVe()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1
torch.manual_seed(SEED)

TRAIN_SAMPLE_SIZE = None

class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.4)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return self.fc(output[:, -1, :])

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        output, hidden = self.rnn(x)
        return self.fc(output[:, -1, :])

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
                        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels=1, 
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embedded):
                
        #embedded.shpae = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


class TextDataset(Dataset):
    def __init__(self, X, labels, transform=None, target_transform=None):
        super().__init__()
        self.X = X
        self.labels = labels
        self.indices_ = list(range(len(self.X)))
        assert len(X) == len(labels)
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        index = self.indices_[index]
        x = self.X[index]
        label = self.labels[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        sample = x, label
        return sample

    def __len__(self):
        return len(self.X)
    
    def sample(self, size):
        if size is not None:
            indices = list(range(len(self.X)))
            sampled = random.sample(indices, size)
            return TextDataset([self.X[i] for i in sampled], [self.labels[i] for i in sampled])
        else:
            return self

    def sort(self):
        self.indices_.sort(key=lambda i: len(self.X[i]))
    


def save_data_imdb():
    train_val_data, test_data = torchtext.datasets.IMDB()


    tokenize = get_tokenizer("basic_english")

    tokens = []
    labels = []
    for label, line in train_val_data:
        tokens.append(tokenize(line))
        labels.append(label)

    train_val_text_dataset = TextDataset(tokens, labels)
    with open('train_val_data_imdb.pt', 'wb') as f:
        torch.save(train_val_text_dataset, f)

    tokens = []
    labels = []
    for label, line in test_data:
        tokens.append(tokenize(line))
        labels.append(label)

    test_text_dataset = TextDataset(tokens, labels)

    with open('test_data_imdb.pt', 'wb') as f:
        torch.save(test_text_dataset, f)

def save_data_yahoo_answers():
    train_val_data, test_data = torchtext.datasets.YahooAnswers()
    TRAIN_LEN = 25000
    TEST_LEN = 25000
    tokenize = get_tokenizer("basic_english")

    tokens = []
    labels = []
    for label, line in random.sample(list(train_val_data), TRAIN_LEN):
        tokens.append(tokenize(line))
        labels.append(label)

    train_val_text_dataset = TextDataset(tokens, labels)
    with open('train_val_data_yahoo.pt', 'wb') as f:
        torch.save(train_val_text_dataset, f)

    tokens = []
    labels = []
    for label, line in random.sample(list(test_data), TEST_LEN):
        tokens.append(tokenize(line))
        labels.append(label)

    test_text_dataset = TextDataset(tokens, labels)

    with open('test_data_yahoo.pt', 'wb') as f:
        torch.save(test_text_dataset, f)



def load_data_imdb():
    with open('train_val_data.pt', 'rb') as f:
        train_val_text_dataset: TextDataset = torch.load(f)


    train_val_text_dataset.X = train_val_text_dataset.X
    train_val_text_dataset.labels = train_val_text_dataset.labels
    val_ratio = 0.3

    val_length = int(len(train_val_text_dataset) * val_ratio)
    train_length = len(train_val_text_dataset) - val_length
    val_subset, train_subset = random_split(train_val_text_dataset, (val_length, train_length))

    sort = True 
    if sort:
        val_subset.indices = sorted(val_subset.indices, key=lambda i: len(val_subset.dataset[i][0]))
        train_subset.indices = sorted(train_subset.indices, key=lambda i: len(train_subset.dataset[i][0]))

    return val_subset, train_subset, None

def load_data_semeval():

    tokenize = get_tokenizer('basic_english')
    data = []
    with open('semeval/gold/twitter-2016dev-A.txt') as f:
        for line in f:
            id_, label, text = line.split('\t')
            if label == 'negative':
                y = 1
            else:
                y = 0
            token = tokenize(text)
            data.append((token, y))
    negatives = [d for d in data if d[1] == 1]
    non_negatives = [d for d in data if d[1] != 1]
    non_negatives = random.sample(non_negatives, len(negatives))
    data = non_negatives + negatives
    random.shuffle(data)
    val_data = TextDataset([d[0] for d in data], [d[1] for d in data])

    data = []
    with open('semeval/gold/twitter-2016train-A.txt') as f:
        for line in f:
            id_, label, text = line.split('\t')
            if label == 'negative':
                y = 1
            else:
                y = 0
            token = tokenize(text)
            data.append((token, y))
    negatives = [d for d in data if d[1] == 1]
    non_negatives = [d for d in data if d[1] != 1]
    non_negatives = random.sample(non_negatives, len(negatives))
    data = non_negatives + negatives
    random.shuffle(data)
    train_data = TextDataset([d[0] for d in data], [d[1] for d in data])

    data = []
    with open('semeval/gold/twitter-2016test-A.txt') as f:
        for line in f:
            id_, label, text, *rest = line.split('\t')
            if label == 'negative':
                y = 1
            else:
                y = 0
            token = tokenize(text)
            data.append((token, y))
    negatives = [d for d in data if d[1] == 1]
    non_negatives = [d for d in data if d[1] != 1]
    non_negatives = random.sample(non_negatives, len(negatives))
    data = non_negatives + negatives
    random.shuffle(data)
    test_data = TextDataset([d[0] for d in data], [d[1] for d in data])

    sort = True 
    if sort:
        train_data.sort()
        test_data.sort()
        val_data.sort()

    return val_data, train_data, test_data


def load_data_semeval_join():

    tokenize = get_tokenizer('basic_english')
    data = []
    lines = []
    with open('semeval/gold/twitter-2016dev-A.txt') as f:
        lines += list(f)
    with open('semeval/gold/twitter-2016test-A.txt') as f:
        lines += list(f)
    with open('semeval/gold/twitter-2016dev-A.txt') as f:
        lines += list(f)
    
    for line in lines:
        id_, label, text, *rest = line.split('\t')
        if label == 'negative':
            y = 1
        elif label == 'neutral':
            continue
        else:
            y = 0
        token = tokenize(text)
        data.append((token, y))
    negatives = [d for d in data if d[1] == 1]
    non_negatives = [d for d in data if d[1] != 1]
    non_negatives = random.sample(non_negatives, len(negatives))
    data = non_negatives + negatives
    random.shuffle(data)

    test_ratio = 0.2
    test_length = round(len(data) * test_ratio)
    train_length = len(data) - test_length

    test_data = data[:test_length]
    test_dataset = TextDataset([d[0] for d in test_data], [d[1] for d in test_data])

    train_data = data[test_length:test_length+train_length]
    train_dataset = TextDataset([d[0] for d in train_data], [d[1] for d in train_data])


    train_dataset = train_dataset.sample(TRAIN_SAMPLE_SIZE)

    return test_dataset, train_dataset, test_dataset


def collate_fn(batch):
    X = []
    labels = []
    for x, label in batch:
        X.append(vector.get_vecs_by_tokens(x).squeeze(1))
        labels.append(label)
    X = pad_sequence(X, batch_first=True)
    X = X.to(device)
    labels = torch.FloatTensor(labels).to(device)
    return {'X': X, 'labels': labels}





def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def triple_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    output = torch.sigmoid(preds)
    correct = (torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

    

def train(model, iterator, optimizer, criterion, train_length):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    total_trained = 0
    rounds = 0
    for batch in iterator:
        total_trained += len(batch['labels'])
        rounds += 1
        if rounds % 20 == 0:
            # print(f'{100 * total_trained/train_length:.2f}')
            pass
        optimizer.zero_grad()
        predictions = model(batch['X']).squeeze(1)
        
        loss = criterion(predictions, batch['labels'])
        
        acc = binary_accuracy(predictions, batch['labels'])
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch['X']).squeeze(1)
            
            loss = criterion(predictions, batch['labels'])
            
            acc = binary_accuracy(predictions, batch['labels'])

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def create_model_lstm():
    EMBEDDING_DIM = vector.dim
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    return model

def create_model_cnn():
    EMBEDDING_DIM = vector.dim
    OUTPUT_DIM = 1
    
    model = CNN(EMBEDDING_DIM, 200, (3, 4, 5, 5), OUTPUT_DIM, 0.5)
    return model

def create_model_rnn():

    EMBEDDING_DIM = vector.dim
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1



    model = RNN(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    return model

def train_lstm():
    BATCH_SIZE = 64


    val_set, train_set, test_set = load_data_semeval_join()
    print('loaded')
    val_iterator = DataLoader(val_set, BATCH_SIZE, collate_fn=collate_fn)
    train_iterator = DataLoader(train_set, BATCH_SIZE, collate_fn=collate_fn)
    test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)

    train_length = len(train_set)

    model = create_model_lstm()
    optimizer = torch.optim.Adam(model.parameters())

    loss = nn.BCEWithLogitsLoss()

    model = model.to(device)
    loss = loss.to(device)
    N_EPOCHS = 50

    best_valid_loss = float('inf')

    loss_history = [] # Seuqnece[(train, val, test)]
    acc_history = []

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, loss, train_length)
        valid_loss, valid_acc = evaluate(model, val_iterator, loss)
        test_loss, test_acc = evaluate(model, test_iterator, loss)
        
        loss_history.append((train_loss, valid_loss, test_loss))
        acc_history.append((train_acc, valid_acc, test_acc))
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'lstm-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
    print('loss history')
    print(loss_history)
    print('acc history')
    print(acc_history)


def train_rnn():
    BATCH_SIZE = 64


    val_set, train_set, test_set = load_data_semeval_join()
    print('loaded')
    val_iterator = DataLoader(val_set, BATCH_SIZE, collate_fn=collate_fn)
    train_iterator = DataLoader(train_set, BATCH_SIZE, collate_fn=collate_fn)
    test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)

    train_length = len(train_set)
    model = create_model_rnn()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = nn.BCEWithLogitsLoss()

    model = model.to(device)
    loss = loss.to(device)
    N_EPOCHS = 100

    best_valid_loss = float('inf')

    loss_history = [] # Seuqnece[(train, val, test)]
    acc_history = []

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, loss, train_length)
        valid_loss, valid_acc = evaluate(model, val_iterator, loss)
        test_loss, test_acc = evaluate(model, test_iterator, loss)
        
        loss_history.append((train_loss, valid_loss, test_loss))
        acc_history.append((train_acc, valid_acc, test_acc))
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'rnn-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
    print('loss history')
    print(loss_history)
    print('acc history')
    print(acc_history)

def train_cnn():
    BATCH_SIZE = 64

    val_set, train_set, test_set = load_data_semeval_join()

    print('loaded')
    train_length = len(train_set)

    val_iterator = DataLoader(val_set, BATCH_SIZE, collate_fn=collate_fn)
    train_iterator = DataLoader(train_set, BATCH_SIZE, collate_fn=collate_fn)
    test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)

    model = create_model_cnn()
    optimizer =  torch.optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 30

    best_valid_loss = float('inf')
    loss_history = [] # Seuqnece[(train, val, test)]
    acc_history = []

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, train_length)
        valid_loss, valid_acc = evaluate(model, val_iterator, criterion)
        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        
        loss_history.append((train_loss, valid_loss, test_loss))
        acc_history.append((train_acc, valid_acc, test_acc))
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'cnn-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

    print('loss history')
    print(loss_history)
    print('acc history')
    print(acc_history)

def predict(X, model):
    with torch.no_grad():
        predictions = torch.sigmoid(model(X).squeeze(1))
        return predictions


def load_model_cnn(path):
    model = create_model_cnn()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_model_lstm(path):
    model = create_model_lstm()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_model_rnn(path):
    model = create_model_rnn()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict_cnn():
    model = load_model_cnn('cnn-model.pt')
    tokenize = get_tokenizer("basic_english")
    min_length = 10
    while True:
        text = input('Please input: ')
        if text:
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            if x.size(0) < min_length:
                padded = torch.zeros((min_length, x.size(1)))
                padded[:x.size(0), :] = x
                x = padded
            X = x.unsqueeze(0)
            print(predict(X, model))
    # val_set, train_set, test_set = load_data_semeval()
    # BATCH_SIZE = 64
    # test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)
    # print(evaluate(model, test_iterator, nn.BCEWithLogitsLoss()))


def predict_lstm():
    model = load_model_lstm('lstm-model.pt')
    tokenize = get_tokenizer("basic_english")
    min_length = 10
    while True:
        text = input('Please input: ')
        if text:
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            if x.size(0) < min_length:
                padded = torch.zeros((min_length, x.size(1)))
                padded[:x.size(0), :] = x
                x = padded
            X = x.unsqueeze(0)
            print(predict(X, model))
    # val_set, train_set, test_set = load_data_semeval()
    # BATCH_SIZE = 64
    # test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)
    # print(evaluate(model, test_iterator, nn.BCEWithLogitsLoss()))

def predict_rnn():
    model = load_model_rnn('rnn-model.pt')
    tokenize = get_tokenizer("basic_english")
    min_length = 10
    while True:
        text = input('Please input: ')
        if text:
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            x = vector.get_vecs_by_tokens(tokenize(text)).to(device).squeeze(1)
            if x.size(0) < min_length:
                padded = torch.zeros((min_length, x.size(1)))
                padded[:x.size(0), :] = x
                x = padded
            X = x.unsqueeze(0)
            print(predict(X, model))
    # val_set, train_set, test_set = load_data_semeval()
    # BATCH_SIZE = 64
    # test_iterator = DataLoader(test_set, BATCH_SIZE, collate_fn=collate_fn)
    # print(evaluate(model, test_iterator, nn.BCEWithLogitsLoss()))

def main():
    assert len(sys.argv) == 3
    cmd = sys.argv[1]
    model = sys.argv[2]
    cmd_dict = {
        'train': {
            'cnn': train_cnn,
            'lstm': train_lstm,
            'rnn': train_rnn,
        },
        'predict': {
            'cnn': predict_cnn,
            'lstm': predict_lstm,
            'rnn': predict_rnn,
        }
    }
    run = cmd_dict.get(cmd, {}).get(model)
    if run:
        run()

main()
