import numpy as np
import pandas as pd
from transformers import AutoModel, BertTokenizerFast
from sklearn import preprocessing
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import clean_text

class BERT_Fake(nn.Module):
    def __init__(self, BERT):
      super(BERT_Fake, self).__init__()
      self.BERT = BERT
      self.fc1 = nn.Linear(768, 512)
      self.fc2 = nn.Linear(512, 1)
      self.dropout = nn.Dropout(0.1)

    def forward(self, sent_id, mask):
      cls_hs = self.BERT(sent_id, attention_mask=mask)['pooler_output']

      x = F.relu(self.fc1(cls_hs))
      x = self.dropout(x)
      x = F.sigmoid((self.fc2(x)))
      return x

# Load BERT model and tokenizer via HuggingFace Transformers
BERT = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

######################### Data

PATH = "data/original-data/"

train = pd.read_csv(PATH + 'Constraint_Train.csv', header=0)
# drop posts that are too long and the duplicates
train = train[train["tweet"].map(len) <= 280].drop_duplicates()
X_train, y_train = train["tweet"], train["label"]

val = pd.read_csv(PATH + 'Constraint_Val.csv', header=0)
# drop posts that are too long and the duplicates
val = val[val["tweet"].map(len) <= 280].drop_duplicates()
X_val, y_val = val["tweet"], val["label"]


test = pd.read_csv(PATH + 'Constraint_Test.csv', header=0)
# drop posts that are too long and the duplicates
test = test[test["tweet"].map(len) <= 280].drop_duplicates()
X_test, y_test = test["tweet"], test["label"]

X_train = X_train.map(clean_text)
X_val = X_val.map(clean_text)
X_test = X_test.map(clean_text)

label_encoder = preprocessing.LabelEncoder()

#  'fake' -> 0, 'real' -> 1
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)
y_test = label_encoder.fit_transform(y_test)

# Preprocessing

# we clean all the tweets
X_train = X_train.map(clean_text)
X_val = X_val.map(clean_text)
X_test = X_test.map(clean_text)

label_encoder = preprocessing.LabelEncoder()

#  'fake' -> 0, 'real' -> 1
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.fit_transform(y_val)
y_test = label_encoder.fit_transform(y_test)

# Tokenize and encode sequences
MAX_LENGHT = 50 # We restricted ourselves to the 50 first words

tokens_train = tokenizer.batch_encode_plus(
    X_train.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)
tokens_val = tokenizer.batch_encode_plus(
    X_val.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)
tokens_test = tokenizer.batch_encode_plus(
    X_test.tolist(),
    max_length = MAX_LENGHT,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(y_train.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(y_val.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(y_test.tolist())

# Data Loader structure definition
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) # dataLoader for train set

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size) # dataLoader for validation set

# Defining training and evaluation functions
def train():
    model.train()
    total_loss, total_accuracy = 0, 0

    for step, batch in enumerate(train_dataloader):
        if step % 20 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r for r in batch]  # push the batch to gpu
        sent_id, mask, labels = batch
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = criterion(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients to 1.0. It helps in preventing exploding gradient problem
        optimizer.step()
        preds = preds.detach().cpu().numpy()  # model predictions are stored on GPU. So, push it to CPU

    avg_loss = total_loss / len(train_dataloader)  # compute training loss of the epoch
    return avg_loss  # returns the loss and predictions


def evaluate():
    print("\nEvaluating...")
    model.eval()  # Deactivate dropout layers
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(val_dataloader):  #
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        batch = [t for t in batch]  # Push the batch to GPU
        sent_id, mask, labels = batch
        with torch.no_grad():  # Deactivate autograd
            preds = model(sent_id, mask)
            loss = criterion(preds, labels)  # Compute the validation loss between actual and predicted values
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
    avg_loss = total_loss / len(val_dataloader)  # compute the validation loss of the epoch
    return avg_loss


for param in BERT.parameters():
   param.requires_grad = False # Fine_tuning phase

model = BERT_Fake(BERT)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()
epochs = 4

# Train and predict
train_losses = []
valid_losses = []

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()  # train model
    valid_loss = evaluate()  # evaluate model
    train_losses.append(train_loss)  # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

torch.save(model.state_dict(), 'BERT_FAKE_weights.pt')

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

print(classification_report(test_y, preds))