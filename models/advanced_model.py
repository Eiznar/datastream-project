import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda")

# ! pip install transformers

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

for param in BERT.parameters():
    param.requires_grad = False

model = BERT_Fake(BERT)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCELoss()
epochs = 2