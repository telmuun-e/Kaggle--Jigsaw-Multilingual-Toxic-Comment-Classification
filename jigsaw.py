import numpy as np 
import pandas as pd 
import transformers
import os
import seaborn as sns
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

train1 = pd.read_csv(r"/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv(r"/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
valid = pd.read_csv(r"/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv")
test = pd.read_csv(r"/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")

print(train1.head())
print(train2.head())
print(valid.head())
print(test.head())

train1["toxic"] = train1["toxic"].round().astype(int)
train2["toxic"] = train2["toxic"].round().astype(int)
train = pd.concat([
    train1.loc[:,["id","comment_text","toxic"]],
    train2.loc[:,["id","comment_text","toxic"]]])
train.reset_index(drop=True,inplace=True)
print(train.info())

print(test.lang.value_counts())

word_len = [len(i.split(" ")) for i in train.comment_text]
sns.distplot(word_len)

def text_cleaning(text):
    text=text.map(lambda x: re.sub(r'\\n',' ',str(x)))
    text=text.map(lambda x: re.sub(r'[0-9"]', '', str(x)))
    text=text.map(lambda x: re.sub(r'#[\S]+\b', '', str(x)))
    text=text.map(lambda x: re.sub(r'@[\S]+\b', '', str(x)))
    text=text.map(lambda x: re.sub(r'https?\S+', '', str(x)))
    text=text.map(lambda x: re.sub(r'\s+', ' ', str(x)))
    text=text.map(lambda x: re.sub(r'\[\[User.*','',str(x)))
    return text


train["comment_text"]=text_cleaning(train["comment_text"])
valid["comment_text"]=text_cleaning(valid["comment_text"])
test["comment_text"]=text_cleaning(test["content"])

train[train["toxic"]==1].head()

def encoding(data,tokenizer,maxlen=512):
    encoded_data=tokenizer.batch_encode_plus(text,add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            pad_to_max_length=True,
                                            max_length=maxlen,
                                            return_tensors='pt')
    return encoded_data


class Data_processing:
    def __init__(self, data, tokenizer):
        self.text = data.comment_text.values
        self.target = data.toxic.values
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.ids)
    def __getitem__(self,index):
        encoded = encoding(self.text, self.tokenizer)
        ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]
        tokens = encoded["token_type_ids"]
        ids, tokens, attention_masks = torch.Tensor(ids), torch.Tensor(tokens), torch.Tensor(attention_mask)
        target = torch.Tensor(self.target)
        dataset = torch.utils.data.TensorDataset(ids, attention_masks, tokens, target)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return data_loader


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.roberta = transformers.AutoModel.from_pretrained(r"/kaggle/input/jplu-tf-xlm-roberta-large")
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(self.roberta.pooler.dense.out_features*2,1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, out2 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        dropped = self.drop(out2)
        output = self.out(dropped)
        return output


tokenizer = transformers.AutoTokenizer.from_pretrained(r"/kaggle/input/jplu-tf-xlm-roberta-large")

EPOCHS = 5
device = torch.device("cuda")
optimizer = transformers.AdamW(model.parameters(), lr=0.001, eps=1e-08)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainset)*EPOCHS)

def acc_fn(outputs,targets):
    output = np.argmax(outputs,axis=1).flatten()
    target = targets.flatten()
    return roc_auc_score(output, target, average="weighted")

def loss_fn(outputs, targets):
    return nn.BCEWithLogitLoss()(outputs, target)


def train_fn(data_loader, model, optimizer, device):
    model.train()
    for batch in tqdm(data_loader, total=len(data_loader)):
        ids = batch['ids']
        attention_mask = batch['attention_mask']
        token = batch['token_type_ids']
        target = batch['target']
        ids = ids.to(device)
        attention_mask = attention_mask.to(device)
        token = token.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input_ids=ids, 
                       attention_mask=attention_mask,
                       token_type_ids = token)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()
    predictions, true_vals = [], []
    with torch.no_grad():
        for batch in data_loader:
            ids = batch['ids']
            attention_mask = batch['attention_mask']
            token = batch['token_type_ids']
            target = batch['target']
            ids = ids.to(device)
            attention_mask = attention_mask.to(device)
            token = token.to(device)
            target = target.to(device)
            output = model(input_ids=ids, 
                         attention_mask=attention_mask,
                         token_type_ids = token)
            
            true_vals.append(target.detach().cpu().numpy())
            predictions.append(output.detach().cpu().numpy())
    return predictions, true_vals


model = Model()
model.to(device)
model = nn.DataParallel(model)

best = 0
for epoch in range(EPOCHS):
    train_fn(Data_processing(train, tokenizer), model, optimizer, device)
    output, target = eval_fn(Data_processing(valid, tokenizer), model, device)
    accuracy = acc_fn(output, target)
    print (f"Epoch: {epoch}, Accuracy: {accuracy}")
    if accuracy > best:
        best = accuracy
        torch.save(model.state_dict(), PATH)
