

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import nltk
from transformers import RobertaTokenizer, RobertaModel
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import re
import nltk
import string
from torch.optim import Adam
from tqdm import tqdm

# Data handling

data_path = "./sarcasm_detection/data/complete_data.csv"
data = pd.read_csv(data_path)

data = data.dropna(subset=['text', 'context'])

# Sample the rows with specific labels and undersample the majority class
label_0 = data[data['label'] == 0].sample(data['label'].value_counts()[1])
label_1 = data[data['label'] == 1]

# Merge the undersampled data and shuffle it
undersampled_data = pd.concat([label_0, label_1])
data = undersampled_data.sample(frac=1).reset_index(drop=True)

# Merge "context" and "text" columns into a new column named "merged_text"
data['merged_text'] = data['context'] + ' ' + data['text']

X = data['merged_text']
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)
X_train, X_test, X_val, y_train, y_test, y_val = list(X_train), list(X_test), list(X_val), list(y_train), list(y_test), list(y_val)

# From https://colab.research.google.com/github/zabir-nabil/sarcasm-detection-roberta/blob/main/SARC_RoBERTa_Sarcasm_Detection_%5BPyTorch%5D.ipynb#scrollTo=rK9AIeuvi11W

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_token = tokenizer.texts_to_sequences(X_train)

vocab_size = len(tokenizer.word_index) + 1

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_length = 50, padding_side = 'right')

input_ids = torch.tensor(tokenizer.encode(X_train[0], add_special_tokens=True, max_length = 50, pad_to_max_length = True, truncation=True)).unsqueeze(0)  

encoded = tokenizer.encode_plus(X_train[0], add_special_tokens=True, max_length = 50, pad_to_max_length = True,
                                return_token_type_ids = False,
                                return_attention_mask = True, truncation=True)  # Batch size 1

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

class SARCDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        texts = X

        texts = [self._preprocess(text) for text in texts]

        self._print_random_samples(texts)

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        self.labels = y

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        #text = self._lowercase(text)
        text = self._remove_punctuation(text)
        #text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        #text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()


    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label

nltk.download('punkt')
nltk.download('stopwords')

train_sarc = SARCDataset(X_train, y_train, tokenizer)
val_sarc = SARCDataset(X_val, y_val, tokenizer)
test_sarc = SARCDataset(X_test, y_test, tokenizer)

train_dataloader = DataLoader(train_sarc, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_sarc, batch_size=8, num_workers=0)
test_dataloader = DataLoader(test_sarc, batch_size=8, num_workers=0)

class SARCClassifier(nn.Module):
    def __init__(self, base_model):
        super(SARCClassifier, self).__init__()

        self.bert = base_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
    

def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0
    EARLY_STOPPING = 3
    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    model_metrics = {}
    model_metrics['train_accuracy'] = []
    model_metrics['val_accuracy'] = []
    model_metrics['train_loss'] = []
    model_metrics['val_loss'] = []
    model_metrics['f1'] = []
    model_metrics['val_f1'] = []
    model_metrics['auc'] = []
    model_metrics['val_auc'] = []


    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_f1_train = 0
        total_auc_train = 0
        
        model.train()
        
        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            out_preds = output.cpu().detach().numpy().flatten()
            targets = train_label.cpu().detach().numpy().flatten()
            try:
              auc_score = roc_auc_score(targets, out_preds)
            except:
              auc_score = 1
            total_auc_train += auc_score

            out_preds[out_preds < 0.5] = 0
            out_preds[out_preds >= 0.5] = 1
            f1_score_ = f1_score(targets, out_preds)
            total_f1_train += f1_score_

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0
            total_f1_val = 0
            total_auc_val = 0
            
            model.eval()
            
            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc

                out_preds = output.cpu().detach().numpy().flatten()
                targets = val_label.cpu().detach().numpy().flatten()
                try:
                  auc_score = roc_auc_score(targets, out_preds)
                except:
                  auc_score = 1.
                total_auc_val += auc_score

                out_preds[out_preds < 0.5] = 0
                out_preds[out_preds >= 0.5] = 1
                f1_score_ = f1_score(targets, out_preds)
                total_f1_val += f1_score_
                    
            
            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            model_metrics['train_accuracy'].append(total_acc_train / (len(train_dataloader.dataset)))
            model_metrics['val_accuracy'].append(total_acc_val / len(val_dataloader.dataset))
            model_metrics['train_loss'].append(total_loss_train / len(train_dataloader))
            model_metrics['val_loss'].append(total_loss_val / len(val_dataloader))
            model_metrics['f1'].append(total_f1_train / len(train_dataloader))
            model_metrics['val_f1'].append(total_f1_val / len(val_dataloader))
            model_metrics['auc'].append(total_auc_train / len(train_dataloader))
            model_metrics['val_auc'].append(total_auc_val / len(val_dataloader))

            print(model_metrics)
            
            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(model, f"best_model.pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1
                
            if early_stopping_threshold_count >= EARLY_STOPPING:
                print("Early stopping")
                break
    return model_metrics

torch.manual_seed(0)
np.random.seed(0)


BERT_MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)


model = SARCClassifier(base_model)


learning_rate = 1e-5
epochs = 10

metrics = train(model, train_dataloader, val_dataloader, learning_rate, epochs)

# Load the trained model
model = torch.load("best_model.pt")  

# Put the model in evaluation mode
model.eval()

# Define lists to store predictions and true labels
predictions = []
true_labels = []

# Iterate over the test dataloader
for test_input, test_label in tqdm(test_dataloader):
    # Move inputs to the appropriate device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    attention_mask = test_input['attention_mask'].to(device)
    input_ids = test_input['input_ids'].squeeze(1).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_ids, attention_mask)

    # Convert probabilities to binary predictions (0 or 1)
    preds = (output >= 0.5).int()

    # Store predictions and true labels
    predictions.extend(preds.cpu().numpy().flatten())
    true_labels.extend(test_label.numpy().flatten())

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
