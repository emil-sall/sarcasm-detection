import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
import optuna
import os

# Specify the folder path
folder_path = './sarcasm_detection/data/'

# Initialize an empty DataFrame
embedding_data = pd.DataFrame()

# Load the CSV files and concatenate them to the existing DataFrame
for i in range(0, 109):
    print(f"Loading file: {i+1}")
    file_path = os.path.join(folder_path, f'processed_data{i}.csv')
    new_data = pd.read_csv(file_path)
    new_data['text_embeddings'] = new_data['text_embeddings'].apply(lambda x: torch.from_numpy(np.array(eval(x), dtype=np.float32)))
    new_data['context_embeddings'] = new_data['context_embeddings'].apply(lambda x: torch.from_numpy(np.array(eval(x), dtype=np.float32)))
    embedding_data = pd.concat([embedding_data, new_data], ignore_index=True)



# Plot the distribution of labels, should be equal
plt.figure()
sns.countplot(x='label', data=embedding_data).set(title='Sarcastic labels')
plt.savefig('undersampled_data.png')

# Split the data into train and test
train_size = 0.85
test_size = 1 - train_size
x_train_context, x_test_context, x_train_text, x_test_text, y_train, y_test = train_test_split(embedding_data['context_embeddings'].tolist(), embedding_data['text_embeddings'].tolist(), torch.tensor(embedding_data['label'].values), test_size=test_size)

# create the data loaders
x_train_text = torch.stack(x_train_text)
x_train_context = torch.stack(x_train_context)
x_test_text = torch.stack(x_test_text)
x_test_context = torch.stack(x_test_context)
train_dataset = TensorDataset(x_train_context, x_train_text, y_train)
test_dataset = TensorDataset(x_test_context, x_test_text, y_test)

#_____________________________________________________________________
# New part where the model is defined

class SarcasmTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout):
        super(SarcasmTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # Implement the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(2 * embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(2 * embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    # Implement the feed forward network
    def forward(self, context, text):
        x = torch.cat((context, text), dim=1)
        x = x.to(torch.float32)
        x = x.view(-1, x.size(0), x.size(1))
        x = self.transformer_encoder(x)
        x = x[-1]
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.squeeze(1)
    

#_____________________________________________________________________
# New part where the best parameters are set


# Hyperparameters
embedding_dim = len(x_test_context[0])
batch_size = 64
device = torch.device('mps')
num_folds = 5
num_trials = 50
num_heads = 12

def train(
        num_layers,
        dropout,
        num_epochs,
        learning_rate,
        weight_decay,
        patience
    ):
    # Split the data into k folds
    kf = KFold(n_splits=num_folds, shuffle=True)

    # Lists to store scores
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # Train the model using k-fold cross validation with num_folds folds
    for fold, (train_index, val_index) in enumerate(kf.split(x_train_context)):
        print(f'Fold {fold + 1}/{num_folds}')

        # Split data into training and validation sets
        x_train_context_fold, x_val_context = x_train_context[train_index], x_train_context[val_index]
        x_train_text_fold, x_val_text = x_train_text[train_index], x_train_text[val_index]
        y_train_fold, y_val = y_train[train_index], y_train[val_index]

        train_dataset_fold = TensorDataset(x_train_context_fold, x_train_text_fold, y_train_fold)
        val_dataset = TensorDataset(x_val_context, x_val_text, y_val)

        # Create the data loaders
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model, optimizer, scheduler and loss function
        model = SarcasmTransformer(embedding_dim, num_heads, num_layers, dropout)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        criterion = nn.BCELoss()

        best_train_loss = float('inf')
        best_val_loss = float('inf')
        consecutive_no_improvement = 0
        train_patience = 2 * patience

        # Train the model using k-fold cross validation with num_folds folds
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (context, text, label) in enumerate(train_loader_fold):
                context, text, label = context.to(torch.float32).to(device), text.to(torch.float32).to(device), label.to(torch.float32).to(device)
                optimizer.zero_grad()
                output = model(context, text)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 200 == 0:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / 200:.4f}')
                    if running_loss < best_train_loss:
                        best_train_loss = running_loss
                        consecutive_no_improvement = 0
                    else:
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= train_patience:
                            print(f'Early stopping at epoch {epoch + 1} due to no improvement in training loss.')
                            break
                    running_loss = 0.0

            consecutive_no_improvement = 0
            # Evaluate the model on the validation set
            model.eval()
            val_loss = 0.0
            y_true_fold = []
            y_pred_fold = []
            with torch.no_grad():
                for context, text, label in val_loader:
                    context, text, label = context.to(torch.float32).to(device), text.to(torch.float32).to(device), label.to(torch.float32).to(device)
                    output = model(context, text)
                    output = torch.round(output)
                    y_true_fold.extend(label.cpu().numpy())
                    y_pred_fold.extend(output.cpu().numpy())
                    val_loss += criterion(output, label).item()

            val_loss /= len(val_loader.dataset)  # Calculate average validation loss

            # Print results
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= patience:
                    print(f'Early stopping at epoch {epoch + 1} due to no improvement in validation loss.')
                    break

            scheduler.step(val_loss)

        # Compute results
        accuracy_fold = accuracy_score(y_true_fold, y_pred_fold)
        precision_fold = precision_score(y_true_fold, y_pred_fold, average="binary", zero_division=1)
        recall_fold = recall_score(y_true_fold, y_pred_fold, average="binary")
        f1_fold = f1_score(y_true_fold, y_pred_fold, average="binary")

        # Print results
        print(f'Validation result for fold {fold + 1}:')
        print(f'Accuracy: {accuracy_fold:.4f}')
        print(f'Precision: {precision_fold:.4f}')
        print(f'Recall: {recall_fold:.4f}')
        print(f'F1-score: {f1_fold:.4f}')

        # Save results
        accuracy_list.append(accuracy_fold)
        precision_list.append(precision_fold)
        recall_list.append(recall_fold)
        f1_list.append(f1_fold)
    
    # Calculate and print average metrics for all folds
    print('\nAverage results:')
    print(f'Average Accuracy: {np.mean(accuracy_list):.4f}')
    print(f'Average Precision: {np.mean(precision_list):.4f}')
    print(f'Average Recall: {np.mean(recall_list):.4f}')
    print(f'Average F1-score: {np.mean(f1_list):.4f}')

    return np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def objective(trial):
    # Hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 1, 8)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2)
    num_epochs = 1
    patience = 5
    accuracy, precision, recall, f1 = train(
        num_layers,
        dropout,
        num_epochs,
        learning_rate,
        weight_decay,
        patience
    )

    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=num_trials)

best_params = study.best_params
print("Best model found.")
print("Best Hyperparameters:", best_params)

num_layers = best_params["num_layers"]
dropout = best_params["dropout"]
num_epochs = 10
learning_rate = best_params["learning_rate"]
weight_decay = best_params["weight_decay"]

# New part where the best model possible is trained
train_dataset = TensorDataset(x_train_context, x_train_text, y_train)

# Create the data loaders
train_loader_fold = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, scheduler and loss function
model = SarcasmTransformer(embedding_dim, num_heads, num_layers, dropout)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
criterion = nn.BCELoss()

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (context, text, label) in enumerate(train_loader_fold):
        context, text, label = context.to(torch.float32).to(device), text.to(torch.float32).to(device), label.to(torch.float32).to(device)
        optimizer.zero_grad()
        output = model(context, text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 200 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / 200:.4f}')
            running_loss = 0.0
    scheduler.step(loss)

# Evaluate the model on the testing set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
y_true_test = []
y_pred_test = []

with torch.no_grad():
    for context, text, label in test_loader:
        context, text, label = context.to(torch.float32).to(device), text.to(torch.float32).to(device), label.to(device)
        output = model(context, text)
        output = torch.round(output)
        y_true_test.extend(label.cpu().numpy())
        y_pred_test.extend(output.cpu().numpy())

# Compute results
accuracy_test = accuracy_score(y_true_test, y_pred_test)
precision_test = precision_score(y_true_test, y_pred_test, average="binary", zero_division=1)
recall_test = recall_score(y_true_test, y_pred_test, average="binary")
f1_test = f1_score(y_true_test, y_pred_test, average="binary")

# Save the results
results_dict = {
    'Accuracy': accuracy_test,
    'Precision': precision_test,
    'Recall': recall_test,
    'F1-score': f1_test
}

results_df = pd.DataFrame([results_dict])
results_df.to_csv('model_results.csv', index=False)

# Print results
print('\nTesting results:')
print(f'Accuracy: {accuracy_test:.4f}')
print(f'Precision: {precision_test:.4f}')
print(f'Recall: {recall_test:.4f}')
print(f'F1-score: {f1_test:.4f}')

# Save the model
torch.save(model.state_dict(), 'sarcasm_model.pth')

print('Model and results saved successfully.')