# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1waGb1thTj705QT4QDv8CoVr6p8Iw6o9K
"""

# Install the required packages
pip install transformers
pip install torch

# Import the required modules
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup

# Load the dataset
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("validate_data.csv")
test_df = pd.read_csv("test_data.csv")
test_labels_df = pd.read_csv("test_labels.csv")

# Preprocess the dataset
# Remove leading and trailing spaces from labels
train_df['Label'] = train_df['Label'].str.strip()
val_df['Label'] = val_df['Label'].str.strip()
test_df['Label'] = test_labels_df['Label'].str.strip()

# Encode the labels
le = LabelEncoder()
le.fit(train_df['Label'])

train_df['Label'] = le.transform(train_df['Label'])
val_df['Label'] = le.transform(val_df['Label'])
test_df['Label'] = le.transform(test_df['Label'])

# Load the IndicBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")

# Create the dataset and data loader classes
class CodeMixedTweetsDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_length):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = CodeMixedTweetsDataset(
        tweets=df['Sentence'].to_numpy(),
        labels=df['Label'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )



# Create the DataLoaders for train, validate, and test datasets
train_data_loader = create_data_loader(train_df, tokenizer, 128, 16)
val_data_loader = create_data_loader(val_df, tokenizer, 128, 16)
test_data_loader = create_data_loader(test_df, tokenizer, 128, 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the IndicBERT model
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=len(le.classes_)).to(device)

# Create the training and evaluation functions
from sklearn.metrics import f1_score

def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()

    correct_predictions = 0
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_examples, total_loss / len(data_loader)

def eval_model(model, data_loader, device, n_examples):
    model.eval()

    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    return correct_predictions.double() / n_examples, np.mean(losses), f1


# Set up the optimizer and learning rate scheduler
epochs=5
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_data_loader) * epochs)
# Early stopping parameters
n_epochs_no_improvement = 0
best_val_loss = float('inf')
early_stopping_patience = 2

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, len(train_df))
    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    val_acc, val_loss = eval_model(model, val_data_loader, device, len(val_df))
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        n_epochs_no_improvement = 0
    else:
        n_epochs_no_improvement += 1
        if n_epochs_no_improvement >= early_stopping_patience:
            print("Early stopping due to no improvement in validation loss")
            break


# Train and validate the model using a for loop
epochs = 10
patience = 2
best_val_f1 = 0
no_improvement_epochs = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, len(train_df))
    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    val_acc, val_loss, val_f1 = eval_model(model, val_data_loader, device, len(val_df))
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1 score: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= patience:
        print("Early stopping triggered.")
        break

# Evaluate the model on the test dataset
test_acc, _, test_f1 = eval_model(model, test_data_loader, device, len(test_df))
print(f"Test accuracy: {test_acc:.4f}, Test F1-score: {test_f1:.4f}")