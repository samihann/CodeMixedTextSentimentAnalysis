# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n3WuZHO57z0v8_L5teNrRkeqJ6XVipg3
"""

# Install the required libraries
# Import the necessary packages
pip install transformers torch pandas scikit-learn
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Read the CSV files
# Load the preprocessed data
train_df = pd.read_csv("train_data.csv")
val_df = pd.read_csv("validate_data.csv")
test_df = pd.read_csv("test_data.csv")
test_labels = pd.read_csv("test_labels.csv")
test_df["Label"] = test_labels["Label"]

# Remove leading and trailing spaces from labels
train_df['Label'] = train_df['Label'].str.strip()
val_df['Label'] = val_df['Label'].str.strip()
test_df['Label'] = test_df['Label'].str.strip()

# Encode the labels using LabelEncoder
encoder = LabelEncoder()
train_df["Label"] = encoder.fit_transform(train_df["Label"])
val_df["Label"] = encoder.transform(val_df["Label"])
test_df["Label"] = encoder.transform(test_df["Label"])
num_classes = len(encoder.classes_)

# Define the CodeMixedTweetsDataset class and the create_data_loader function
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

# Initialize the BERT tokenizer and create DataLoaders
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

# Create the data loaders
train_data_loader = create_data_loader(train_df, tokenizer, 128, 16)
val_data_loader = create_data_loader(val_df, tokenizer, 128, 16)
test_data_loader = create_data_loader(test_df, tokenizer, 128, 16)

# Initialize the BERT model and set up the device
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased",
    num_labels=num_classes,
    output_attentions=False,
    output_hidden_states=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the training and evaluation functions
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

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
            labels = batch["labels"].to(device)

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

# Set up the optimizer, scheduler, and training parameters
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * 3

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Train and validate the model
epochs = 20
patience = 3
best_val_loss = float('inf')
no_improvement_epochs = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, len(train_df))
    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

    val_acc, val_loss, val_f1 = eval_model(model, val_data_loader, device, len(val_df))
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

    # Save the model with the best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1

    # Early stopping
    if no_improvement_epochs >= patience:
        print("Early stopping triggered.")
        break

# Evaluate the model on the test set, test the model and calculate evaluation metrics
test_acc, test_loss, test_f1 = eval_model(model, test_data_loader, device, len(test_df))
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}, F1-score: {test_f1:.4f}")

# Save the model
torch.save(model.state_dict(), "sentiment_model.pt")