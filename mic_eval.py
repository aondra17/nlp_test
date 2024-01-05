from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import torch
import csv

# Load data
train_df = pd.read_csv("train.csv", quoting=csv.QUOTE_ALL)
# only 1 missing drop it
train_df = train_df.dropna(subset=["Text"])
# drop duplicates
train_df = train_df.drop_duplicates(subset=["Text"])

valid_df = pd.read_csv("valid.csv", quoting=csv.QUOTE_ALL)
valid_df = valid_df.drop_duplicates(subset=["Text"])

test_df = pd.read_csv("test.csv", quoting=csv.QUOTE_ALL)

# Load pre-trained BERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(train_df['ParagraphType'].unique()))

# Tokenize text data


def tokenize_data(df):
    tokenized = tokenizer(df['Text'].tolist(), padding=True,
                          truncation=True, return_tensors='pt')
    return {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}


train_data = tokenize_data(train_df)
valid_data = tokenize_data(valid_df)
test_data = tokenize_data(test_df)

# Prepare PyTorch datasets


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CustomDataset(
    train_data, train_df['ParagraphType'].astype('category').cat.codes.tolist())
valid_dataset = CustomDataset(
    valid_data, valid_df['ParagraphType'].astype('category').cat.codes.tolist())
# Dummy labels for test set
test_dataset = CustomDataset(test_data, [0] * len(test_df))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=20,  # Save checkpoint every 20 steps
    save_total_limit=3  # Limit the number of checkpoints to save
)

# Load the trained model from the checkpoint
model = BertForSequenceClassification.from_pretrained(
    "./results_distilberts/checkpoint-4440")

# Trainer for evaluation
eval_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use train_dataset for evaluation if needed
    eval_dataset=valid_dataset
)

# Evaluate on validation dataset
eval_results = eval_trainer.evaluate()

# Get predicted labels on validation set
val_predictions = eval_trainer.predict(valid_dataset)
val_predicted_labels = val_predictions.predictions.argmax(-1)

# Map predicted labels back to actual classes
predicted_paragraph_types_val = [
    train_df['ParagraphType'].unique()[label] for label in val_predicted_labels]

# Calculate accuracy
actual_labels_val = valid_df['ParagraphType'].astype(
    'category').cat.codes.tolist()
accuracy = accuracy_score(actual_labels_val, val_predicted_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(actual_labels_val, val_predicted_labels)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
