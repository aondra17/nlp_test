import csv
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
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

# Define Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    save_total_limit=3,  # Limit the number of checkpoints to save
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Fine-tune the model
trainer.train()

# Predict labels for test data
test_predictions = trainer.predict(test_dataset)
predicted_labels = label_encoder.inverse_transform(
    test_predictions.predictions.argmax(axis=1))
test_df['Predicted_ParagraphType'] = predicted_labels

# Save or use the predictions
test_df.to_csv('predicted_test.csv', index=False)
