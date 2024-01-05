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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=25,  # Save checkpoint every 20 steps
    save_total_limit=3  # Limit the number of checkpoints to save
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

# Fine-tuning BERT
trainer.train()

# Save the trained model
model.save_pretrained('./fine_tuned_bert_model')

# Load the fine-tuned model for prediction
model = DistilBertTokenizer.from_pretrained(
    './fine_tuned_bert_model')

# Prepare test dataset for prediction
test_predictions = []
for i in range(0, len(test_data['input_ids']), 8):
    with torch.no_grad():
        outputs = model(**{k: v[i:i+8] for k, v in test_data.items()})
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        test_predictions.extend(predictions)

# Map predicted labels back to actual classes
predicted_paragraph_types = [
    train_df['ParagraphType'].unique()[label] for label in test_predictions]

# Assign predicted paragraph types to the test dataframe
test_df['ParagraphType'] = predicted_paragraph_types
