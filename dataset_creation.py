
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

# Custom dataset class for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Tokenizer initialization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare the texts and labels
texts = df["cleaned_comment"].tolist()
labels = df["label"].tolist()

# Print tokenizer example
print(tokenizer(texts[0], max_length=128, padding="max_length", truncation=True))
