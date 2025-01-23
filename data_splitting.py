
from sklearn.model_selection import train_test_split

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create dataset objects
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_len=128)

# Check dataset sample
print(train_dataset[0])
