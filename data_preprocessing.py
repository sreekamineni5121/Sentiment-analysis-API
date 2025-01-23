
import pandas as pd
import spacy

# Load SpaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Sample data
data = {
    "comment": [
        "I love this product!",
        "Not what I expected, very disappointed.",
        "It's okay, but could be better.",
        "Absolutely fantastic, highly recommend!",
        "Terrible experience, will not buy again.",
    ],
    "platform": ["Twitter", "Facebook", "Instagram", "LinkedIn", "Twitter"],
}
df = pd.DataFrame(data)

# Preprocess comments using SpaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

df["cleaned_comment"] = df["comment"].apply(preprocess_text)

# Assign sentiment labels: Positive (2), Neutral (1), Negative (0)
df["label"] = df["comment"].apply(lambda x: 2 if "love" in x or "fantastic" in x else (1 if "okay" in x else 0))

# Display the processed data
print(df)
