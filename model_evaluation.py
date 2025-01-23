
import matplotlib.pyplot as plt
import numpy as np

# Evaluate the model
predictions = trainer.predict(val_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Add predictions to the dataframe for visualization
df["predicted_label"] = ["Positive" if l == 2 else "Neutral" if l == 1 else "Negative" for l in predicted_labels]

# Visualize sentiment trends
sentiment_counts = df["predicted_label"].value_counts()
sentiment_counts.plot(kind="bar", color=["green", "orange", "red"], title="Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
