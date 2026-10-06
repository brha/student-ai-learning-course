# Week 2 Starter Code

This folder contains a simple machine learning example for Week 2.

## Simple Spam Detector

```python
# week2/spam_detector_starter.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Tiny dataset of messages
messages = [
    "Free prize winner! Click now",
    "Hi team, meeting at 3pm",
    "You have won a cash reward",
    "Can we talk later?",
    "Claim your free gift today",
    "Let's review the project",
    "Urgent: click this link",
    "Please send the notes",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Turn text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train a simple model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Try new samples
new_samples = [
    "Free prize click here",
    "Can we meet tomorrow?",
    "Urgent reward",
    "Project is ready",
]

for sample in new_samples:
    sample_vector = vectorizer.transform([sample])
    prediction = model.predict(sample_vector)[0]
    label = "spam" if prediction == 1 else "not spam"
    print(f"Sample: {sample} -> {label}")
```

## Challenge

- Add more messages to the training data
- Try a different model from scikit-learn
- Create a sentiment classifier for movie reviews
