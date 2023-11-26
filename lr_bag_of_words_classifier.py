import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample DataFrame (replace with your actual DataFrame)
data = {'text': ['This is a positive sentence.', 'This is a negative sentence.', 'Another positive example.'],
        'labels': [1, 0, 1]}
df = pd.DataFrame(data)

# Preprocess the text data (you can customize this preprocessing)
df['text'] = df['text'].str.lower()  # Convert to lowercase

# Create a bag of words representation using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['labels'], test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)


# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier (you can use different metrics as needed)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)