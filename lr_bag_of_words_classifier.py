import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import clean_data

def train_classifier(input_df, text_column_name='text',labels_column_name='labels'):
    # Create a bag of words representation using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(input_df[text_column_name])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, input_df[labels_column_name], test_size=0.2, random_state=42)

    # Train a logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier (you can use different metrics as needed)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    gao = clean_data.clean_data(filename='gao')
    print("gao.columns =", gao.columns)
    print("gao.head(5) =", gao.head(5))
    train_classifier(gao)

    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
