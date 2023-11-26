import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import clean_data

def train_classifier(input_df, text_column_name='text',labels_column_name='labels', test_size=0.05, random_seed=8):
    # Create a bag of words representation using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(input_df[text_column_name])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, input_df[labels_column_name], test_size=test_size, random_state=random_seed)

    # Train a logistic regression classifier
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier (you can use different metrics as needed)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return classifier, vectorizer


def pretty_print_preds(input_list, predictions):
    # Header row
    print("=" * 58)  # Separator line
    print("||\tInput\t\t|\t\tPrediction\t||")
    print("=" * 58)  # Separator line
    # print("||\tword\t|\tprob of toxicity\t||")
    for i in range(len(input_list)):
        # Convert the prediction value to a regular Python float
        prediction_value = float(predictions[i][1])

        # print("||\t", input_list[i],"\t\t|\t\t",predictions[i][1],"\t||")
        # Format the output with fixed column widths and right-aligned content
        formatted_output = f"||\t{input_list[i]:<10}\t|\t\t{prediction_value:.6f}\t||"
        print(formatted_output)
    print("=" * 58)  # Separator line
    print("")

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    gao_data = clean_data.clean_data(filename='gao')
    print("gao_data.columns =", gao_data.columns)
    print("gao_data.head(5) =", gao_data.head(5))
    gao_classifier, vectorizer_gao = train_classifier(gao_data)

    zampieri_data = clean_data.clean_data(filename='zampieri')
    print("zampieri_data.columns =", zampieri_data.columns)
    print("zampieri_data.head(5) =", zampieri_data.head(5))
    zampieri_classifier, vectorizer_zampieri = train_classifier(zampieri_data)

    # Make predictions
    test_array = ["female", "muslim", "black", "gay", "hispanic", "african", "hi"]
    # Convert the list of words into a bag of words representation
    # using the same vectorizer that you used for training
    X_test = vectorizer_gao.transform(test_array)
    gao_predictions = gao_classifier.predict_proba(X_test)
    # print("gao_classifier predictions=", gao_predictions)
    pretty_print_preds(test_array, gao_predictions)

    X_test = vectorizer_zampieri.transform(test_array)
    zampieri_predictions = zampieri_classifier.predict_proba(X_test)
    # print("zampieri_classifier predictions=",zampieri_predictions)
    pretty_print_preds(test_array, zampieri_predictions)

    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
