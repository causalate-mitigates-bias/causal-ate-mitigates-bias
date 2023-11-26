import numpy as np
import pandas as pd
import lr_bag_of_words_classifier, clean_data
from lr_bag_of_words_classifier import pretty_print_preds

gao_data = clean_data.clean_data(filename='gao')
gao_classifier, vectorizer_gao = lr_bag_of_words_classifier.train_classifier(gao_data)
gao_unmasked_df = pd.read_csv("outputs/gao_unmasked.csv", sep="|")
gao_unmasked_df = gao_unmasked_df[gao_unmasked_df['unmasked_sentences'].notna()]
gao_unmasked_sentences = gao_unmasked_df["unmasked_sentences"]

unmasked_sentences_transformed = vectorizer_gao.transform(gao_unmasked_df["unmasked_sentences"])
inputs_sentences_transformed = vectorizer_gao.transform(gao_unmasked_df["text"])

# Predict using the trained classifier
new_predictions = gao_classifier.predict_proba(unmasked_sentences_transformed)
new_predictions = [x[1] for x in new_predictions]

input_predictions = gao_classifier.predict_proba(inputs_sentences_transformed)
input_predictions = [x[1] for x in input_predictions]

# If you want to add these predictions back into the DataFrame
gao_unmasked_df['unmasked_predictions'] = new_predictions
gao_unmasked_df['input_predictions'] = input_predictions
gao_unmasked_df['decrease_on_replacement'] = gao_unmasked_df['input_predictions'] - gao_unmasked_df[
    'unmasked_predictions']
gao_unmasked_df['abs_decrease_on_replacement'] = np.maximum(0, gao_unmasked_df['decrease_on_replacement'])

gao_unmasked_df.to_csv("outputs/gao_unmasked_with_scores.csv", sep="|")

# Group by 'masked_sentence_id' and calculate the mean of 'unmasked_predictions'
ate_scores = gao_unmasked_df.groupby('masked_words')['abs_decrease_on_replacement'].mean()
sorted_ate_scores = ate_scores.sort_values(ascending=False)
sorted_ate_scores.to_csv("outputs/gao_scores.csv", sep="|")

# Convert DataFrame to dictionary
atedf = pd.read_csv("outputs/gao_scores.csv", sep="|")
ate_dict = atedf.set_index('masked_words')['abs_decrease_on_replacement'].to_dict()

test_array = ["female", "black", "gay", "hispanic", "african"]
X_test = vectorizer_gao.transform(test_array)
gao_predictions = gao_classifier.predict_proba(X_test)
gao_ate_predictions = [ate_dict[x] for x in test_array]

def pretty_print_preds(input_list, predictions, ate_scores):
    # Header row
    print("=" * 75)  # Separator line
    print("||\tInput\t\t|\t\tClassifier Pred\t\t|\t\tATE Scores\t\t||")
    print("=" * 75)  # Separator line
    # print("||\tword\t|\tprob of toxicity\t||")
    for i in range(len(input_list)):
        # Convert the prediction value to a regular Python float
        prediction_value = float(predictions[i][1])
        ate_score = float(ate_scores[i])

        # print("||\t", input_list[i],"\t\t|\t\t",predictions[i][1],"\t||")
        # Format the output with fixed column widths and right-aligned content
        formatted_output = f"||\t{input_list[i]:<10}\t\t|\t\t{prediction_value:.6f}\t\t|\t\t{ate_score:.6f}\t\t||"
        print(formatted_output)
    print("=" * 75)  # Separator line
    print("")


# print("gao_classifier predictions=", gao_predictions)
pretty_print_preds(test_array, gao_predictions,gao_ate_predictions)
