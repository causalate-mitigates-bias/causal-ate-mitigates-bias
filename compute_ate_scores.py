import numpy as np
import pandas as pd
import time
import classifier, clean_data, mask_and_replace
from utilities import pretty_print_three_preds
from pathlib import Path

# Disable Setting with copy warning. This is false positive error. See following discussion.
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'


def getATEScores(input_unmasked_df, input_classifier, input_vectorizer,
                 output_unmasked_file_with_scores="outputs/gao_unmasked_with_scores.csv",
                 output_ate_csv_file="outputs/gao_scores.csv"):
    input_unmasked_df = input_unmasked_df[input_unmasked_df['unmasked_sentences'].notna()]
    gao_unmasked_sentences = input_unmasked_df["unmasked_sentences"]

    unmasked_sentences_transformed = input_vectorizer.transform(input_unmasked_df["unmasked_sentences"])
    inputs_sentences_transformed = input_vectorizer.transform(input_unmasked_df["text"])

    # Predict using the trained classifier
    new_predictions = input_classifier.predict_proba(unmasked_sentences_transformed)
    new_predictions = [x[1] for x in new_predictions]

    input_predictions = input_classifier.predict_proba(inputs_sentences_transformed)
    input_predictions = [x[1] for x in input_predictions]

    # If you want to add these predictions back into the DataFrame
    input_unmasked_df.loc[:,'unmasked_predictions'] = new_predictions
    input_unmasked_df.loc[:,'input_predictions'] = input_predictions
    input_unmasked_df.loc[:,'decrease_on_replacement'] = input_unmasked_df['input_predictions'] - input_unmasked_df[
        'unmasked_predictions']
    input_unmasked_df.loc[:,'abs_decrease_on_replacement'] = np.maximum(0, input_unmasked_df['decrease_on_replacement'])

    input_unmasked_df.to_csv(output_unmasked_file_with_scores, sep="|")

    # Group by 'masked_sentence_id' and calculate the mean of 'unmasked_predictions'
    ate_scores = input_unmasked_df.groupby('masked_words')['abs_decrease_on_replacement'].mean()
    sorted_ate_scores = ate_scores.sort_values(ascending=False)
    # Save Series to csv
    sorted_ate_scores.to_csv(output_ate_csv_file, sep="|")

    # Read DataFrame from csv
    atedf = pd.read_csv(output_ate_csv_file, sep="|")
    ate_dict = atedf.set_index('masked_words')['abs_decrease_on_replacement'].to_dict()
    return ate_dict


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # GAO
    data = clean_data.clean_data(filename='gao')
    classifier, vectorizer = classifier.train_classifier(data)

    unmasked_path = "outputs/gao_unmasked.csv"
    my_file = Path(unmasked_path)
    if my_file.is_file():
        # file exists
        unmasked_df = pd.read_csv(unmasked_path, sep="|")
    else:
        masked_df, unmasked_df = mask_and_replace.create_masked_replacements(input_df=data,
                                                            outfile_masked="outputs/gao_masked.csv",
                                                            outfile_unmasked=unmasked_path,
                                                            normalizer_sequence=None,
                                                            replacement_model='roberta-base')

    ate_dict = getATEScores(input_unmasked_df=unmasked_df, input_classifier=classifier,
                            input_vectorizer=vectorizer,
                            output_unmasked_file_with_scores="outputs/gao_unmasked_with_scores.csv",
                            output_ate_csv_file="outputs/gao_scores.csv")

    test_array = ["female", "black", "gay", "hispanic", "african"]
    X_test = vectorizer.transform(test_array)
    predictions = classifier.predict_proba(X_test)
    ate_predictions = [ate_dict[x] for x in test_array]

    # print("gao_classifier predictions=", gao_predictions)
    pretty_print_three_preds(test_array, predictions, ate_predictions)

    # ZAMPIERI
    data = clean_data.clean_data(filename='zampieri')
    classifier, vectorizer = classifier.train_classifier(data)

    unmasked_path = "outputs/zampieri_unmasked.csv"
    my_file = Path(unmasked_path)
    if my_file.is_file():
        # file exists
        unmasked_df = pd.read_csv(unmasked_path, sep="|")
    else:
        masked_df, unmasked_df = mask_and_replace.create_masked_replacements(input_df=data,
                                                            outfile_masked="outputs/gao_masked.csv",
                                                            outfile_unmasked=unmasked_path,
                                                            normalizer_sequence=None,
                                                            replacement_model='roberta-base')


    ate_dict = getATEScores(input_unmasked_df=unmasked_df, input_classifier=classifier,
                            input_vectorizer=vectorizer,
                            output_unmasked_file_with_scores="outputs/zampieri_unmasked_with_scores.csv",
                            output_ate_csv_file="outputs/zampieri_scores.csv")

    test_array = ["female", "black", "gay", "hispanic", "african"]
    X_test = vectorizer.transform(test_array)
    predictions = classifier.predict_proba(X_test)
    ate_predictions = [ate_dict[x] for x in test_array]

    pretty_print_three_preds(test_array, predictions, ate_predictions)

    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
