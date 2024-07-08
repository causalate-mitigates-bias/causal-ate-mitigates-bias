import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import time
from utilities.clean_data import clean_data
from utilities.mask_and_replace import create_masked_replacements
from utilities.printing_utilities import pretty_print_three_preds
from utilities.classifier_models import train_classifier
from pathlib import Path
from compute_ate_scores import getATEScores

# Disable Setting with copy warning. This is false positive error. See following discussion.
# https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None  # default='warn'




if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducability
    np.random.seed(8)
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    # Expect 40 minutes of runtime on a single cpu thread (if the unmasked df's are available).

    # ZAMPIERI
    data = clean_data(filename='zampieri')
    # Gao
    # data = clean_data(filename='gao')

    # classifier_names = ["LR", "SVM", "GradientBoost", "NaiveBayes"]
    # classifier_names = ["LR", "GradientBoost", "NaiveBayes"] #SVM is super slow
    # classifier_names = ["NN1Layer5", "NN2Layer105", "NN3Layer20105"]  # SVM is super slow
    # classifier_names = ["NN2Layer105", "NN3Layer20105"]  # SVM is super slow
    classifier_names = ["LR", "SVM", "GradientBoost", "NaiveBayes", "NN1Layer5", "NN2Layer105", "NN3Layer20105"]  # SVM is super slow
    for classifier_name in classifier_names:
        input_classifier, vectorizer = train_classifier(data, classifier=classifier_name)

        # unmasked_path = "outputs/zampieri_unmasked.csv"
        unmasked_path = "../outputs/gao_unmasked.csv"
        my_file = Path(unmasked_path)
        if my_file.is_file():
            # file exists
            unmasked_df = pd.read_csv(unmasked_path, sep="|")
        else:
            masked_df, unmasked_df = create_masked_replacements(input_df=data,
                                                                                 outfile_masked="outputs/zampieri_masked.csv",
                                                                                 outfile_unmasked=unmasked_path,
                                                                                 normalizer_sequence=None,
                                                                                 replacement_model='roberta-base')
            # masked_df, unmasked_df = create_masked_replacements(input_df=data,
            #                                                                      outfile_masked="outputs/gao_masked.csv",
            #                                                                      outfile_unmasked=unmasked_path,
            #                                                                      normalizer_sequence=None,
            #                                                                      replacement_model='roberta-base')

        ate_dict = getATEScores(input_unmasked_df=unmasked_df, input_classifier=input_classifier,
                                input_vectorizer=vectorizer,
                                output_unmasked_file_with_scores="outputs/zampieri_unmasked_with_scores_" + classifier_name + ".csv",
                                output_ate_csv_file="outputs/zampieri_scores_" + classifier_name + ".csv")
        # ate_dict = getATEScores(input_unmasked_df=unmasked_df, input_classifier=input_classifier,
        #                         input_vectorizer=vectorizer,
        #                         output_unmasked_file_with_scores="outputs/gao_unmasked_with_scores_" + classifier_name + ".csv",
        #                         output_ate_csv_file="outputs/gao_scores_" + classifier_name + ".csv")

        test_array = ["female", "black", "gay", "hispanic", "african"]
        X_test = vectorizer.transform(test_array)
        predictions = input_classifier.predict_proba(X_test)
        ate_predictions = [ate_dict[x] for x in test_array]

        print("-"*50)
        print("Results for classifier: "+classifier_name)
        print("-" * 50)

        pretty_print_three_preds(test_array, predictions, ate_predictions)

    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))
