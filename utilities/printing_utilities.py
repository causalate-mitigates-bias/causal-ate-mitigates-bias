import pandas as pd
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase
from nltk.tokenize import word_tokenize
import time
import numpy as np

def pretty_print_three_preds(input_list, predictions, ate_scores):
    # Header row
    print("=" * 75)  # Separator line
    print("||\tInput\t|\tClassifier Pred\t|\tATE Scores\t||")
    print("=" * 75)  # Separator line
    # print("||\tword\t|\tprob of toxicity\t||")
    for i in range(len(input_list)):
        # Convert the prediction value to a regular Python float
        prediction_value = float(predictions[i][1])
        ate_score = float(ate_scores[i])

        # print("||\t", input_list[i],"\t|\t",predictions[i][1],"\t||")
        # Format the output with fixed column widths and right-aligned content
        formatted_output = f"||\t{input_list[i]:<10}\t|\t{prediction_value:.6f}\t|\t{ate_score:.6f}\t||"
        print(formatted_output)
    print("=" * 75)  # Separator line
    print("")



if __name__=="__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducibility
    np.random.seed(8)
    # Set NumPy print options for displaying floating-point numbers
    np.set_printoptions(precision=6, suppress=True, linewidth=200)


    # Read the CSV file into a Pandas DataFrame named 'gao'
    gao = pd.read_csv("../datasets/gao.csv")

    # Print the column names of the 'gao' DataFrame
    print("gao.columns =", gao.columns)

    # Print the first 5 rows of the 'gao' DataFrame
    print("gao.head(5) =", gao.head(5))

    # Get the length (number of rows) of the 'gao' DataFrame and print it
    print("len(gao) =", len(gao))

    # Filter for rows where the "labels" column is equal to 1
    toxic_dao = gao[gao['labels'] == 1]

    # Get the length (number of rows) of the 'toxic_dao' DataFrame and print it
    print("len(toxic_dao) =", len(toxic_dao))

    # Read the CSV file into a Pandas DataFrame named 'zampieri'
    zampieri = pd.read_csv("../datasets/zampieri.csv")

    # Print the column names of the 'zampieri' DataFrame
    print("zampieri.columns =", zampieri.columns)

    print("zampieri.head(5)=",zampieri.head(5))
    print("len(zampieri)=", len(zampieri))

    toxic_zampieri = zampieri[zampieri["labels"] == 1]
    print("len(toxic_zampieri) =", len(toxic_zampieri))



    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))

