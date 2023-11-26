import pandas as pd
from tokenizers.normalizers import NFD, StripAccents, Strip, Lowercase
from nltk.tokenize import word_tokenize




if __name__=="__main__":
    # Record the start time
    start_time = time.time()
    # Set a random seed for reproducibility
    np.random.seed(8)
    # Set NumPy print options for displaying floating-point numbers
    np.set_printoptions(precision=6, suppress=True, linewidth=200)


    # Read the CSV file into a Pandas DataFrame named 'gao'
    gao = pd.read_csv("datasets/gao.csv")

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
    zampieri = pd.read_csv("datasets/zampieri.csv")

    # Print the column names of the 'zampieri' DataFrame
    print("zampieri.columns =", zampieri.columns)

    print("zampieri.head(5)=",zampieri.head(5))
    print("len(zampieri)=", len(zampieri))

    toxic_zampieri = zampieri[zampieri["labels"] == 1]
    print("len(toxic_zampieri) =", len(toxic_zampieri))



    # Calculate and print the total time taken to run the code
    print("time taken to run = %0.6f seconds" % (time.time() - start_time))

