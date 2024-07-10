import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim

from data_loaders.imdb import load_imdb_reviews, load_imdb_reviews_with_text, vocab
from models.simpleNN import SimpleNN
from utilities.train import train, train_sklearn_model
from utilities.test import test_nn, test_sklearn_model
from utilities.train_ate import train_ate_model, train_sklearn_ate_model
from utilities.generate_ate_data_20240710 import generate_perturbed_dataloader, compute_perturbation_scores
from utilities.generate_ate_data_20240710 import generate_perturbed_dataloader_sklearn, compute_sklearn_perturbation_scores
from utilities.general_utils import set_all_seeds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


set_all_seeds(42)
# Define the models to train and evaluate
models = {
    "SimpleNN": "torch",
    # "SVM": "sklearn",
    # "LogisticRegression": "sklearn"
}

def model_generator(model_name):
    if model_name == "SimpleNN":
        return SimpleNN(vocab_size=len(vocab), embed_dim=100, output_dim=1)
    elif model_name == "SVM":
        return make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', probability=True))
    elif model_name == "LogisticRegression":
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression())
    else:
        raise ValueError(f"Model {model_name} not recognized")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    for model_name, model_type in models.items():
        print(f"Training the {model_name} model...")
        model = model_generator(model_name)
        ate_model = model_generator(model_name)
        if model_type == "torch":
            # Load IMDB data
            train_loader, test_loader = load_imdb_reviews(batch_size=64)
            model_save_path = "saved_models/simplenn_imdb_model.pt"
            criterion = nn.BCELoss()
            if os.path.exists(model_save_path):
                model.load_state_dict(torch.load(model_save_path))
                model.to(device)
            else:
                optimizer = optim.Adam(model.parameters(), lr=0.1)
                print("Training the regular model...")
                # train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
                train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

                # Save the trained model
                torch.save(model.state_dict(), model_save_path)
                print("simple NN IMDB model saved.")

            # Test the regular model
            print("Testing the regular model...")
            test_accuracy = test_nn(model, test_loader, criterion, device)

        else:
            train_loader, test_loader = load_imdb_reviews_with_text(batch_size=64)

            model, vectorizer = train_sklearn_model(model, train_loader, device)
            print("Testing the regular model...")
            test_accuracy = test_sklearn_model(model, vectorizer, test_loader, device)
        print(f"Regular model test accuracy: {test_accuracy:.2f}%")

        # Load IMDB data with text strings for perturbation
        text_train_loader, _ = load_imdb_reviews_with_text(batch_size=64)

        if model_type == "torch":

            # Generate perturbed DataLoader
            print("Generating perturbed DataLoader...")
            perturbed_loader = generate_perturbed_dataloader(text_train_loader, vocab, batch_size=64, num_perturbations=25)

            # Compute perturbation scores
            print("Computing perturbation scores...")
            new_batch_inputs, new_batch_outputs = compute_perturbation_scores(model, perturbed_loader, device)

            # Define and train the ATE model
            ate_criterion = nn.BCELoss()
            ate_optimizer = optim.Adam(ate_model.parameters(), lr=0.1)

            print("Training the ATE model...")
            train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, ate_criterion, ate_optimizer, device, num_epochs=num_epochs)
            # Save the ATE model
            ate_model_save_path = "saved_models/ate_model.pt"
            torch.save(ate_model.state_dict(), ate_model_save_path)
            print("ATE model saved.")

            # Test the ATE model
            print("Testing the ATE model...")
            test_accuracy = test_nn(ate_model, test_loader, ate_criterion, device)

        else:
            # Generate perturbation data
            print(f"Generating perturbation data for {model_name} model...")
            perturbed_loader = generate_perturbed_dataloader_sklearn(text_train_loader, vocab,
                                                                     batch_size=64, num_perturbations=25)


            # Compute perturbation scores
            print(f"Computing perturbation scores for {model_name} model...")
            new_batch_inputs, new_batch_outputs = compute_sklearn_perturbation_scores(model, perturbed_loader)

            # Train the ATE model using the same model type
            print(f"Training the ATE model for {model_name}...")
            ate_model = train_sklearn_ate_model(ate_model, new_batch_inputs, new_batch_outputs,
                                                num_epochs=num_epochs)

            # Test the ATE model
            print("Testing the regular model...")
            test_accuracy = test_sklearn_model(ate_model, vectorizer, test_loader, device)

        print(f"ATE model test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
