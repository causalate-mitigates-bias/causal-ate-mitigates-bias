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
from utilities.train import train
from utilities.test import test_nn
from utilities.generate_ate_data import generate_perturbed_dataloader, tokenize_and_pad
from utilities.generate_ate_data import compute_training_batch_filtered_using_model_scores, create_training_dataloader
from utilities.general_utils import set_all_seeds, save_dataloader, load_dataloader
from utilities.train_ate import train_ate_model

set_all_seeds(42)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    model = SimpleNN(vocab_size=len(vocab), embed_dim=100, output_dim=1)
    # Load IMDB data
    train_loader, test_loader = load_imdb_reviews(batch_size=64)
    model_save_path = "saved/models/simplenn_imdb_model.pt"

    criterion = nn.BCELoss()
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Training the regular model...")
        # train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
        train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

        # Save the trained model
        torch.save(model.state_dict(), model_save_path)
        print("simple NN IMDB model saved.")

    # Test the regular model
    print("Testing the regular model...")
    test_accuracy = test_nn(model, test_loader, criterion, device)
    print(f"model test accuracy: {test_accuracy:.2f}%")

    tokenized_loader_path = 'saved/data_loaders/tokenized_loader.pt'

    if os.path.exists(tokenized_loader_path):
        print("Loading tokenized loader from file...")
        tokenized_loader = load_dataloader(tokenized_loader_path, batch_size=64)
    else:
        # Load IMDB data with text strings for perturbation
        text_train_loader, _ = load_imdb_reviews_with_text(batch_size=64)

        # Example usage
        text_train_loader, _ = load_imdb_reviews_with_text(batch_size=64)
        perturbed_loader = generate_perturbed_dataloader(text_train_loader,
                                                         vocab,
                                                         batch_size=64,
                                                         perturbation_rate=0.5,
                                                         num_perturbations=50)

        # View the perturbed sentences
        # for original, perturbed, perturbed_part in perturbed_loader:
        #     for orig, pert, pert_part in zip(original, perturbed, perturbed_part):
        #         print(f"Original: {orig}")
        #         print(f"Perturbed: {pert}")
        #         print(f"Perturbed Part: {pert_part}")
        #         print("-" * 50)

        tokenized_loader = tokenize_and_pad(perturbed_loader, vocab, batch_size=64)

        # Save the tokenized loader
        save_dataloader(tokenized_loader, tokenized_loader_path)



    # Compute the score differences
    training_data = compute_training_batch_filtered_using_model_scores(model, tokenized_loader, device,
                                                                       change_threshold=0.5)

    ate_training_data_loader = create_training_dataloader(training_data, vocab, batch_size=64)

    # Define and train the ATE model
    ate_model = SimpleNN(vocab_size=len(vocab), embed_dim=100, output_dim=1)
    ate_criterion = nn.BCELoss()
    ate_optimizer = optim.Adam(ate_model.parameters(), lr=0.001)

    print("Training the ATE model...")
    train_ate_model(ate_model, ate_training_data_loader, ate_criterion, ate_optimizer, device, num_epochs=num_epochs)

    ate_model_save_path = 'saved/models/ate_model.pt'
    # Save the trained model
    torch.save(model.state_dict(), ate_model_save_path)
    print("ATE based IMDB model saved.")

    # Test the ATE model
    print("Testing the ATE model...")
    test_accuracy = test_nn(ate_model, test_loader, ate_criterion, device)
    print(f"ATE model test accuracy: {test_accuracy:.2f}%")



if __name__ == "__main__":
    main()
