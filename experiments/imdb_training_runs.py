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
from utilities.printing_utilities import train
from utilities.printing_utilities import test_nn
from utilities.printing_utilities import train_ate_model
from utilities.printing_utilities import generate_perturbed_dataloader, compute_perturbation_scores


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load IMDB data
    train_loader, test_loader = load_imdb_reviews(batch_size=64)

    # Define and train the regular model
    vocab_size = len(vocab)
    embed_dim = 100
    output_dim = 1
    model = SimpleNN(vocab_size, embed_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    print("Training the regular model...")
    train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Test the regular model
    print("Testing the regular model...")
    test_accuracy = test_nn(model, test_loader, criterion, device)
    print(f"Regular model test accuracy: {test_accuracy:.2f}%")

    # Load IMDB data with text strings for perturbation
    text_train_loader, _ = load_imdb_reviews_with_text(batch_size=64)

    # Generate perturbed DataLoader
    print("Generating perturbed DataLoader...")
    perturbed_loader = generate_perturbed_dataloader(text_train_loader, vocab, batch_size=64, num_perturbations=10)

    # Compute perturbation scores
    print("Computing perturbation scores...")
    new_batch_inputs, new_batch_outputs = compute_perturbation_scores(model, perturbed_loader, device)

    # Define and train the ATE model
    ate_model = SimpleNN(vocab_size, embed_dim, 1)
    ate_criterion = nn.MSELoss()
    ate_optimizer = optim.Adam(ate_model.parameters(), lr=0.001)

    print("Training the ATE model...")
    train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, ate_criterion, ate_optimizer, device,
                    num_epochs=num_epochs)

    # Test the ATE model
    print("Testing the ATE model...")
    test_accuracy = test_nn(ate_model, test_loader, ate_criterion, device)
    print(f"ATE model test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
