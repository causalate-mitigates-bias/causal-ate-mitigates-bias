import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from data_loaders.imdb import load_imdb_reviews, vocab
from models.simpleNN import SimpleNN
from train import train
from test import test_nn
from train_ate import perturb_sentence, generate_perturbation_data, train_ate_model


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

    # Generate perturbation data
    print("Generating perturbation data...")
    new_batch_inputs, new_batch_outputs = generate_perturbation_data(model, train_loader, vocab, device,
                                                                     num_perturbations=10)

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
