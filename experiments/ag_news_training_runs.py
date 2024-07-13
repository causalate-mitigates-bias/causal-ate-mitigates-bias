import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim

from data_loaders.ag_news import load_dataloaders, load_dataloaders_with_text, vocab
from models.simpleRNN import SimpleRNN
from utilities.train import train
from utilities.test import test_nn, test_ate_nn, test_avg_nn_with_length
from utilities.generate_ate_data import generate_perturbed_dataloader, tokenize_and_pad
from utilities.generate_ate_data import compute_training_batch_filtered_using_model_scores
from utilities.generate_ate_data import create_training_dataloader
from utilities.general_utils import set_all_seeds, save_tokenized_loader_with_length, load_tokenized_loader_with_length
from utilities.train_ate import train_ate_model_with_length

set_all_seeds(42)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    output_dim = 4  # AG News has 4 classes
    pad_idx = vocab["<pad>"]
    model = SimpleRNN(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=output_dim, pad_idx=pad_idx)
    model.to(device)
    # Load AG News data
    train_loader, test_loader = load_dataloaders(batch_size=64)
    model_save_path = "saved/models/simplernn_ag_news_model.pt"

    criterion = nn.CrossEntropyLoss()
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print("Training the regular model...")
        train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)

        # Save the trained model
        torch.save(model.state_dict(), model_save_path)
        print("SimpleRNN AG News model saved.")

    # Test the regular model
    print("Testing the regular model...")
    test_accuracy = test_nn(model, test_loader, criterion, device)
    print(f"Model test accuracy: {test_accuracy:.2f}%")

    # Load AG News data with text strings for perturbation
    text_train_loader, text_test_loader = load_dataloaders_with_text(batch_size=64)
    tokenized_loader_path = 'saved/data_loaders/tokenized_loader_agnews.pt'

    if os.path.exists(tokenized_loader_path):
        print("Loading tokenized loader from file...")
        tokenized_loader = load_tokenized_loader_with_length(tokenized_loader_path)
    else:
        perturbed_loader = generate_perturbed_dataloader(text_train_loader,
                                                         vocab,
                                                         batch_size=64,
                                                         perturbation_rate=0.5,
                                                         num_perturbations=50,
                                                         n_gram_length=5)

        tokenized_loader = tokenize_and_pad(perturbed_loader, vocab, batch_size=64)

        # Save the tokenized loader
        # save_tokenized_loader_with_length(tokenized_loader, tokenized_loader_path)

    ate_model_save_path = 'saved/models/ate_model_agnews.pt'
    # Define and train the ATE model
    ate_model = SimpleRNN(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=output_dim, pad_idx=pad_idx)
    ate_criterion = nn.CrossEntropyLoss()
    if os.path.exists(ate_model_save_path):
        print("Loading the ATE model...")
        ate_model.load_state_dict(torch.load(ate_model_save_path))
        ate_model.to(device)
    else:
        # Compute the score differences
        training_data = compute_training_batch_filtered_using_model_scores(model, tokenized_loader, device,
                                                                           change_threshold=1.0/output_dim)
        # Create training data loader
        ate_training_data_loader = create_training_dataloader(training_data, vocab,
                                                                          batch_size=64, n_classes=output_dim)

        ate_optimizer = optim.Adam(ate_model.parameters(), lr=0.0003)

        print("Training the ATE model...")
        train_ate_model_with_length(ate_model, ate_training_data_loader, ate_criterion, ate_optimizer, device, num_epochs=num_epochs)

        # Save the trained model
        torch.save(ate_model.state_dict(), ate_model_save_path)
        print("ATE based AG News model saved.")

    # Test the ATE model
    print("Testing the ATE model...")
    test_accuracy = test_nn(ate_model, test_loader, ate_criterion, device)
    print(f"ATE model test accuracy: {test_accuracy:.2f}%")
    test_accuracy = test_avg_nn_with_length(ate_model, model, test_loader, ate_criterion, device)
    print(f"Average model test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()
