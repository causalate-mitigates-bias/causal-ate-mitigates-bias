import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# Perturb sentence function
def perturb_sentence(sentence, vocab, num_perturbations=10):
    tokens = sentence.split()
    perturbed_sentences = []
    for _ in range(num_perturbations):
        sentence_as_tokens = tokens[:]
        perturbed_tokens = []
        for _ in range(random.randint(1, len(tokens))):  # Randomly decide how many tokens to perturb
            idx = random.randint(0, len(tokens) - 1)
            old_token = sentence_as_tokens[idx]
            new_token = vocab[random.randint(0, len(vocab) - 1)]  # Replace with random token from vocab
            sentence_as_tokens[idx] = new_token
            perturbed_tokens.append(old_token)
        perturbed_tokens_as_sentence = ' '.join(perturbed_tokens)
        rejoined_sentence = ' '.join(sentence_as_tokens)
        perturbed_sentences.append((sentence, rejoined_sentence, perturbed_tokens_as_sentence))

    return perturbed_sentences


# Generate perturbed DataLoader
def generate_perturbed_dataloader(train_loader, vocab, batch_size=64, num_perturbations=10):
    perturbed_sentences = []

    num = 0
    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        if num > 10:
            break
        num += 1
        for sentence in text_list:
            perturbed_sentences.extend(perturb_sentence(sentence, vocab.get_itos(), num_perturbations))

    perturbed_inputs = []
    for _, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
        perturbed_input = torch.tensor(vocab(tokenizer(perturbed_tokens_as_sentence)), dtype=torch.int64)
        rejoined_input = torch.tensor(vocab(tokenizer(rejoined_sentence)), dtype=torch.int64)
        perturbed_inputs.append((perturbed_input, rejoined_input))

    # Create DataLoader for perturbed sentences
    def collate_fn(batch):
        perturbed_batch, rejoined_batch = zip(*batch)
        perturbed_batch_padded = pad_sequence(perturbed_batch, batch_first=True, padding_value=vocab["<pad>"])
        rejoined_batch_padded = pad_sequence(rejoined_batch, batch_first=True, padding_value=vocab["<pad>"])
        return perturbed_batch_padded, rejoined_batch_padded

    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return perturbed_loader


# # Generate perturbed DataLoader
# def generate_perturbed_dataloader(train_loader, vocab, batch_size=64, num_perturbations=10):
#     perturbed_sentences = []
#     num = 0
#     for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
#         if num > 10:
#             break
#         num += 1
#         for sentence in text_list:
#             perturbed_sentences.extend(perturb_sentence(sentence, vocab.get_itos(), num_perturbations))
#
#     perturbed_inputs = []
#     print("finished creating perturbed_sentences")
#     for _, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
#         perturbed_input = torch.tensor(vocab(tokenizer(perturbed_tokens_as_sentence)), dtype=torch.int64)
#         rejoined_input = torch.tensor(vocab(tokenizer(rejoined_sentence)), dtype=torch.int64)
#         perturbed_inputs.append((perturbed_input, rejoined_input))
#
#     # Create DataLoader for perturbed sentences
#     perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False)
#
#     return perturbed_loader


# Compute scores for perturbed sentences
def compute_perturbation_scores_old(model, perturbed_loader, device):
    model.to(device)
    model.eval()

    new_batch_inputs = []
    new_batch_outputs = []

    with torch.no_grad():
        for perturbed_batch, rejoined_batch in tqdm(perturbed_loader, desc="Computing Scores for Perturbed Sentences"):
            perturbed_input_batch = perturbed_batch.to(device).long()
            rejoined_input_batch = rejoined_batch.to(device).long()

            perturbed_scores = model(perturbed_input_batch).cpu().numpy()
            rejoined_scores = model(rejoined_input_batch).cpu().numpy()

            for perturbed_input, perturbed_score, rejoined_score in zip(perturbed_input_batch, perturbed_scores,
                                                                        rejoined_scores):
                change_in_score = rejoined_score - perturbed_score
                new_batch_inputs.append(perturbed_input.cpu())
                new_batch_outputs.append(change_in_score)

    return new_batch_inputs, new_batch_outputs


# Compute scores for perturbed sentences
def compute_perturbation_scores(model, perturbed_loader, device, padding_value=0):
    model.to(device)
    model.eval()

    new_batch_inputs = []
    new_batch_outputs = []

    with torch.no_grad():
        for perturbed_batch, rejoined_batch in tqdm(perturbed_loader, desc="Computing Scores for Perturbed Sentences"):
            perturbed_input_batch = perturbed_batch.to(device).long()
            rejoined_input_batch = rejoined_batch.to(device).long()

            perturbed_scores = model(perturbed_input_batch).cpu().numpy()
            rejoined_scores = model(rejoined_input_batch).cpu().numpy()

            for perturbed_input, perturbed_score, rejoined_score in zip(perturbed_input_batch, perturbed_scores,
                                                                        rejoined_scores):
                change_in_score = rejoined_score - perturbed_score
                new_batch_inputs.append(perturbed_input.cpu())
                new_batch_outputs.append(change_in_score)

    # Pad sequences to ensure they have the same length
    new_batch_inputs_padded = pad_sequence(new_batch_inputs, batch_first=True, padding_value=padding_value)

    return new_batch_inputs_padded, new_batch_outputs


# Define a simple training function for the ATE model
def train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, criterion, optimizer, device, num_epochs=10):
    # Convert new_batch_inputs to a list of tensors
    new_batch_inputs = [torch.tensor(t) for t in new_batch_inputs]

    dataset = TensorDataset(torch.stack(new_batch_inputs).long(), torch.tensor(new_batch_outputs, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    ate_model.to(device)
    ate_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ate_model(inputs.long()).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")
#
#
#
# # Define a simple training function for the ATE model
# def train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, criterion, optimizer, device, num_epochs=10):
#     dataset = TensorDataset(torch.stack(new_batch_inputs).long(), torch.tensor(new_batch_outputs, dtype=torch.float32))
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     ate_model.to(device)
#     ate_model.train()
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#             outputs = ate_model(inputs.long()).squeeze()
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")
