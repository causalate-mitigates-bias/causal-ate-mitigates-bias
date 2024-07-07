import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer

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

# Prepare perturbation tensor function
def prepare_perturbation_tensor(perturbation, vocab):
    perturbation_vector = []
    for idx, old_token, new_token in perturbation:
        perturbation_vector.extend([idx, vocab[old_token], vocab[new_token]])
    return torch.tensor(perturbation_vector, dtype=torch.float32)

# Generate perturbation data using the trained regular model
def generate_perturbation_data(model1, train_loader, vocab, device, num_perturbations=10):
    model1.to(device)
    model1.eval()
    new_batch_inputs = []
    new_batch_outputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(train_loader, desc="Generating Perturbed Batch"):
            inputs = inputs.to(device)
            original_outputs = model1(inputs).detach()

            for i, input_sentence in enumerate(inputs):
                sentence_str = ' '.join([vocab.get_itos()[idx] for idx in input_sentence if idx != vocab["<pad>"]])
                perturbed_sentences = perturb_sentence(sentence_str, vocab.get_itos(), num_perturbations)

                for original_sentence, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
                    perturbed_input = torch.tensor(vocab(tokenizer(perturbed_tokens_as_sentence)), dtype=torch.int64).unsqueeze(0).to(device)
                    rejoined_input = torch.tensor(vocab(tokenizer(rejoined_sentence)), dtype=torch.int64).unsqueeze(0).to(device)

                    original_score = model1(torch.tensor(vocab(tokenizer(original_sentence)), dtype=torch.int64).unsqueeze(0).to(device)).item()
                    perturbed_score = model1(rejoined_input).item()
                    change_in_score = original_score - perturbed_score

                    new_batch_inputs.append(perturbed_input)
                    new_batch_outputs.append(change_in_score)

    return new_batch_inputs, new_batch_outputs

# Define a simple training function for the ATE model
def train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, criterion, optimizer, device, num_epochs=10):
    dataset = TensorDataset(torch.stack(new_batch_inputs), torch.tensor(new_batch_outputs, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    ate_model.to(device)
    ate_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ate_model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")



# Example usage
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model1 = ...  # Your trained model
# train_loader = ...  # Your DataLoader
# vocab = ...  # Your vocabulary
# num_perturbations = 10

# Generate perturbation data
new_batch_inputs, new_batch_outputs = generate_perturbation_data(model1, train_loader, vocab, device, num_perturbations)

# Define and train the ATE model
input_dim = new_batch_inputs[0].shape[1]  # Assuming all inputs have the same shape
ate_model = ATEModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(ate_model.parameters(), lr=0.001)

train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, criterion, optimizer, device, num_epochs=10)
