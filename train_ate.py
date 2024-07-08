import random
import torch
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

# Generate perturbation data using the trained regular model
def generate_perturbation_data(model1, train_loader, vocab, device, num_perturbations=10, batch_size=64):
    model1.to(device)
    model1.eval()
    perturbed_sentences = []

    with torch.no_grad():
        for inputs, targets, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore lengths
            inputs = inputs.to(device)

            for input_sentence in inputs:
                sentence_str = ' '.join([vocab.get_itos()[idx] for idx in input_sentence if idx != vocab["<pad>"]])
                perturbed_sentences.extend(perturb_sentence(sentence_str, vocab.get_itos(), num_perturbations))

    perturbed_inputs = []
    for _, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
        perturbed_input = torch.tensor(vocab(tokenizer(perturbed_tokens_as_sentence)), dtype=torch.int64)
        rejoined_input = torch.tensor(vocab(tokenizer(rejoined_sentence)), dtype=torch.int64)
        perturbed_inputs.append((perturbed_input, rejoined_input))

    # Create DataLoader for perturbed sentences
    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False)

    new_batch_inputs = []
    new_batch_outputs = []

    with torch.no_grad():
        for perturbed_batch in tqdm(perturbed_loader, desc="Computing Scores for Perturbed Sentences"):
            perturbed_input_batch = torch.stack([x[0] for x in perturbed_batch]).to(device).long()
            rejoined_input_batch = torch.stack([x[1] for x in perturbed_batch]).to(device).long()

            perturbed_scores = model1(perturbed_input_batch).cpu().numpy()
            rejoined_scores = model1(rejoined_input_batch).cpu().numpy()

            for perturbed_input, perturbed_score, rejoined_score in zip(perturbed_input_batch, perturbed_scores, rejoined_scores):
                change_in_score = rejoined_score - perturbed_score
                new_batch_inputs.append(perturbed_input.cpu())
                new_batch_outputs.append(change_in_score)

    return new_batch_inputs, new_batch_outputs
