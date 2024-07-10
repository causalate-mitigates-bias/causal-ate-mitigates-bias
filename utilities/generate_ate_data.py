import spacy
import random
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')


# Perturb sentence function
def perturb_sentence(original_input, vocab, perturbation_rate=0.5, num_perturbations=10):
    tokens = original_input.split()
    perturbed_sentences = []
    num_to_perturb = max(1, int(len(tokens) * perturbation_rate))  # Ensure at least one token is perturbed

    for _ in range(num_perturbations):
        sentence_as_tokens = tokens[:]
        perturbed_tokens = []
        perturbed_indices = random.sample(range(len(tokens)), num_to_perturb)

        for idx in perturbed_indices:
            old_token = sentence_as_tokens[idx]
            new_token = vocab[random.randint(0, len(vocab) - 1)]  # Replace with random token from vocab
            sentence_as_tokens[idx] = new_token
            perturbed_tokens.append(old_token)

        perturbed_part_of_input = ' '.join(perturbed_tokens)
        input_after_perturbation = ' '.join(sentence_as_tokens)
        perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input))

    return perturbed_sentences

# Generate perturbed DataLoader
def generate_perturbed_dataloader(train_loader, vocab, batch_size=64, perturbation_rate=0.5, num_perturbations=10):
    perturbed_sentences = []

    num = 0
    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        if num>10:
            break
        num+=1
        for sentence in text_list:
            perturbed_sentences.extend(perturb_sentence(sentence, vocab.get_itos(), perturbation_rate, num_perturbations))

    perturbed_inputs = []
    for original_input, input_after_perturbation, perturbed_part_of_input in perturbed_sentences:
        perturbed_inputs.append((original_input,
                                 input_after_perturbation,
                                 perturbed_part_of_input))

    # Create DataLoader for perturbed sentences
    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch = zip(*batch)
        return list(original_batch), list(perturbed_batch), list(perturbed_part_batch)

    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return perturbed_loader