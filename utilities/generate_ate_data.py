import spacy
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def perturb_sentence(original_input, original_label, vocab, perturbation_rate=0.5, num_perturbations=10, n_gram_length=5):
    tokens = original_input.split()
    perturbed_sentences = []
    total_tokens = len(tokens)
    num_to_perturb = max(3, int(total_tokens * perturbation_rate))  # Ensure at least three tokens are perturbed
    num_ngrams = num_to_perturb // n_gram_length + 1 # Number of n-grams to perturb

    for _ in range(num_perturbations):
        sentence_as_tokens = tokens[:]
        perturbed_tokens = []
        perturbed_indices = []

        # Generate random starting indices for the n-grams
        for _ in range(num_ngrams):
            start_idx = random.randint(0, total_tokens - n_gram_length)
            perturbed_indices.extend(range(start_idx, start_idx + n_gram_length))

        # Sort and remove duplicates
        perturbed_indices = sorted(set(perturbed_indices))

        for idx in perturbed_indices:
            old_token = sentence_as_tokens[idx]
            new_token = vocab[random.randint(0, len(vocab) - 1)]  # Replace with random token from vocab
            sentence_as_tokens[idx] = new_token
            perturbed_tokens.append(old_token)

        perturbed_part_of_input = ' '.join(perturbed_tokens)
        input_after_perturbation = ' '.join(sentence_as_tokens)
        perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input, original_label))

    return perturbed_sentences

# Generate perturbed DataLoader
def generate_perturbed_dataloader(train_loader, vocab, batch_size=64,
                                  perturbation_rate=0.5, num_perturbations=10, n_gram_length=5):
    perturbed_sentences = []

    for text_list, label_list in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        for sentence, label in zip(text_list,label_list):
            perturbed_sentences.extend(
                perturb_sentence(sentence, label,
                                 vocab.get_itos(),
                                 perturbation_rate,
                                 num_perturbations, n_gram_length=n_gram_length))

    perturbed_inputs = []
    for original_input, input_after_perturbation, perturbed_part_of_input, original_label in perturbed_sentences:
        perturbed_inputs.append((original_input,
                                 input_after_perturbation,
                                 perturbed_part_of_input,
                                 original_label))

    # Create DataLoader for perturbed sentences
    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch, original_label_batch = zip(*batch)
        return list(original_batch), list(perturbed_batch), list(perturbed_part_batch), list(original_label_batch)

    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return perturbed_loader


def tokenize_and_pad(perturbed_loader, vocab, batch_size=64):
    tokenized_inputs = []

    for original_batch, perturbed_batch, perturbed_part_batch, original_label_batch in tqdm(perturbed_loader,
                                                                                            desc="Tokenizing and Padding"):
        for original_input, input_after_perturbation, perturbed_part_of_input, original_label in zip(original_batch,
                                                                                                     perturbed_batch,
                                                                                                     perturbed_part_batch,
                                                                                                     original_label_batch):
            tokenized_original_input = torch.tensor(vocab(tokenizer(original_input)), dtype=torch.int64)
            tokenized_input_after_perturbation = torch.tensor(vocab(tokenizer(input_after_perturbation)),
                                                              dtype=torch.int64)
            tokenized_perturbed_part_of_input = torch.tensor(vocab(tokenizer(perturbed_part_of_input)),
                                                             dtype=torch.int64)

            tokenized_inputs.append(
                (tokenized_original_input, tokenized_input_after_perturbation, tokenized_perturbed_part_of_input,
                 original_label))

    # Create DataLoader for tokenized sentences
    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch, original_labels_batch = zip(*batch)

        original_batch_padded = pad_sequence(original_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_batch_padded = pad_sequence(perturbed_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_part_batch_padded = pad_sequence(perturbed_part_batch, batch_first=True, padding_value=vocab["<pad>"])
        # original_labels_batch = torch.stack(original_label_batch)
        # print(f"original_label_batch: {original_labels_batch}")
        return original_batch_padded, perturbed_batch_padded, perturbed_part_batch_padded, original_labels_batch

    tokenized_loader = DataLoader(tokenized_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return tokenized_loader



def compute_training_batch_filtered_using_model_scores(model, tokenized_loader, device,
                                                                   change_threshold=0.3):
    model.to(device)
    model.eval()
    training_batch_filtered = []
    with torch.no_grad():
        for original_batch, perturbed_batch, perturbed_part_batch, original_labels_batch in \
                tqdm(tokenized_loader, desc="Computing Scores"):
            original_batch = original_batch.to(device).long()
            perturbed_batch = perturbed_batch.to(device).long()

            original_lengths = torch.tensor([len(x) for x in original_batch], dtype=torch.int64).cpu()
            perturbed_lengths = torch.tensor([len(x) for x in perturbed_batch], dtype=torch.int64).cpu()

            original_scores = model(original_batch, original_lengths).cpu().numpy()
            perturbed_scores = model(perturbed_batch, perturbed_lengths).cpu().numpy()

            for original_score, perturbed_score, perturbed_sentence, original_label in \
                    zip(original_scores, perturbed_scores, perturbed_part_batch, original_labels_batch):
                score_differences = abs(original_score - perturbed_score)
                max_change = max(score_differences)
                if max_change >= change_threshold:
                    change_in_score_class = original_label
                    training_batch_filtered.append((perturbed_sentence, change_in_score_class))

    return training_batch_filtered

def create_training_dataloader(training_data, vocab, batch_size=32, n_classes=4):
    # Convert the sentences and labels to tensors
    inputs = [torch.tensor(t) for t, _ in training_data]
    outputs = [label if isinstance(label, torch.Tensor) else torch.tensor([label], dtype=torch.int64) for _, label in training_data]

    # Filter out empty sequences
    non_empty_indices = [i for i, t in enumerate(inputs) if len(t) > 0]
    inputs = [inputs[i] for i in non_empty_indices]
    outputs = [outputs[i] for i in non_empty_indices]

    # Validate target labels
    valid_indices = [i for i, label in enumerate(outputs) if 0 <= label.item() < n_classes]
    inputs = [inputs[i] for i in valid_indices]
    outputs = [outputs[i] for i in valid_indices]

    # Calculate lengths of the sequences
    lengths = [len(seq) for seq in inputs]
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

    # Pad the sequences to ensure they have the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    outputs_tensor = torch.stack(outputs).squeeze().long()  # Ensure outputs are Long tensors

    # Create a TensorDataset
    dataset = TensorDataset(inputs_padded.long(), lengths_tensor, outputs_tensor)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader