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


# # Perturb sentence function
# def perturb_sentence_old(original_input, vocab, perturbation_rate=0.5, num_perturbations=10):
#     tokens = original_input.split()
#     perturbed_sentences = []
#     num_to_perturb = max(3, int(len(tokens) * perturbation_rate))  # Ensure at least three tokens is perturbed
#
#     for _ in range(num_perturbations):
#         sentence_as_tokens = tokens[:]
#         perturbed_tokens = []
#         perturbed_indices = random.sample(range(len(tokens)), num_to_perturb)
#         perturbed_indices.sort()
#         for idx in perturbed_indices:
#             old_token = sentence_as_tokens[idx]
#             new_token = vocab[random.randint(0, len(vocab) - 1)]  # Replace with random token from vocab
#             sentence_as_tokens[idx] = new_token
#             perturbed_tokens.append(old_token)
#
#         perturbed_part_of_input = ' '.join(perturbed_tokens)
#         input_after_perturbation = ' '.join(sentence_as_tokens)
#         perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input))
#
#     return perturbed_sentences

def perturb_sentence(original_input, vocab, perturbation_rate=0.5, num_perturbations=10, n_gram_length=5):
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
        perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input))

    return perturbed_sentences

# Generate perturbed DataLoader
def generate_perturbed_dataloader(train_loader, vocab, batch_size=64,
                                  perturbation_rate=0.5, num_perturbations=10, n_gram_length=5):
    perturbed_sentences = []

    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        for sentence in text_list:
            perturbed_sentences.extend(
                perturb_sentence(sentence,
                                 vocab.get_itos(),
                                 perturbation_rate,
                                 num_perturbations, n_gram_length=n_gram_length))

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


def tokenize_and_pad(perturbed_loader, vocab, batch_size=64):
    tokenized_inputs = []

    for original_batch, perturbed_batch, perturbed_part_batch in tqdm(perturbed_loader, desc="Tokenizing and Padding"):
        for original_input, input_after_perturbation, perturbed_part_of_input in zip(original_batch, perturbed_batch,
                                                                                     perturbed_part_batch):
            tokenized_original_input = torch.tensor(vocab(tokenizer(original_input)), dtype=torch.int64)
            tokenized_input_after_perturbation = torch.tensor(vocab(tokenizer(input_after_perturbation)),
                                                              dtype=torch.int64)
            tokenized_perturbed_part_of_input = torch.tensor(vocab(tokenizer(perturbed_part_of_input)),
                                                             dtype=torch.int64)

            tokenized_inputs.append(
                (tokenized_original_input, tokenized_input_after_perturbation, tokenized_perturbed_part_of_input))

    # Create DataLoader for tokenized sentences
    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch = zip(*batch)
        original_batch_padded = pad_sequence(original_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_batch_padded = pad_sequence(perturbed_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_part_batch_padded = pad_sequence(perturbed_part_batch, batch_first=True, padding_value=vocab["<pad>"])

        return original_batch_padded, perturbed_batch_padded, perturbed_part_batch_padded

    tokenized_loader = DataLoader(tokenized_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return tokenized_loader


def tokenize_and_pad_with_length(perturbed_loader, vocab, batch_size=64):
    tokenized_inputs = []

    for original_batch, perturbed_batch, perturbed_part_batch in tqdm(perturbed_loader, desc="Tokenizing and Padding"):
        for original_input, input_after_perturbation, perturbed_part_of_input in zip(original_batch, perturbed_batch,
                                                                                     perturbed_part_batch):
            tokenized_original_input = torch.tensor(vocab(tokenizer(original_input)), dtype=torch.int64)
            tokenized_input_after_perturbation = torch.tensor(vocab(tokenizer(input_after_perturbation)),
                                                              dtype=torch.int64)
            tokenized_perturbed_part_of_input = torch.tensor(vocab(tokenizer(perturbed_part_of_input)),
                                                             dtype=torch.int64)

            tokenized_inputs.append(
                (tokenized_original_input, tokenized_input_after_perturbation, tokenized_perturbed_part_of_input))

    # Create DataLoader for tokenized sentences
    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch = zip(*batch)

        original_lengths = torch.tensor([len(x) for x in original_batch], dtype=torch.int64)
        perturbed_lengths = torch.tensor([len(x) for x in perturbed_batch], dtype=torch.int64)
        perturbed_part_lengths = torch.tensor([len(x) for x in perturbed_part_batch], dtype=torch.int64)

        original_batch_padded = pad_sequence(original_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_batch_padded = pad_sequence(perturbed_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_part_batch_padded = pad_sequence(perturbed_part_batch, batch_first=True, padding_value=vocab["<pad>"])

        return (original_batch_padded, original_lengths,
                perturbed_batch_padded, perturbed_lengths,
                perturbed_part_batch_padded, perturbed_part_lengths)

    tokenized_loader = DataLoader(tokenized_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return tokenized_loader


class TokenizedDataset(Dataset):
    def __init__(self, perturbed_loader, vocab):
        self.perturbed_loader = perturbed_loader
        self.vocab = vocab
        self.tokenized_data = []
        self._generate_tokenized_data()

    def _generate_tokenized_data(self):
        for original_batch, perturbed_batch, perturbed_part_batch in tqdm(self.perturbed_loader, desc="Tokenizing and Padding"):
            for original_input, input_after_perturbation, perturbed_part_of_input in zip(original_batch, perturbed_batch, perturbed_part_batch):
                tokenized_original_input = torch.tensor(self.vocab(tokenizer(original_input)), dtype=torch.int64)
                tokenized_input_after_perturbation = torch.tensor(self.vocab(tokenizer(input_after_perturbation)), dtype=torch.int64)
                tokenized_perturbed_part_of_input = torch.tensor(self.vocab(tokenizer(perturbed_part_of_input)), dtype=torch.int64)
                self.tokenized_data.append((tokenized_original_input, tokenized_input_after_perturbation, tokenized_perturbed_part_of_input))

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]



def tokenize_and_pad_with_length(perturbed_loader, vocab, batch_size=64):
    dataset = TokenizedDataset(perturbed_loader, vocab)

    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch = zip(*batch)

        original_lengths = torch.tensor([len(x) for x in original_batch], dtype=torch.int64)
        perturbed_lengths = torch.tensor([len(x) for x in perturbed_batch], dtype=torch.int64)
        perturbed_part_lengths = torch.tensor([len(x) for x in perturbed_part_batch], dtype=torch.int64)

        original_batch_padded = pad_sequence(original_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_batch_padded = pad_sequence(perturbed_batch, batch_first=True, padding_value=vocab["<pad>"])
        perturbed_part_batch_padded = pad_sequence(perturbed_part_batch, batch_first=True, padding_value=vocab["<pad>"])

        return (original_batch_padded, original_lengths,
                perturbed_batch_padded, perturbed_lengths,
                perturbed_part_batch_padded, perturbed_part_lengths)

    tokenized_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return tokenized_loader

def compute_training_batch_filtered_using_model_scores(model, tokenized_loader, device, change_threshold=0.3):
    model.to(device)
    model.eval()
    training_batch_filtered = []
    with torch.no_grad():
        for original_batch, perturbed_batch, perturbed_part_batch in tqdm(tokenized_loader, desc="Computing Scores"):
            original_batch = original_batch.to(device).long()
            perturbed_batch = perturbed_batch.to(device).long()

            original_scores = model(original_batch).cpu().numpy()
            perturbed_scores = model(perturbed_batch).cpu().numpy()

            for original_score, perturbed_score, perturbed_sentence in \
                    zip(original_scores, perturbed_scores, perturbed_part_batch):
                score_difference = original_score - perturbed_score
                if abs(score_difference) >= change_threshold:
                    change_in_score_class = 1 if score_difference > change_threshold else 0
                    training_batch_filtered.append((perturbed_sentence, change_in_score_class))

    return training_batch_filtered


def compute_training_batch_filtered_using_model_scores_with_length(model, tokenized_loader, device, change_threshold=0.3):
    model.to(device)
    model.eval()
    training_batch_filtered = []
    with torch.no_grad():
        for original_batch, original_lengths, perturbed_batch, perturbed_lengths, perturbed_part_batch, _ in tqdm(tokenized_loader, desc="Computing Scores"):
            original_batch = original_batch.to(device).long()
            perturbed_batch = perturbed_batch.to(device).long()
            original_lengths = original_lengths.cpu().int()  # Ensure lengths are on CPU and int type
            perturbed_lengths = perturbed_lengths.cpu().int()  # Ensure lengths are on CPU and int type

            original_scores = model(original_batch, original_lengths).cpu().numpy()
            perturbed_scores = model(perturbed_batch, perturbed_lengths).cpu().numpy()

            for original_score, perturbed_score, perturbed_sentence in \
                    zip(original_scores, perturbed_scores, perturbed_part_batch):
                score_differences = abs(original_score - perturbed_score)
                max_change = max(score_differences)
                if max_change >= change_threshold:
                    change_in_score_class = score_differences.argmax()
                    training_batch_filtered.append((perturbed_sentence, change_in_score_class))

    return training_batch_filtered

def create_training_dataloader(training_data, vocab, batch_size=32):
    # Convert the sentences and labels to tensors
    # inputs = [t for t, _ in training_data]
    inputs = [torch.tensor(t) for t, _ in training_data]
    outputs = [torch.tensor([label], dtype=torch.float32) for _, label in training_data]

    # Pad the sequences to ensure they have the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    outputs_tensor = torch.tensor(outputs).squeeze()

    # Create a TensorDataset
    dataset = TensorDataset(inputs_padded.long(), outputs_tensor)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def create_training_dataloader_with_length(training_data, vocab, batch_size=32, n_classes=4):
    # Convert the sentences and labels to tensors
    inputs = [torch.tensor(t) for t, _ in training_data]
    outputs = [torch.tensor([label], dtype=torch.int64) for _, label in training_data]  # Ensure labels are int64

    # Filter out empty sequences
    non_empty_indices = [i for i, t in enumerate(inputs) if len(t) > 0]
    inputs = [inputs[i] for i in non_empty_indices]
    outputs = [outputs[i] for i in non_empty_indices]

    # Validate target labels
    valid_outputs = [label for label in outputs if 0 <= label < n_classes]
    valid_indices = [i for i, label in enumerate(outputs) if 0 <= label < n_classes]
    inputs = [inputs[i] for i in valid_indices]
    outputs = valid_outputs

    # Calculate lengths of the sequences
    lengths = [len(seq) for seq in inputs]
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

    # Pad the sequences to ensure they have the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=vocab["<pad>"])
    outputs_tensor = torch.tensor(outputs).squeeze()

    # Create a TensorDataset
    dataset = TensorDataset(inputs_padded.long(), lengths_tensor, outputs_tensor)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader
