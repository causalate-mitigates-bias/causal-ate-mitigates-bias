import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from nltk.util import ngrams

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# Perturb sentence function
def perturb_sentence(original_input, vocab, num_perturbations=10):
    tokens = original_input.split()
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
        perturbed_part_of_input = ' '.join(perturbed_tokens)
        input_after_perturbation = ' '.join(sentence_as_tokens)
        # print(f"perturbed_part_of_input={perturbed_part_of_input}")
        perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input))
    return perturbed_sentences

#
# def perturb_sentence(original_input, vocab, num_perturbations=25):
#     tokens = original_input.split()
#     perturbed_sentences = []
#     for _ in range(num_perturbations):
#         sentence_as_tokens = tokens[:]
#         perturbed_tokens = []
#         # Randomly select the n-gram size (1-5)
#         n = random.randint(1, 5)
#         # Create n-grams from the sentence
#         sentence_ngrams = list(ngrams(sentence_as_tokens, n))
#         if not sentence_ngrams:
#             continue
#         # Randomly select an n-gram to replace
#         idx = random.randint(0, len(sentence_ngrams) - 1)
#         ngram_to_replace = sentence_ngrams[idx]
#         # Find the start index of the selected n-gram
#         start_idx = idx
#         # Replace the n-gram with random tokens from the vocabulary
#         for j in range(len(ngram_to_replace)):
#             sentence_as_tokens[start_idx + j] = vocab[random.randint(0, len(vocab) - 1)]
#         # Capture the original tokens that were replaced
#         perturbed_tokens = list(ngram_to_replace)
#         perturbed_part_of_input = ' '.join(perturbed_tokens)
#         input_after_perturbation = ' '.join(sentence_as_tokens)
#         # print(f"perturbed_part_of_input={perturbed_part_of_input}")
#         perturbed_sentences.append((original_input, input_after_perturbation, perturbed_part_of_input))
#     return perturbed_sentences

def convert_perturbed_inputs_to_tensors(perturbed_sentences, vocab):
    perturbed_inputs = []
    for original_input, input_after_perturbation, perturbed_part_of_input in tqdm(perturbed_sentences):
        tokenized_original_input = torch.tensor(vocab(tokenizer(original_input)), dtype=torch.int64)
        tokenized_input_after_perturbation = torch.tensor(vocab(tokenizer(input_after_perturbation)), dtype=torch.int64)
        tokenized_perturbed_part_of_input = torch.tensor(vocab(tokenizer(perturbed_part_of_input)), dtype=torch.int64)
        perturbed_inputs.append((tokenized_original_input,
                                 tokenized_input_after_perturbation,
                                 tokenized_perturbed_part_of_input))
    return perturbed_inputs

# Generate perturbed DataLoader
def generate_perturbed_dataloader(train_loader, vocab, batch_size=64, num_perturbations=10):
    perturbed_sentences = []
    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        for sentence in text_list:
            perturbed_sentences.extend(perturb_sentence(sentence, vocab.get_itos(), num_perturbations))
    perturbed_inputs = convert_perturbed_inputs_to_tensors(perturbed_sentences, vocab)
    # Create DataLoader for perturbed sentences
    def collate_fn(batch):
        tokenized_orig_batch, tokenized_iap_batch, tokenized_ppi_batch = zip(*batch)
        orig_batch_padded = pad_sequence(tokenized_orig_batch, batch_first=True, padding_value=vocab["<pad>"])
        iap_batch_padded = pad_sequence(tokenized_iap_batch, batch_first=True, padding_value=vocab["<pad>"])
        ppi_batch_padded = pad_sequence(tokenized_ppi_batch, batch_first=True, padding_value=vocab["<pad>"])
        return orig_batch_padded, iap_batch_padded, ppi_batch_padded

    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return perturbed_loader


# Compute scores for perturbed sentences
def compute_perturbation_scores(model, perturbed_loader, device, padding_value=0, change_threshold=0.3):
    model.to(device)
    model.eval()
    new_batch_inputs = []
    new_batch_outputs = []
    with torch.no_grad():
        for oi_batch, iap_batch, ppi_batch in tqdm(perturbed_loader, desc="Computing Scores for Perturbed Sentences"):
            oi_batch = oi_batch.to(device).long()
            iap_batch = iap_batch.to(device).long()
            perturbed_part_of_input_batch = ppi_batch.to(device).long()
            original_input_scores = model(oi_batch).cpu().numpy()
            input_after_perturbation_scores = model(iap_batch).cpu().numpy()
            for perturbed_part_of_input, original_input_score, input_after_perturbation_score in zip(
                    perturbed_part_of_input_batch, original_input_scores, input_after_perturbation_scores):
                change_in_score = original_input_score - input_after_perturbation_score
                if abs(change_in_score) < change_threshold:
                    continue
                elif change_in_score > change_threshold:
                    change_in_score_class = 1
                else:
                    change_in_score_class = 0
                # change_in_score_class = 1 if change_in_score > change_threshold else 0
                new_batch_inputs.append(perturbed_part_of_input.cpu())
                new_batch_outputs.append(change_in_score_class)
    # Pad sequences to ensure they have the same length
    new_batch_inputs_padded = pad_sequence(new_batch_inputs, batch_first=True, padding_value=padding_value)
    return new_batch_inputs_padded, new_batch_outputs




def decode_sentence(tensor, vocab):
    # Decode a tensor of token IDs to a sentence
    tokens = [vocab.get_itos()[token] for token in tensor if token != vocab["<pad>"]]
    return ' '.join(tokens)

def view_perturbed_loader(perturbed_loader, vocab):
    # Iterate over perturbed_loader and print the sentences
    for batch_idx, (original_input_batch_padded, input_after_perturbation_batch_padded, perturbed_part_of_input_batch_padded) in enumerate(perturbed_loader):
        print(f"Batch {batch_idx + 1}:")
        for i in range(original_input_batch_padded.size(0)):
            original_sentence = decode_sentence(original_input_batch_padded[i], vocab)
            perturbed_sentence = decode_sentence(input_after_perturbation_batch_padded[i], vocab)
            perturbed_part = decode_sentence(perturbed_part_of_input_batch_padded[i], vocab)
            print(f"Original Sentence: {original_sentence}")
            print(f"Perturbed Sentence: {perturbed_sentence}")
            print(f"Perturbed Part: {perturbed_part}")
            print("-" * 50)

def generate_sklearn_perturbation_data(model, vectorizer, train_loader, num_perturbations=10):
    perturbed_sentences = []
    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):
        for sentence in text_list:
            perturbed_sentences.extend(
                perturb_sentence(sentence, vectorizer.get_feature_names_out(), num_perturbations))

    perturbed_inputs = []
    for _, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
        perturbed_input = vectorizer.transform([perturbed_tokens_as_sentence])
        rejoined_input = vectorizer.transform([rejoined_sentence])
        perturbed_inputs.append((perturbed_input, rejoined_input))

    return perturbed_inputs


# Generate perturbed DataLoader
def generate_perturbed_dataloader_sklearn(train_loader, vocab, batch_size=64, num_perturbations=10):
    perturbed_sentences = []
    for text_list, _ in tqdm(train_loader, desc="Generating Perturbed Sentences"):  # Ignore labels
        for sentence in text_list:
            perturbed_sentences.extend(perturb_sentence(sentence, vocab.get_itos(), num_perturbations))

    perturbed_inputs = []
    for original_sentence, rejoined_sentence, perturbed_tokens_as_sentence in perturbed_sentences:
        perturbed_inputs.append((original_sentence, rejoined_sentence, perturbed_tokens_as_sentence))

    # Create DataLoader for perturbed sentences
    def collate_fn(batch):
        original_batch, rejoined_batch, perturbed_tokens_batch = zip(*batch)
        return list(original_batch), list(rejoined_batch), list(perturbed_tokens_batch)

    perturbed_loader = DataLoader(perturbed_inputs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return perturbed_loader


def compute_sklearn_perturbation_scores(model, perturbed_inputs):
    new_batch_inputs = []
    new_batch_outputs = []

    for original_input, rejoined_input, perturbed_tokens_input in tqdm(perturbed_inputs,
                                                                       desc="Computing Scores for Perturbed Sentences"):
        original_score = model.predict_proba(original_input)[0, 1]
        rejoined_score = model.predict_proba(rejoined_input)[0, 1]
        change_in_score = rejoined_score - original_score

        new_batch_inputs.append(perturbed_tokens_input.toarray()[0])
        new_batch_outputs.append(change_in_score)

    return new_batch_inputs, new_batch_outputs
