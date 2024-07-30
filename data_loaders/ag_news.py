import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext

torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


# Define a function to yield tokens from the dataset
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# Load your dataset
train_iter, test_iter = AG_NEWS(split=('train', 'test'))

# Build the vocabulary from the training dataset
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])


# Print unique labels for debugging
def print_unique_labels(data, description_string):
    labels = [label for label, _ in data]
    unique_labels = set(labels)
    print(f"Unique labels in the {description_string} data: {unique_labels}")


# Print unique labels in train and test datasets
# print_unique_labels(train_iter, "train")
# print_unique_labels(test_iter, "test")


# Function to convert text to tensor
def text_pipeline(x, vocab):
    return vocab(tokenizer(x))


# Function to convert label to tensor
def label_pipeline(x):
    return x - 1  # AG_NEWS labels are 1, 2, 3, 4. Convert to 0, 1, 2, 3


# Collate function for DataLoader
def collate_batch(batch, vocab=vocab):
        label_list, text_list, lengths = [], [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
            text_list.append(processed_text)
            lengths.append(len(processed_text))
        # Pad sequences
        text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
        label_list = torch.tensor(label_list, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        return text_list, label_list, lengths

# Function to load AG News dataset
def load_dataloaders(batch_size=64, num_training_samples=2500):
    # Convert iterators to lists for DataLoader
    # train_data = list(train_iter)[:num_training_samples] + list(train_iter)[-num_training_samples:]
    train_data = list(train_iter)
    test_data = list(test_iter)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader


# Function to load AG News dataset with text strings for perturbation
def load_dataloaders_with_text(batch_size=64, num_training_samples=250):
    # Convert iterators to lists for DataLoader
    # train_data = list(train_iter)[:num_training_samples] + list(train_iter)[-num_training_samples:]
    train_data = list(train_iter)
    test_data = list(test_iter)

    # Create DataLoader with text strings
    def collate_text_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_pipeline(_label))
            text_list.append(_text)
        label_list = torch.tensor(label_list, dtype=torch.float32)

        return text_list, label_list

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_text_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_text_batch)

    return train_loader, test_loader
