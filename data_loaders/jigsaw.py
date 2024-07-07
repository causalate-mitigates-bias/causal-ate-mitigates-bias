import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


class JigsawToxicityDataset(Dataset):
    def __init__(self, data_path, vocab=None):
        self.data = pd.read_csv(data_path)
        self.vocab = vocab if vocab is not None else self.build_vocab()

    def build_vocab(self):
        # Build the vocabulary from the dataset
        vocab = build_vocab_from_iterator(yield_tokens(self.data), specials=["<unk>", "<pad>"])
        vocab.set_default_index(vocab["<unk>"])
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text, label = row['comment_text'], row['toxic']
        return text, label


def yield_tokens(data):
    for text in data['comment_text']:
        yield tokenizer(text)


def text_pipeline(x, vocab):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return 1 if x >= 0.5 else 0


def collate_batch(batch, vocab):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    # Pad sequences
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths


def load_jigsaw_toxicity(batch_size=64, data_path='path_to_jigsaw_toxicity_dataset.csv'):
    dataset = JigsawToxicityDataset(data_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    vocab = dataset.vocab
    collate_fn = lambda batch: collate_batch(batch, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader

