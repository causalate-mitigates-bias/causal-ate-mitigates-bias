import torch
import numpy as np
import random
from torch.utils.data import DataLoader


def set_all_seeds(seed):
    """
    Set all seeds for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Function to save the tokenized loader
def save_tokenized_loader(tokenized_loader, filepath):
    all_data = []
    for original_batch, perturbed_batch, perturbed_part_batch in tokenized_loader:
        for orig, pert, pert_part in zip(original_batch, perturbed_batch, perturbed_part_batch):
            all_data.append((orig, pert, pert_part))
    torch.save(all_data, filepath)

def save_tokenized_loader_with_length(tokenized_loader, filepath):
    all_data = []
    for original_batch, original_lengths, perturbed_batch, perturbed_lengths, perturbed_part_batch, perturbed_part_lengths in tokenized_loader:
        for orig, orig_len, pert, pert_len, pert_part, pert_part_len in zip(original_batch, original_lengths, perturbed_batch, perturbed_lengths, perturbed_part_batch, perturbed_part_lengths):
            all_data.append((orig, orig_len, pert, pert_len, pert_part, pert_part_len))
    torch.save(all_data, filepath)


def load_tokenized_loader(filepath, batch_size=64):
    all_data = torch.load(filepath)

    def collate_fn(batch):
        original_batch, perturbed_batch, perturbed_part_batch = zip(*batch)
        return torch.stack(original_batch), torch.stack(perturbed_batch), torch.stack(perturbed_part_batch)

    return DataLoader(all_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def load_tokenized_loader_with_length(filepath, batch_size=64):
    all_data = torch.load(filepath)

    def collate_fn(batch):
        original_batch, original_lengths, perturbed_batch, perturbed_lengths, perturbed_part_batch, perturbed_part_lengths = zip(
            *batch)

        original_batch = torch.stack(original_batch)
        perturbed_batch = torch.stack(perturbed_batch)
        perturbed_part_batch = torch.stack(perturbed_part_batch)
        original_lengths = torch.tensor(original_lengths, dtype=torch.int64)
        perturbed_lengths = torch.tensor(perturbed_lengths, dtype=torch.int64)
        perturbed_part_lengths = torch.tensor(perturbed_part_lengths, dtype=torch.int64)

        return (original_batch, original_lengths,
                perturbed_batch, perturbed_lengths,
                perturbed_part_batch, perturbed_part_lengths)

    return DataLoader(all_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
