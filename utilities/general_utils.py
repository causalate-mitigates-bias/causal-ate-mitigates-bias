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


# Function to save DataLoader
def save_dataloader(dataloader, file_path):
    torch.save(dataloader.dataset, file_path)

# Function to load DataLoader
def load_dataloader(file_path, batch_size=64):
    dataset = torch.load(file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader