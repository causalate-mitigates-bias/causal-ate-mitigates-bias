# models/simpleNN.py
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):  # Lengths included for consistency with template
        x = self.embedding(x).mean(dim=1)  # Average the embeddings along the sequence dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)
