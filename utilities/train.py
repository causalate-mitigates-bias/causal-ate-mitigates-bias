from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
import numpy as np


def train(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets, _ in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):  # Ignore lengths
            inputs, targets = inputs.to(device).long(), targets.to(device).view(-1, 1)  # Ensure targets have shape (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")


def train_sklearn_model(model, train_loader, device):
    texts, labels = [], []
    for text_list, label_list in train_loader:
        texts.extend(text_list)
        labels.extend(label_list)

    # Convert labels to a numpy array and check unique classes
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    # Ensure there are more than one class in the labels
    if len(unique_classes) <= 1:
        raise ValueError(f"The number of classes has to be greater than one; got {len(unique_classes)} class(es)")

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(texts)
    y_train = labels

    model.fit(X_train, y_train)
    return model, vectorizer
