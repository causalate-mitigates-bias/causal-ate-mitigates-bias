import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

from tqdm import tqdm

# Define a simple training function for the ATE model
def train_ate_model(ate_model, train_loader, criterion, optimizer, device, num_epochs=10):
    ate_model.to(device)
    ate_model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ate_model(inputs.long()).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")

#
# # Define a simple training function for the ATE model
# def train_ate_model(ate_model, new_batch_inputs, new_batch_outputs, criterion, optimizer, device, num_epochs=10):
#     # Convert new_batch_inputs to a list of tensors
#     new_batch_inputs = [torch.tensor(t) for t in new_batch_inputs]
#
#     dataset = TensorDataset(torch.stack(new_batch_inputs).long(), torch.tensor(new_batch_outputs, dtype=torch.float32))
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     ate_model.to(device)
#     ate_model.train()
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#             inputs, targets = inputs.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#             outputs = ate_model(inputs.long()).squeeze()
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         print(f"Epoch {epoch + 1}/{num_epochs} completed, Average Loss: {epoch_loss / len(train_loader):.4f}")


def train_sklearn_ate_model(ate_model, new_batch_inputs, new_batch_outputs, num_epochs=10):
    # Convert inputs and outputs to numpy arrays for sklearn
    X_train = np.array(new_batch_inputs)
    y_train = np.array(new_batch_outputs)

    for epoch in range(num_epochs):
        ate_model.fit(X_train, y_train)
        predictions = ate_model.predict(X_train)
        loss = mean_squared_error(y_train, predictions)
        print(f"Epoch {epoch + 1}/{num_epochs} completed, MSE Loss: {loss:.4f}")

    return ate_model