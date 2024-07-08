from tqdm import tqdm

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
