from tqdm import tqdm

def train(model, optimizer, data_loader, num_epochs:int, device) -> list[float]:
    model.train()

    epoch_losses = []  # List to store loss values for each epoch

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for i, data in enumerate(progress_bar, 0):
            inputs = data[0].to(device)

            optimizer.zero_grad()

            _, loss, _ = model(inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        epoch_loss = running_loss / len(data_loader)
        epoch_losses.append(epoch_loss)

    return epoch_losses