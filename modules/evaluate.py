import torch
from tqdm import tqdm


def evaluate(model, data_loader, device) -> float:
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs = data[0].to(device)

            _, loss, _ = model(inputs)
            running_loss += loss.item()

    test_loss = running_loss / len(data_loader)

    return test_loss
