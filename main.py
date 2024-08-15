from modules.models import VQVAE
from modules.data import load_data
from modules.train import train
from modules.evaluate import evaluate
from modules.utils import plot_loss, count_parameters, show_images

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import yaml
import time
import os

with open('config/default.yaml', 'r') as file:
    config = yaml.safe_load(file)


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_set, test_set = load_data(root=config['dataset']['path'])

    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=True)

    model = VQVAE(in_channels=config['model']['in_channels'],
                  hidden_channels=config['model']['hidden_channels'],
                  num_embeddings=config['model']['num_embeddings'],
                  embedding_dim=config['model']['embedding_dim'],
                  commitment_cost=config['model']['commitment_cost']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    print(f"No of Parameters: {count_parameters(model)}")

    training_losses = train(model, optimizer, train_loader, config['training']['num_epochs'], device)

    os.makedirs(config['output']['model_dir'], exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, f"{config['output']['model_dir']}/model_{int(time.time())}.pth")

    plot_loss(training_losses)

    test_loss = evaluate(model, test_loader, device)
    print(f'Test Loss: {test_loss}')

    dataiter = iter(test_loader)
    original, _ = next(dataiter)

    reconstructions, _, _ = model(original.to(device))

    show_images(original)
    show_images(reconstructions)


if __name__ == "__main__":
    main()
