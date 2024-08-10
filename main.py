from modules.models import VQVAE
from modules.data import load_data
from modules.train import train
from modules.evaluate import evaluate
from modules.utils import plot_loss, count_parameters, show_images

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import logging
import yaml
import time
import os

with open('config/default.yaml', 'r') as file:
        config = yaml.safe_load(file)

logging.basicConfig(
    filename=config['logging']['log_file'],
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG
)


def log_devices() -> None:
    if torch.cuda.is_available():
        logging.info('CUDA is available')
        device_names = []
        for i in range(torch.cuda.device_count()):
            device_names.append(f"GPU {i} : {torch.cuda.get_device_name(i)}")
            logging.info(' | '.join(device_names))
    else:
            logging.info('CUDA is not available')


def main() -> None:
    session_id = int(time.time())
    logging.info(f'Session {session_id} Start')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_devices()
    
    train_set, test_set = load_data(root=config['dataset']['path'])

    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config['training']['batch_size'], shuffle=True)

    model = VQVAE(in_channels=config['model']['in_channels'],
                  latent_dim=config['model']['latent_dim'],
                  num_embeddings=config['model']['num_embeddings'],
                  embedding_dim=config['model']['embedding_dim'],
                  commitment_cost=config['model']['commitment_cost']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    logging.info(f"No of Parameters: {count_parameters(model)}")

    model_dir = os.path.join(config['output']['model_dir'], str(session_id))
    os.makedirs(model_dir, exist_ok=True)

    training_losses = train(model, optimizer, train_loader, config['training']['num_epochs'], device, model_dir)

    plot_loss(training_losses)

    test_loss = evaluate(model, test_loader, device)
    print(f'Test Loss: {test_loss}')

    dataiter = iter(test_loader)
    original, _ = next(dataiter)

    reconstructions, _, _ = model(original)

    show_images(original)
    show_images(reconstructions)

    logging.info(f'Session {session_id} End')


if __name__ == "__main__":
    main()
