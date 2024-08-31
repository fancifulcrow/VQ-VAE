import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


def calculate_psnr(original:torch.Tensor, reconstruction:torch.Tensor) -> float:
    mse = F.mse_loss(reconstruction, original)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def calculate_ssim(original:torch.Tensor, reconstruction:torch.Tensor) -> float:
    return structural_similarity_index_measure(original, reconstruction, data_range=1.0).item()


def evaluate(model, data_loader, device) -> tuple[float, float, float, float]:
    model.eval()

    running_loss = 0.0
    psnr_values = []
    ssim_values = []
    used_codebook_indices = set()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs = data[0].to(device)

            recon, loss, _, encoding_indices = model(inputs)

            running_loss += loss.item()
            psnr_values.append(calculate_psnr(inputs, recon))
            ssim_values.append(calculate_ssim(inputs, recon))
            used_codebook_indices.update(encoding_indices.cpu().numpy().flatten())

    test_loss = running_loss / len(data_loader)
    psnr = np.mean(psnr_values)
    ssim = np.mean(ssim_values)
    codebook_utilization = len(used_codebook_indices) / model.num_embeddings

    return test_loss, psnr, ssim, codebook_utilization
