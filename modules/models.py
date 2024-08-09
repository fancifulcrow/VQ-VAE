import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim:int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
    

class Encoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.res_layers = nn.Sequential(* [ResidualBlock(out_channels) for _ in range(3)])

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_layers(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.res_layers = nn.Sequential(* [ResidualBlock(in_channels) for _ in range(3)])
        self.conv2 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)

    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res_layers(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, commitment_cost:float) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)


    def forward(self, inputs:torch.Tensor) -> tuple[torch.Tensor, float]:
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        original_shape = inputs.shape

        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = torch.cdist(flat_inputs, self.codebook.weight)
        encoding_indices = torch.argmin(distances, dim=1)

        quantized = torch.index_select(self.codebook.weight, 0, encoding_indices).view(original_shape)

        encoding_loss = F.mse_loss(quantized.detach(), inputs)
        quantization_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = quantization_loss + self.commitment_cost * encoding_loss

        quantized = inputs + (quantized - inputs).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, commitment_loss
    


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, num_embeddings=512, embedding_dim=64, commitment_cost=0.25) -> None:
        super().__init__()

        self.encoder = Encoder(in_channels, latent_dim)

        self.conv_vq = nn.Conv2d(latent_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = Decoder(embedding_dim, in_channels)


    def forward(self, inputs:torch.Tensor) -> tuple[torch.Tensor, float, float]:
        encoding = self.encoder(inputs)
        quantized, commitment_loss = self.vector_quantizer(self.conv_vq(encoding))
        reconstructions = self.decoder(quantized)

        reconstruction_loss = F.mse_loss(reconstructions, inputs)        
        total_loss = reconstruction_loss + commitment_loss

        return reconstructions, total_loss, reconstruction_loss
