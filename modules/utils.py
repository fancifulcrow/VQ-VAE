import matplotlib.pyplot as plt


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss(epoch_losses: list[float]) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(epoch_losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


def show_images(data_loader) -> None:
    plt.figure(figsize=(12, 12))

    dataiter = iter(data_loader)
    images, _ = next(dataiter)

    images = images.cpu()

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        image = images[i].permute(1, 2, 0)  # Transpose from (C, H, W) to (H, W, C)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
    