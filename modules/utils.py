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


def show_images(images, num_rows:int, num_cols:int) -> None:
    plt.figure(figsize=(8, 8))

    images = images.detach().cpu()

    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        image = images[i].permute(1, 2, 0)  # Transpose from (C, H, W) to (H, W, C)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
    