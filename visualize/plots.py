import matplotlib.pyplot as plt
import os

# Plot training and testing loss/accuracy curves and save them
def plot_curves(train_losses, test_losses, train_accs, test_accs, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Plot and save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, test_losses, label="Test Loss", color="red")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()  # avoid displaying in some environments

    # Plot and save accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy", color="green")
    plt.plot(epochs, test_accs, label="Test Accuracy", color="orange")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    acc_path = os.path.join(save_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Plots saved to '{save_dir}/'")
