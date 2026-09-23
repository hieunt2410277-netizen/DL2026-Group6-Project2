import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # TODO:
    # 1. Load configuration
    # 2. Load train/validation DataLoader
    # 3. Get num_classes from dataset
    # 4. Initialize BaselineCNN
    # 5. Train for multiple epochs
    # 6. Validate model
    # 7. Save metrics and results

    print("Baseline training pipeline is waiting for the common DataLoader.")


if __name__ == "__main__":
    main()