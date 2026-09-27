from src.data.cub200 import get_dataloaders

train_loader, val_loader = get_dataloaders(
    data_dir="data/raw/CUB_200_2011",
    batch_size=32,
    num_workers=0,
    augment=False,
)

images, labels = next(iter(train_loader))

print("DataLoader OK")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Training images:", len(train_loader.dataset))
print("Validation/test images:", len(val_loader.dataset))
print("First labels:", labels[:10].tolist())
