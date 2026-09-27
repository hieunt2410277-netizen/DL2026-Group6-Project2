from src.models.resnet18 import build_resnet18, count_parameters

for strategy in ["frozen", "finetune"]:
    model = build_resnet18(num_classes=200, strategy=strategy)
    total, trainable = count_parameters(model)

    print(f"\nStrategy: {strategy}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
