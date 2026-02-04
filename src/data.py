
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(
    data_dir,
    batch_size=64,
    val_split=0.2,
    num_workers=2
):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.STL10(
        root=data_dir,
        split="train",
        download=True,
        transform=transform
    )

    test_dataset = datasets.STL10(
        root=data_dir,
        split="test",
        download=True,
        transform=transform
    )

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
