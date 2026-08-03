"""Loads casting images into PyTorch DataLoaders using ImageFolder.

ImageFolder expects a directory shaped like:
    train/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img3.jpg
It automatically assigns labels based on subfolder name — alphabetical order
by default, so "def_front" = 0, "ok_front" = 1 (verify this matches your
config's class_names before trusting predictions later).
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool):
    """Different transforms for training vs. evaluation.

    Training gets random augmentation (flips, slight rotation) so the model
    sees varied versions of each image and generalizes better instead of
    memorizing exact pixels. Test/eval data is NOT augmented — we want to
    evaluate on realistic, unaltered images.
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # ImageNet mean/std — required when using ImageNet-pretrained weights,
        # since the pretrained model expects inputs normalized this way.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
        return transforms.Compose(augment + base)
    return transforms.Compose(base)


def build_dataloaders(train_dir: str, test_dir: str, image_size: int, batch_size: int):
    train_dataset = datasets.ImageFolder(
        train_dir, transform=build_transforms(image_size, train=True)
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=build_transforms(image_size, train=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes