from PIL import Image
from defect_detector.data.load import build_transforms


def test_transform_output_shape():
    """A transformed image should always come out as a fixed-size tensor,
    regardless of the original image's size — this is the whole point of
    the resize step."""
    transform = build_transforms(image_size=128, train=False)

    # Create a fake image with a DIFFERENT size than our target, on purpose
    fake_image = Image.new("RGB", (300, 200), color="white")
    tensor = transform(fake_image)

    assert tensor.shape == (3, 128, 128)  # channels, height, width


def test_train_transform_includes_augmentation():
    """Sanity check that train=True actually returns a different (longer)
    pipeline than train=False, confirming augmentation steps are added."""
    train_transform = build_transforms(image_size=128, train=True)
    eval_transform = build_transforms(image_size=128, train=False)

    assert len(train_transform.transforms) > len(eval_transform.transforms)