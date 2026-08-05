"""End-to-end training entrypoint.

Run with:  uv run python -m defect_detector.models.train
"""
import torch
import torch.nn as nn
from pathlib import Path

from defect_detector.data.load import build_dataloaders
from defect_detector.models.model import build_model
from defect_detector.models.evaluate import compute_metrics
from defect_detector.utils.config import load_config, resolve
from defect_detector.utils.logging import get_logger

logger = get_logger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # tell the model "we're training" (affects some layers' behavior)
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear leftover gradients from last batch
        outputs = model(images)        # forward pass: model makes predictions
        loss = criterion(outputs, labels)  # how wrong were those predictions?
        loss.backward()                # backprop: figure out how to adjust
        optimizer.step()               # actually apply the adjustment

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()  # tell the model "we're evaluating" — no learning happening
    all_preds, all_labels = [], []
    with torch.no_grad():  # don't bother computing gradients, we're not training
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return compute_metrics(all_labels, all_preds)


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, test_loader, classes = build_dataloaders(
        resolve(cfg["data"]["train_dir"]).as_posix(),
        resolve(cfg["data"]["test_dir"]).as_posix(),
        cfg["data"]["image_size"],
        cfg["data"]["batch_size"],
    )
    logger.info(f"Classes: {classes}")

    model = build_model(cfg["model"]["num_classes"], cfg["model"]["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=cfg["train"]["learning_rate"])

    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, test_loader, device)
        logger.info(f"Epoch {epoch+1}/{cfg['train']['epochs']} - loss: {train_loss:.4f} - {metrics}")

    save_path = resolve(cfg["model"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved model to {save_path}")


if __name__ == "__main__":
    main()