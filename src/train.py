
import os
import torch
import torch.nn as nn
import mlflow
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data import get_dataloaders
from src.model import SimpleCNN

def train(
    data_dir,
    experiment_name="stl10_cnn",
    epochs=20,
    batch_size=64,
    lr=1e-3,
    patience=5,
    device=None,
    model_dir="models"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_experiment(experiment_name)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_val_loss = float("inf")
    epochs_without_improve = 0

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau"
        })

        for epoch in range(1, epochs + 1):
            # ---- Training ----
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_loss /= len(train_loader)
            train_acc = correct / total

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = correct / total

            scheduler.step(val_loss)

            # ---- MLflow logging ----
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # ---- Early stopping + checkpoint ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improve = 0

                checkpoint_path = os.path.join(model_dir, "best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)

            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print("Early stopping triggered.")
                    break
