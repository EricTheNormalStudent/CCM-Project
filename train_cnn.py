from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ROWS = 6
COLUMNS = 7


class SelfPlayDataset(Dataset):
    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as data:
            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.int64)

        self.states = torch.from_numpy(states).unsqueeze(1)
        self.actions = torch.from_numpy(actions)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


class Connect4PolicyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * ROWS * COLUMNS, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, COLUMNS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_examples = 0

    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(states)
        loss = criterion(logits, actions)

        if is_training:
            loss.backward()
            optimizer.step()

        batch_size = states.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy_from_logits(logits, actions) * batch_size

    return total_loss / total_examples, total_accuracy / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on self-play Connect4 data.")
    parser.add_argument("--data", default="self_play_data.npz", help="Path to the self-play .npz file.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay for Adam.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split in [0, 1).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", default="connect4_policy_cnn.pt", help="Path to save the trained model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = SelfPlayDataset(args.data)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split leaves no training data.")

    generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4PolicyCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Loaded {len(dataset)} samples from {args.data}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Device: {device}")
    print(f"Weight decay: {args.weight_decay}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            print(
                f"epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            print(f"epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    output_path = Path(args.output)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "rows": ROWS,
            "columns": COLUMNS,
            "input_channels": 1,
        },
        output_path,
    )
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
