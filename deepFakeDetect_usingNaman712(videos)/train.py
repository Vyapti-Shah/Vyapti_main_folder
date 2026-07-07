import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from service.dataset import DeepfakeDataset
from service.models.mkfanet import MkfaNet
from service.models.resnext_lstm import ResNeXtLSTMDetector

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Training on {device.type.upper()}")

    # 1. Load Dataset
    print(f"[*] Loading dataset from {args.dataset}...")
    train_dataset = DeepfakeDataset(root_dir=args.dataset, image_size=256, is_train=True)
    
    if len(train_dataset) == 0:
        print("[-] ERROR: No images found in dataset. Please add images to dataset/real/ and dataset/fake/.")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 2. Initialize Model
    print(f"[*] Initializing model architecture: {args.model}")
    if args.model == "mkfanet":
        model = MkfaNet(num_classes=2)
        # MkfaNet expects a 4D tensor (Batch, Channels, Height, Width)
        is_sequence_model = False
    elif args.model == "resnext-lstm":
        model = ResNeXtLSTMDetector(num_classes=2)
        # ResNeXtLSTM expects a 5D sequence tensor (Batch, Seq, Channels, Height, Width)
        is_sequence_model = True
    else:
        print(f"[-] ERROR: Unknown model {args.model}")
        sys.exit(1)

    model = model.to(device)

    # 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 4. Training Loop
    print(f"[*] Beginning training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # If using LSTM, we need to fake a sequence dimension of length 1 for images
            # (In reality, for robust video training, the Dataset should return sequences of frames)
            if is_sequence_model:
                images = images.unsqueeze(1) # shape becomes (Batch, 1, Channels, Height, Width)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"    Epoch [{epoch}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"[+] Epoch {epoch} Summary: Accuracy: {epoch_acc:.2f}%, Avg Loss: {epoch_loss:.4f}\n")

    # 5. Save Weights
    save_path = f"{args.model}_trained_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[+] Training Complete! Weights saved to {save_path}")
    print(f"    You can now load these weights in your detector.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom Deepfake models on local data.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory (must contain 'real' and 'fake' subfolders)")
    parser.add_argument("--model", type=str, choices=["mkfanet", "resnext-lstm"], default="mkfanet", help="Architecture to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train(args)
