# train_cpu.py

import os
import json
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CSRNet
from dataset import listDataset

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, original_lr, steps, scales):
    lr = original_lr
    for i in range(len(steps)):
        if epoch >= steps[i]:
            lr = lr * scales[i]
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(train_list, model, criterion, optimizer, epoch, batch_size):
    model.train()
    losses = AverageMeter()
    train_loader = DataLoader(
        listDataset(train_list, shuffle=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), train=True, batch_size=batch_size),
        batch_size=batch_size)

    for i, (img, target) in enumerate(train_loader):
        output = model(img)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), img.size(0))

        if i % 10 == 0:
            print(f'Epoch [{epoch}] Step [{i}/{len(train_loader)}], Loss: {losses.val:.4f} (Avg: {losses.avg:.4f})')

    return losses.avg

def validate(val_list, model, criterion, batch_size):
    model.eval()
    mae = 0.0
    val_loader = DataLoader(
        listDataset(val_list, shuffle=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]), train=False, batch_size=batch_size),
        batch_size=batch_size)

    with torch.no_grad():
        for i, (img, target) in enumerate(val_loader):
            output = model(img)
            mae += abs(output.sum().item() - target.sum().item())
    mae /= len(val_loader)
    print(f'\nValidation MAE: {mae:.3f}')
    return mae

def run_training(train_json_path, val_json_path, epochs=10, batch_size=1, lr=1e-7):
    with open(train_json_path, 'r') as f:
        train_list = json.load(f)
    with open(val_json_path, 'r') as f:
        val_list = json.load(f)

    model = CSRNet(load_weights=False)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4)

    best_mae = float('inf')
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]

    for epoch in range(epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, steps, scales)
        print(f"\n--- Epoch {epoch+1}/{epochs} --- Learning Rate: {lr:.1e}")
        train_loss = train(train_list, model, criterion, optimizer, epoch, batch_size)
        val_mae = validate(val_list, model, criterion, batch_size)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f'best_model_epoch_{epoch+1}.pth')
            print(f">>> Best model saved (MAE: {best_mae:.3f})")

    print(f"\nTraining completed. Best MAE: {best_mae:.3f}")
