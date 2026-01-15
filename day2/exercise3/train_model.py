"""
Script for training an Unet model for land segmentation.
The main Python libraries are Pytorch and Torchgeo.

Created on Wed 7 Jan 2026

@author: ihakulin
Ideas and codesnippets from: 
* https://www.kaggle.com/code/cordmaur/38-cloud-simple-unet

"""

import os, sys, time, datetime
from typing import Any, Dict, List

# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader

# TorchGeo
from torchgeo.datasets import RasterDataset, UnionDataset
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# Matplotlib
import matplotlib.pyplot as plt

# Mixed precision
from torch import amp


# The data contains both imagery and ground truth masks. We want to load both of these rasters and combine  them into a one dataset that can be fed to the neural network. 
# We will first create a TorchGeo RasterDataset of both rasters and then combine them with UnionDataset from TorchGeo. 
# The is_image attribute is used to control how the data stored in the dataset is handled. 
def create_union_dataset(images, labels):
    class Image(RasterDataset):
        filename_glob = images
        is_image = True
        
    class Mask(RasterDataset):
        filename_glob = labels
        is_image = False

    return UnionDataset(Image("."), Mask("."))


def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        masks  = torch.stack([item["mask"] for item in batch])                                                        
        return {"image": images, "mask": masks.long()}

# Create training and validation dataloaders
def Create_dataloaders(
    data_deep,
    labels_deep,
    data_validation,
    labels_validation,
    tile_size,
    batch_size,
    num_workers,
    collate_fn,
    sampler_length,
):
    # create datasets
    train_dataset = create_union_dataset(data_deep, labels_deep)
    val_dataset = create_union_dataset(data_validation, labels_validation)

    # Randomgeosampler
    train_sampler = RandomGeoSampler(
        train_dataset,
        size=tile_size,
        length=sampler_length,
    )
    # Gridgeosampler
    val_sampler = GridGeoSampler(
        val_dataset,
        size=tile_size,
        stride=tile_size // 2,
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader, val_loader


# initialize Unet model
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # decoder
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))
        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
                )
        return expand
    
# Helper function for calculating accuracy of the predictions.
def acc_metric(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()

# Define earlystopping
class EarlyStopping:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Train model
def train(model, train_loader, val_loader, loss_fn, optimizer, acc_fn, epochs, device, early_stopper):
    start = time.time()
    model.to(device)
    # Use mixed-precision to make training more efficient
    scaler = torch.amp.GradScaler('cuda')

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs - 1}")
        print("-" * 10)

        model.train()
        train_loss = 0
        train_acc = 0
        n_train = 0

        for step, batch in enumerate(train_loader, 1):
            # copy image and labels to gpu
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            
            # reset gradients 
            optimizer.zero_grad()
            # run forward pass using mixed-precision
            with amp.autocast('cuda'):
                # do a forward pass
                output = model(x)
                # calculate training loss
                loss = loss_fn(output, y)
            
            # backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            train_acc += acc_fn(output, y).item() * batch_size
            n_train += batch_size

        # calculate average training loss and accuracy
        train_loss = train_loss / n_train
        train_acc = train_acc / n_train

        val_loss = 0 
        val_acc = 0 
        n_val = 0
        # set model to evaluation mode
        model.eval()

        # disable gradients and use mixed-precision
        with torch.no_grad(), amp.autocast('cuda'):
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)

                output = model(x)
                # calculate validation loss
                loss = loss_fn(output, y)

                batch_size = x.size(0)
                val_loss += loss.item() * batch_size
                val_acc += acc_fn(output, y).item() * batch_size
                n_val += batch_size

        val_loss = val_loss / n_val
        val_acc = val_acc / n_val

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train | loss={train_loss:.4f} acc={train_acc:.4f}\n"
            f"Val   | loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if early_stopper.early_stop(val_loss): 
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")

    return history  


def main():
    # Set path to data and label files
    base_folder = os.path.join(os.sep, 'scratch', 'project_xxxxxxxx', 'students', os.environ.get('USER'), 'course')
    data_folder = os.path.join(base_folder,'data', 'raster')
    data_deep = os.path.join(data_folder, 'data_deep.tif')
    data_validation = os.path.join(data_folder, 'data_validation.tif')

    labels_deep = os.path.join(data_folder, 'labels_deep.tif')
    labels_validation = os.path.join(data_folder, 'labels_validation.tif')

    output_folder = os.path.join(base_folder, 'model_training')
    os.makedirs(output_folder, exist_ok=True)

    # Training settings:
    in_channels = 8 # Number of bands in data image 
    num_classes = 4 # Number of classes in the labels data
    learning_rate = 1e-3 
    patience = 5 # How many epochs model training is continued, if validation loss does not improve any more.
    batch_size = 8 
    tile_size = 512 
    sampler_length = 1600
    num_epochs = 50 
    num_workers = 7
    # Set computing device: GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = Create_dataloaders(
    data_deep,
    labels_deep,
    data_validation,
    labels_validation,
    tile_size,
    batch_size,
    num_workers,
    collate_fn,
    sampler_length
    )

    # Create model
    model = UNET(8,4)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=patience, min_delta=0)
    # Train model
    history = train(model, train_loader, val_loader, loss_fn, opt, acc_metric, num_epochs, device, early_stopper)

    # Plot training and validation loss
    plt.figure(figsize=(10,8))
    plt.plot(history["train_loss"], label='Train loss')
    plt.plot(history["val_loss"], label='Valid loss')
    plt.legend()
    plt.savefig(f"{output_folder}/loss_plot.png")

    # Save the trained model for inference
    torch.save(model.state_dict(), f"{output_folder}/best_model.pt")


if __name__ == '__main__':
    ### This part just runs the main method and times it
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round(((end - start)/60),0)) + " minutes") 