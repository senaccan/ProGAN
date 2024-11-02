from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

baseFolder = "//content//drive//MyDrive//flowers"

images = []
labels = []

for flowerType in os.listdir(baseFolder):
    folderPath = os.path.join(baseFolder, flowerType)
    if os.path.isdir(folderPath):
        for imgFile in glob.glob(os.path.join(folderPath, "*.jpg")):
            img = Image.open(imgFile).resize((64, 64))
            imgArray = np.array(img) / 255.0
            images.append(imgArray)
            labels.append(flowerType)

images = np.array(images)
labels = np.array(labels)

class FlowerDataset(Dataset):
    def __init__(self, images, labels, img_size=64, transform=None):
        self.images = images
        self.labels = labels
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray((image * 255).astype(np.uint8)).resize((self.img_size, self.img_size))
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
])

class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, img_channels * img_size * img_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.gen(x)
        return x.view(-1, 3, self.img_size, self.img_size)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, img_size=64):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.img_size * self.img_size * 3)
        return self.disc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 128

def show_generated_images(images, epoch):
    images = (images + 1) / 2
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Generated Images at Epoch {epoch}")
    for i in range(min(9, images.size(0))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].cpu().detach().numpy().transpose(1, 2, 0))
        plt.axis('off')
    plt.show()

def train_gan(img_size, epochs=300, batch_size=32, z_dim=128):

    flowerDataset = FlowerDataset(images, labels, img_size=img_size, transform=transform)
    dataLoader = DataLoader(flowerDataset, batch_size=batch_size, shuffle=True)

    gen = Generator(z_dim=z_dim, img_size=img_size).to(device)
    disc = Discriminator(img_size=img_size).to(device)

    genOpt = torch.optim.Adam(gen.parameters(), lr=0.00005, betas=(0.5, 0.999))
    discOpt = torch.optim.Adam(disc.parameters(), lr=0.00005, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for real, _ in dataLoader:
            real = real.view(-1, img_size * img_size * 3).float().to(device)
            batch_size = real.size(0)

            realLabels = torch.ones(batch_size, 1).to(device)
            fakeLabels = torch.zeros(batch_size, 1).to(device)

            noise = torch.randn(batch_size, z_dim).to(device)
            fakeImages = gen(noise)
            discReal = disc(real)
            discFake = disc(fakeImages.detach())

            lossReal = criterion(discReal, realLabels)
            lossFake = criterion(discFake, fakeLabels)
            discLoss = (lossReal + lossFake) / 2
            discOpt.zero_grad()
            discLoss.backward()
            discOpt.step()

            output = disc(fakeImages)
            genLoss = criterion(output, realLabels)
            genOpt.zero_grad()
            genLoss.backward()
            genOpt.step()

        if (epoch + 1) % 10 == 0:
            print(f"Image Size {img_size} | Epoch [{epoch+1}/{epochs}] | Generator Loss: {genLoss:.4f} | Discriminator Loss: {discLoss:.4f}")
            show_generated_images(fakeImages, epoch + 1)

resolutions = [4, 8, 16, 32, 64, 256]
for img_size in resolutions:
    print(f"\nStarting training with resolution {img_size}x{img_size}")
    train_gan(img_size=img_size)

