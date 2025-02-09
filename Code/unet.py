import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class CrackDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("L")

        mask = np.array(mask) // 255
        image = np.array(image)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sigmoid(self.out_conv(d1))


def train_model(model, train_loader, valid_loader, device, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        validate_model(model, valid_loader, device)

def validate_model(model, valid_loader, device):
    model.eval()
    total_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss = loss_fn(output, masks)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    f1_scores, acc_scores, prec_scores, rec_scores = [], [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()

            f1_scores.append(f1_score(masks, preds))
            acc_scores.append(accuracy_score(masks, preds))
            prec_scores.append(precision_score(masks, preds))
            rec_scores.append(recall_score(masks, preds))

    print(f"Avg F1: {np.mean(f1_scores):.4f}, Acc: {np.mean(acc_scores):.4f}, Precision: {np.mean(prec_scores):.4f}, Recall: {np.mean(rec_scores):.4f}")

if __name__ == "__main__":
    data_path = "/CRACK500_unzip"
    train_folder = os.path.join(data_path, "traincrop/traincrop")
    validation_folder = os.path.join(data_path, "valcrop/valcrop")
    test_folder = os.path.join(data_path, "testcrop/testcrop")

   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5,), (0.5,))
    ])

 
    train_ds = CrackDataset(train_folder, transform)
    valid_ds = CrackDataset(validation_folder, transform)
    test_ds = CrackDataset(test_folder, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

   
    train_model(model, train_loader, valid_loader, device, epochs=10)
    evaluate_model(model, test_loader, device)

    torch.save(model.state_dict(), "unet_crack_detection.pth")
