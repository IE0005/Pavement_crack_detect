import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt


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

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(SegNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.ModuleList([
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 512)
        ])

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.decoder = nn.ModuleList([
            conv_block(512, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            conv_block(64, out_channels)
        ])

        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x):
        indices, sizes = [], []
        for enc in self.encoder:
            x = enc(x)
            sizes.append(x.size())
            x, ind = self.pool(x)
            indices.append(ind)

        for dec in self.decoder:
            x = self.unpool(x, indices.pop(), output_size=sizes.pop())
            x = dec(x)

        return F.softmax(x, dim=1)


def train_model(model, train_loader, valid_loader, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.long().to(device).squeeze(1)
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
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.long().to(device).squeeze(1)
            output = model(images)
            loss = loss_fn(output, masks)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss/len(valid_loader):.4f}")

def evaluate_model(model, test_loader, device):
    model.eval()
    f1_scores, acc_scores, prec_scores, rec_scores = [], [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.long().to(device).squeeze(1)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()

            f1_scores.append(f1_score(masks, preds))
            acc_scores.append(accuracy_score(masks, preds))
            prec_scores.append(precision_score(masks, preds))
            rec_scores.append(recall_score(masks, preds))

    print(f"Avg F1: {np.mean(f1_scores):.4f}, Acc: {np.mean(acc_scores):.4f}, Precision: {np.mean(prec_scores):.4f}, Recall: {np.mean(rec_scores):.4f}")


def save_prediction(tensor, path):
    img = torch.argmax(tensor, dim=1).squeeze().cpu().numpy() * 255
    Image.fromarray(img.astype(np.uint8)).save(path)

def save_input(image, path):
    img = image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    Image.fromarray(img.astype(np.uint8)).save(path)

def save_mask(mask, path):
    img = mask.squeeze().cpu().numpy() * 255
    Image.fromarray(img.astype(np.uint8)).save(path)

if __name__ == "__main__":
    # Paths
    data_path = "/CRACK500_unzip"
    train_folder = os.path.join(data_path, "traincrop/traincrop")
    validation_folder = os.path.join(data_path, "valcrop/valcrop")
    test_folder = os.path.join(data_path, "testcrop/testcrop")

   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360, 640)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    
    train_ds = CrackDataset(train_folder, transform)
    valid_ds = CrackDataset(validation_folder, transform)
    test_ds = CrackDataset(test_folder, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet().to(device)

  
    train_model(model, train_loader, valid_loader, device, epochs=10)
    evaluate_model(model, test_loader, device)

    
    torch.save(model.state_dict(), "segnet_crack_detection.pth")
