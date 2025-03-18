#!/usr/bin/env python
# coding: utf-8

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image

# ### Step 1. Define Encoder and Decoder Networks
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        x_hat = x_hat.view(x.size(0), 3, 350, 350)  # Reshape back to image size
        return x_hat

class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var

# ### Step 2. Custom Dataset to Load Image Files from a Directory
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning dummy label

# ### Step 3. Load Dataset and Create DataLoader
if __name__ == '__main__':
    dataset_path = './dataset/bettery'  # Training image folder
    dataset_path_val = './dataset/ori_rotate'  # Validation image folder

    img_transform = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomImageDataset(img_dir=dataset_path, transform=img_transform)
    val_dataset = CustomImageDataset(img_dir=dataset_path_val, transform=img_transform)

    cuda = True
    DEVICE = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    # 注意：此 batch_size 設定較大，請根據你的 GPU VRAM 調整
    batch_size = 1000  
    x_dim = 367500  # 350*350*3 (Flattened image size)
    hidden_dim = 400
    latent_dim = 200
    lr = 1e-4
    kwargs = {'num_workers': 1, 'pin_memory': True}

    epochs = 100000
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    model = Model(Encoder=encoder, Decoder=decoder)

    # ### 多 GPU 支援：使用 DataParallel 將模型分散到所有可用 GPU 上
    if torch.cuda.device_count() >= 1:
        print("使用", torch.cuda.device_count(), "張 GPU 進行訓練")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    # ### Step 4. Define Loss Function and Optimizer
    from torch.optim import Adam

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=lr)

    # ### Step 5. Train Variational AutoEncoder (VAE) with a Single Overall Progress Bar
    print("Start training VAE...")
    model.train()

    total_steps = epochs * len(train_loader)
    with tqdm(total=total_steps, desc="Training VAE", unit="step") as pbar:
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(DEVICE)
                optimizer.zero_grad()

                # 在每個 batch 上也可以分別跑 validation（根據需求而定）
                for val_x, _ in val_loader:
                    val_x = val_x.to(DEVICE)
                    val_x_hat, val_mean, val_log_var = model(val_x)
                    loss = loss_function(val_x, val_x_hat, val_mean, val_log_var)
                    overall_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                pbar.update(1)
            
            if epoch % 5000 == 0:
                torch.save(model.state_dict(), f'autoencoder_model_epoch_{epoch}.pth')
                print("Model Saved!!")

    torch.save(model.state_dict(), 'autoencoder_model_final_100k.pth')
    print("Training Complete!")