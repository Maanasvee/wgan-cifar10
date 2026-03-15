import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time

LATENT_DIM   = 100
IMG_CHANNELS = 3
FEATURES_G   = 64
FEATURES_D   = 64
BATCH_SIZE   = 128
NUM_EPOCHS   = 50
LR           = 0.00005
CLIP_VALUE   = 0.01
N_CRITIC     = 2
IMG_SIZE     = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(LATENT_DIM,    FEATURES_G*16, 4, 1, 0),
            self._block(FEATURES_G*16, FEATURES_G*8,  4, 2, 1),
            self._block(FEATURES_G*8,  FEATURES_G*4,  4, 2, 1),
            self._block(FEATURES_G*4,  FEATURES_G*2,  4, 2, 1),
            self._block(FEATURES_G*2,  FEATURES_G,    4, 2, 1),
            nn.ConvTranspose2d(FEATURES_G, IMG_CHANNELS, 4, 2, 1),
            nn.Tanh()
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, FEATURES_D, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._block(FEATURES_D,   FEATURES_D*2, 4, 2, 1),
            self._block(FEATURES_D*2, FEATURES_D*4, 4, 2, 1),
            self._block(FEATURES_D*4, FEATURES_D*8, 4, 2, 1),
            nn.Conv2d(FEATURES_D*8, 1, 4, 1, 0),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def get_loader():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=2, drop_last=True)

def train():
    os.makedirs("samples", exist_ok=True)
    os.makedirs("logs",    exist_ok=True)

    gen    = Generator().to(DEVICE)
    critic = Critic().to(DEVICE)
    init_weights(gen)
    init_weights(critic)

    opt_gen    = optim.RMSprop(gen.parameters(),    lr=LR)
    opt_critic = optim.RMSprop(critic.parameters(), lr=LR)

    loader      = get_loader()
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1).to(DEVICE)
    history     = {"critic_loss": [], "gen_loss": []}

    print("="*50)
    print("WGAN on CIFAR-10  |  Device:", DEVICE)
    print("="*50)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        c_losses, g_losses = [], []

        for real, _ in loader:
            real = real.to(DEVICE)
            B    = real.size(0)

            for _ in range(N_CRITIC):
                noise  = torch.randn(B, LATENT_DIM, 1, 1).to(DEVICE)
                fake   = gen(noise).detach()
                loss_c = -(torch.mean(critic(real)) - torch.mean(critic(fake)))
                opt_critic.zero_grad()
                loss_c.backward()
                opt_critic.step()
                for p in critic.parameters():
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
            c_losses.append(loss_c.item())

            noise  = torch.randn(B, LATENT_DIM, 1, 1).to(DEVICE)
            loss_g = -torch.mean(critic(gen(noise)))
            opt_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()
            g_losses.append(loss_g.item())

        avg_c = float(np.mean(c_losses))
        avg_g = float(np.mean(g_losses))
        history["critic_loss"].append(avg_c)
        history["gen_loss"].append(avg_g)

        print("Epoch [" + str(epoch).rjust(3) + "/" + str(NUM_EPOCHS) + "]  "
              "Critic: " + str(round(avg_c,4)) + "  "
              "Gen: "    + str(round(avg_g,4)) + "  "
              "Time: "   + str(round(time.time()-t0,1)) + "s")

        with torch.no_grad():
            fake = gen(fixed_noise)
        torchvision.utils.save_image(
            fake,
            "samples/epoch_" + str(epoch).zfill(3) + ".png",
            normalize=True, nrow=8
        )

    with open("logs/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Done!")

train()