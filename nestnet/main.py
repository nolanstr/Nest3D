import numpy as np
import sys
sys.path.append("./data")

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


from data.get_data import getData
from custom_data_loader import CustomDataLoader

from nn_models.generator import EllipsoidNet
from nn_models.slicer import Slicer
from nn_models.discriminator import Discriminator

dim_to_tag = {1:"xz",
              2:"xy"}
tag_to_dim = {"xz":1,
              "xy":2}
def getDataLoaders(pathToData, batch_size=4, shuffle=True):
    data = getData(pathToData)
    datasets = {
        "xy_20": [],
        "xy_50": [], 
        "xy_100": [], 
        "xy_150": [],
        "xy_200": [], 
        "xz_20": [],
        "xz_50": [], 
        "xz_100": [], 
        "xz_150": [], 
        "xz_200": [],
        }
    for key1 in data.keys():
        tag = dim_to_tag[data[key1]["dim"]]
        for key2 in data[key1].keys():
            if key2 != "dim":
                shape_tag = key2[0]
                datasets[f"{tag}_{shape_tag}"].append(data[key1][key2])
    tags = [key for key in datasets.keys()]
    d_ = [torch.vstack(datasets[key]) for key in tags]
    dataLoader = CustomDataLoader(d_, tags, batch_size=batch_size, shuffle=shuffle)
    return dataLoader

if __name__ == "__main__":
    
    batch_size = 4
    learning_rate = 0.0002
    shuffle = True

    pathToData = "./data/images"
    #dataLoaders = getDataLoaders(pathToData, batch_size=batch_size,
    #                             shuffle=shuffle)
    dataloader = getDataLoaders(pathToData, batch_size=batch_size,
                                 shuffle=shuffle)
    input_size = 200
    in_channels = 1

    # Generator
    generator = EllipsoidNet(input_size, in_channels, maxPasses=20)
    
    # Slicer
    slicer = Slicer(sizes=[20,50,100,150,200])

    # Discriminators
    discriminators = {
        "xy_20": Discriminator(n=20),
        "xy_50": Discriminator(n=50),
        "xy_100": Discriminator(n=100),
        "xy_150": Discriminator(n=150),
        "xy_200": Discriminator(n=200),
        "xz_20": Discriminator(n=20),
        "xz_50": Discriminator(n=50),
        "xz_100": Discriminator(n=100),
        "xz_150": Discriminator(n=150),
        "xz_200": Discriminator(n=200),
    }
    epochs = 100
    sample_interval = 100 // 10

    # Loss function
    adversarial_loss = nn.BCELoss()
    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    optimizers_D = [
        optim.Adam(discriminators[key].parameters(), lr=learning_rate, betas=(0.5, 0.999))
        for key in discriminators.keys()
    ]
    optimizers_D = dict(zip(discriminators.keys(), optimizers_D))

    
    for epoch in range(epochs):
        print(f"EPOCH: {epoch}")
        for i, imgs in enumerate(dataloader):
            print(f"RUNNING image set {i}...")
            #fake = torch.zeros(size, 1, requires_grad=False)
            # Configure input
            sizes = np.array([img.shape[1] for img in imgs])
            # Trainer Generator
            optimizer_G.zero_grad()
            z = torch.zeros((1, 1, input_size, input_size, input_size))
            gen_img = generator(z)
            slices = slicer(gen_img)
            g_loss = 0.0

            for i, key in enumerate(discriminators.keys()):

                tag = key.split("_")[0]
                dim = tag_to_dim[tag]
                size = int(key.split("_")[1])
                valid = torch.ones(1, requires_grad=False)
                gen_imgs = slices[dim][size]

                for gen_img in gen_imgs:
                    g_loss += adversarial_loss(discriminators[key](gen_img), valid)
            g_loss /= len(discriminators)
            g_loss.backward()
            optimizer_G.step()

            # Train each Discriminator separately
            for key in discriminators.keys():

                optimizer_D = optimizers_D[key]
                d = discriminators[key]

                tag = key.split("_")[0]
                dim = tag_to_dim[tag]
                size = int(key.split("_")[1])
                optimizer_D.zero_grad()
                gen_imgs = slices[dim][size]

                key_idx = dataloader.idx_from_key[key]
                real_imgs = imgs[key_idx]
                valid = torch.ones(1, requires_grad=False)
                fake = torch.ones(1, requires_grad=False)
                
                # Measure discriminator's ability to classify real from generated samples
                real_loss = 0.0
                fake_loss = 0.0
                for real_img in real_imgs:
                    real_loss += adversarial_loss(d(real_img), valid)
                for gen_img in gen_imgs:
                    fake_loss += adversarial_loss(d(gen_img), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

            # Print training stats
        if epoch % sample_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
    
    path_fn = lambda model_tag: f"./models/{model_tag}"
    torch.save(generator.state_dict(), path_fn("generator"))
    
    discriminator_path = "./models"
    for key in discriminators.keys():
        d = discriminators[key]
        torch.save(d.state_dict(), path_fn(key))


