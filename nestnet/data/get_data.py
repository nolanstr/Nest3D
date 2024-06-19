import torch
import numpy as np
import glob


def getData(path):
    imageDirs = glob.glob(path + "/*")
    data_ = {}

    for imageDir in imageDirs:
        tag = imageDir.split("/")[-1]
        data_[tag] = {}
        if "XY" in tag:
            data_[tag]["dim"] = 2
        elif "XZ" in tag:
            data_[tag]["dim"] = 1
        elif "YZ" in tag:
            data_[tag]["dim"] = 0
        else:
            raise ValueError(
                    "Expected to find XY, XZ, or YZ in directory name, {tag}.")
        filenames = glob.glob(imageDir + "/*.npy")
        images = []
        for filename in filenames:
            images.append(np.load(filename))
        shapes = [image.shape for image in images]
        for shape in shapes:
            data_[tag][shape] = []
        for image in images:
            image = image + 1
            image[np.isnan(image)] = 0
            data_[tag][image.shape].append(torch.tensor(image,
                                                        dtype=torch.float32))
    
    
    data = {}

    for key1 in data_.keys():
        data[key1] = {}
        for key2 in data_[key1].keys():
            if key2 != "dim":
                images = data_[key1][key2]
                if len(images)>1:
                    data[key1][key2] = torch.stack(images)
                else:
                    data[key1][key2] = images[0][None, :, :]
            else:
                data[key1][key2] = data_[key1][key2]
    return data
            



