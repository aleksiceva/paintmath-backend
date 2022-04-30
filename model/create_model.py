# %%

# dataset from https://www.kaggle.com/xainano/handwrittenmathsymbols

import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.utils.data import TensorDataset
from torchvision import datasets

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import CNN, train, test

data = datasets.ImageFolder('./data/extracted_images',
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.Grayscale(num_output_channels=1),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Resize((45, 45))
                            ]),
                            )

train_data, test_data = train_test_split(data, test_size=0.2)

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}

cnn = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)

num_epochs = 3

train(num_epochs, cnn, loaders, loss_func, optimizer)
test(cnn, loaders)

path = "/Users/Korisnik/Documents/bzvzprojekt/NotPhotoMath/cnn.pth"
torch.save(cnn.state_dict(), path)

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
