import base64
import io
import os

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from PIL import ImageEnhance, Image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 14 classes, 10 digits, 4 operators
        self.out = nn.Linear(128 * 64, 14)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 128 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization


def train(num_epochs, cnn, loaders, loss_func, optimizer):
    cnn.train()
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)[0]

            loss = loss_func(output, b_y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass
        pass
    pass


def test(cnn, loaders):
    cnn.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loaders['test']):
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            if (i + 1) % 100 == 0:
                print('Step [{}/{}]'
                      .format(i + 1, len(loaders['test'])))
            pass
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
    pass


def classify(image_bytes):
    options = ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/", "*"]

    image_bytes = base64.b64decode(image_bytes)

    image = Image.open(io.BytesIO(image_bytes))

    model = CNN()

    model.load_state_dict(torch.load(os.path.join("model/model.pth")))
    model.eval()

    img = image.resize((45, 45))
    img = torchvision.transforms.Grayscale()(img)

    img = ImageEnhance.Contrast(img).enhance(2)

    input = torchvision.transforms.ToTensor()(img)

    input = input.unsqueeze(0)

    output, lay = model(input)

    pred_y = torch.max(output, 1)[1].data.squeeze()
    pred = pred_y.item()

    return options[pred]
