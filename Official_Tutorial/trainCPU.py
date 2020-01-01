# 1. Loading and normalizing CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='.\data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='.\data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2. Define a Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save our trained model
PATH = '.\cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 5. Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network thinks that the image is of the particular class.
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Whole Network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Detailed Class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

"""
Files already downloaded and verified
Files already downloaded and verified
 bird   cat  deer  frog
[1,  2000] loss: 2.207
[1,  4000] loss: 1.901
[1,  6000] loss: 1.670
[1,  8000] loss: 1.555
[1, 10000] loss: 1.508
[1, 12000] loss: 1.461
[2,  2000] loss: 1.403
[2,  4000] loss: 1.342
[2,  6000] loss: 1.329
[2,  8000] loss: 1.297
[2, 10000] loss: 1.310
[2, 12000] loss: 1.272
[3,  2000] loss: 1.187
[3,  4000] loss: 1.191
[3,  6000] loss: 1.193
[3,  8000] loss: 1.202
[3, 10000] loss: 1.173
[3, 12000] loss: 1.168
[4,  2000] loss: 1.098
[4,  4000] loss: 1.098
[4,  6000] loss: 1.090
[4,  8000] loss: 1.092
[4, 10000] loss: 1.073
[4, 12000] loss: 1.094
[5,  2000] loss: 0.988
[5,  4000] loss: 1.032
[5,  6000] loss: 1.020
[5,  8000] loss: 1.027
[5, 10000] loss: 1.027
[5, 12000] loss: 1.042
[6,  2000] loss: 0.939
[6,  4000] loss: 0.974
[6,  6000] loss: 0.953
[6,  8000] loss: 0.950
[6, 10000] loss: 0.979
[6, 12000] loss: 1.000
[7,  2000] loss: 0.878
[7,  4000] loss: 0.915
[7,  6000] loss: 0.928
[7,  8000] loss: 0.918
[7, 10000] loss: 0.954
[7, 12000] loss: 0.928
[8,  2000] loss: 0.856
[8,  4000] loss: 0.859
[8,  6000] loss: 0.896
[8,  8000] loss: 0.869
[8, 10000] loss: 0.905
[8, 12000] loss: 0.906
[9,  2000] loss: 0.794
[9,  4000] loss: 0.824
[9,  6000] loss: 0.837
[9,  8000] loss: 0.869
[9, 10000] loss: 0.867
[9, 12000] loss: 0.890
[10,  2000] loss: 0.769
[10,  4000] loss: 0.799
[10,  6000] loss: 0.812
[10,  8000] loss: 0.831
[10, 10000] loss: 0.842
[10, 12000] loss: 0.843
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat  ship truck plane
Accuracy of the network on the 10000 test images: 63 %
Accuracy of plane : 64 %
Accuracy of   car : 74 %
Accuracy of  bird : 53 %
Accuracy of   cat : 49 %
Accuracy of  deer : 57 %
Accuracy of   dog : 51 %
Accuracy of  frog : 66 %
Accuracy of horse : 70 %
Accuracy of  ship : 70 %
Accuracy of truck : 79 %

Process finished with exit code 0
"""
