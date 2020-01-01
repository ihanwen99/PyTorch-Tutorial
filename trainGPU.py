# 1. Loading and normalizing CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
net.to(device)

# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. Train the network
# !!! GPU change here !!! times
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # !!! GPU change here !!! to(device)
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i == 11999:  # print every 2000 mini-batches
            print('%d loss: %.3f' % (epoch + 1, running_loss / 12000))
            running_loss = 0.0

print('Finished Training')

# Save our trained model
# !!! GPU change here !!! path
PATH = '.\cifar_net_gpu.pth'
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
cuda:0
Files already downloaded and verified
Files already downloaded and verified
1 loss: 1.752
2 loss: 1.346
3 loss: 1.192
4 loss: 1.096
5 loss: 1.026
6 loss: 0.977
7 loss: 0.929
8 loss: 0.889
9 loss: 0.853
10 loss: 0.823
11 loss: 0.797
12 loss: 0.775
13 loss: 0.753
14 loss: 0.738
15 loss: 0.711
16 loss: 0.704
17 loss: 0.690
18 loss: 0.679
19 loss: 0.659
20 loss: 0.660
21 loss: 0.653
22 loss: 0.644
23 loss: 0.632
24 loss: 0.632
25 loss: 0.631
26 loss: 0.624
27 loss: 0.615
28 loss: 0.620
29 loss: 0.605
30 loss: 0.603
31 loss: 0.606
32 loss: 0.612
33 loss: 0.608
34 loss: 0.604
35 loss: 0.598
36 loss: 0.599
37 loss: 0.599
38 loss: 0.602
39 loss: 0.593
40 loss: 0.597
41 loss: 0.595
42 loss: 0.603
43 loss: 0.594
44 loss: 0.592
45 loss: 0.598
46 loss: 0.587
47 loss: 0.593
48 loss: 0.612
49 loss: 0.612
50 loss: 0.608
51 loss: 0.604
52 loss: 0.602
53 loss: 0.592
54 loss: 0.612
55 loss: 0.620
56 loss: 0.619
57 loss: 0.602
58 loss: 0.611
59 loss: 0.626
60 loss: 0.616
61 loss: 0.610
62 loss: 0.609
63 loss: 0.609
64 loss: 0.610
65 loss: 0.612
66 loss: 0.615
67 loss: 0.621
68 loss: 0.633
69 loss: 0.624
70 loss: 0.630
71 loss: 0.651
72 loss: 0.627
73 loss: 0.638
74 loss: 0.631
75 loss: 0.634
76 loss: 0.622
77 loss: 0.626
78 loss: 0.649
79 loss: 0.631
80 loss: 0.632
81 loss: 0.644
82 loss: 0.619
83 loss: 0.655
84 loss: 0.658
85 loss: 0.630
86 loss: 0.641
87 loss: 0.668
88 loss: 0.659
89 loss: 0.681
90 loss: 0.675
91 loss: 0.656
92 loss: 0.661
93 loss: 0.654
94 loss: 0.639
95 loss: 0.660
96 loss: 0.687
97 loss: 0.672
98 loss: 0.676
99 loss: 0.680
100 loss: 0.677
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat  ship truck plane
Accuracy of the network on the 10000 test images: 55 %
Accuracy of plane : 63 %
Accuracy of   car : 64 %
Accuracy of  bird : 37 %
Accuracy of   cat : 43 %
Accuracy of  deer : 52 %
Accuracy of   dog : 47 %
Accuracy of  frog : 57 %
Accuracy of horse : 61 %
Accuracy of  ship : 60 %
Accuracy of truck : 67 %

Process finished with exit code 0
"""