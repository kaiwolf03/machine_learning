import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from os import listdir
from PIL import Image
import random
import os

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    normalize
])

# Target: [isCat, isDog]
train_data_list = []
train_data = []
target_list = []
files = listdir('C:/Users/User/Desktop/cat_dog/train/')

for i in range(len(listdir('C:/Users/User/Desktop/cat_dog/train/'))):

    # images
    f = random.choice(files)
    files.remove(f)
    img = Image.open("C:/Users/User/Desktop/cat_dog/train/" + f)
    img_tensor = transform(img) # (3, 64, 64)
    train_data_list.append(img_tensor)

    # labels
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)

    # make batches (size 30)
    if len(train_data_list) >= 50:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.fc1 = nn.Linear(288, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #return F.sigmoid(x)
        return F.softmax(x, dim=1)


model = Netz()

if os.path.isfile('Cat_Dog.pt'):
    model = torch.load('Cat_Dog.pt')

optimizer = optim.Adam(model.parameters(), lr=0.0015)

def train(epoch):
    model.train()

    batch_id = 0
    for data, target in train_data:
        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)

        criterion = F.cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id * len(data), len(train_data), 100. * batch_id / len(train_data), loss.data))
        batch_id += 1


def test():
    model.eval()

    files = listdir('C:/Users/User/Desktop/cat_dog/test/')
    f = random.choice(files)
    img = Image.open('C:/Users/User/Desktop/cat_dog/test/' + f)
    img_eval_tensor = transform(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor)

    out = model(data)
    print(out.data.max(1, keepdim=True)[1])
    img.show()
    x = input('')


for epoch in range(1, 5):
    train(epoch)
    test()

torch.save(model, 'Cat_Dog.pt')