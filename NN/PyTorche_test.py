import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


netz = MeinNetz()

if os.path.isfile('meinNetz_test.pt'):
    netz = torch.load('meinNetz_test.pt')

for i in range(100):
    # x = torch.randn(10, 10) # example inputs
    x = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    # input = Variable(x)

    out = netz(input)

    y = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]   # example target for one input
    target = Variable(torch.Tensor([y for _ in range(10)]))  # example target from all inputs
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    netz.zero_grad()
    loss.backward()
    optimizer = optim.SGD(netz.parameters(), lr=0.10)
    optimizer.step()

torch.save(netz, 'meinNetz_test.pt')