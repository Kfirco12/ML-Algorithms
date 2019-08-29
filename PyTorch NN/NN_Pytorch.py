from gcommand_loader import GCommandLoader
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Net(nn.Module):
    '''
    This class is represent our model. one network with 2 convolution layers for feature extraction
    and 2 fully connect linear layers for prediction using PyTorch implementation.
    '''

    def __init__(self, input_size=64000, hidden_size=128, output_size=30):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def train(epoch):
    '''
    Training process
    :param epoch: number of epochs
    '''

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # if (batch_idx + 1) % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
        #                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))

def evaluate(data_loader):
    '''
    Evaluating process.
    :param data_loader: data loader object of the evaluated data
 
    '''

    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    # print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
    #     loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
    return loss


def prediciton(data_loader):
    '''
    Prediction of data loader data
    :param data_loader: data loader object data
    '''

    model.eval()
    test_pred = torch.LongTensor()

    for i, (data, label) in enumerate(data_loader):
        data = Variable(data, volatile=True)
        if torch.cuda.is_available():
            data = data.cuda()

            output = model(data)

            pred = output.cpu().data.max(1, keepdim=True)[1]
            test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred

def load_data(train_path = r'./data/train', valid_path = r'./data/valid', test_path = r'./data/test'):
    ''' This method gets paths for .wav files arrange by folders of validation,training and test.
    each folder contains .wav files of speech commands into folders named as the right labels.
    :param train_path: training set path
    :param valid_path: validation set path
    :param test_path: test set path
    :return: train, validation and test loaders objects '''

    train_set = GCommandLoader(train_path)
    valid_set = GCommandLoader(valid_path)
    test_set = GCommandLoader(test_path)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

    return train_loader, valid_loader, test_loader


def test_results(pred_vec, names, file_name='test_y'):
    '''
    Prints prediction to 'test_y' file
    :param pred_vec: prediction vector
    :param names: test files names
    :param file_name: output file name
    '''

    with open(file_name, 'w')as file:
        for item in zip(names, pred_vec.flatten()):
            file.write("{}, {}\n".format(item[0], item[1]))

# Load data
train_loader, valid_loader, test_loader = load_data()

# Define the network, optimizer and loss.
model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# Train the model and evaluate the loss on validation set.
n_epochs = 100
loss_arr = []

for epoch in range(n_epochs):
    train(epoch)
    loss_arr.append(evaluate(valid_loader))

# Predict test set
test_pred = prediciton(test_loader).numpy()

# Print results to 'test_y' file
names = [Path(path).name for path, _ in test_loader.dataset.spects]
test_results(test_pred, names)

