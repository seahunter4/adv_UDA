from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from trades import *
import numpy as np
import time
import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=1.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--save-model', default='smallCNN_cifar10_softmax',
                    help='directory of model for saving checkpoint')
parser.add_argument('--stats-dir', default='./stats-cifar-smallCNN/softmax',
                    help='directory of stas for saving checkpoint')
parser.add_argument('--model-dir', default='./model-cifar-smallCNN/softmax',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

stats_dir = args.stats_dir
if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='./data_attack/cifar10', train=True, download=True,
                                        transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data_attack/cifar10', train=False, download=True,
                                       transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, labels = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def main():
    # init model, ResNet18() can be also used here for training
    # model = WideResNet().to(device)
    model = SmallCNN().to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    natural_acc = []

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        start_time = time.time()
        # softmax training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        _, natural_err_total = eval_test(model, device, test_loader)
        print('using time:', datetime.timedelta(seconds=round(time.time() - start_time)))

        natural_acc.append(natural_err_total)
        print('================================================================')

        file_name = os.path.join(stats_dir, '{}_stat{}.npy'.format(args.save_model, epoch))
        # np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss),
        #                              np.array(self.train_acc), np.array(self.test_acc),
        #                              np.array(self.elasticity), np.array(self.x_grads),
        #                              np.array(self.fgsms), np.array(self.pgds),
        #                              np.array(self.cws))))
        np.save(file_name, np.array(natural_acc))

        # save checkpoint
        # if epoch % args.save_freq == 0:
        #     torch.save(model.state_dict(),
        #                os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
        #     torch.save(optimizer.state_dict(),
        #                os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '{}_ep{}.pt'.format(args.save_model, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}_ep{}.tar'.format(args.save_model, epoch)))
            print("Ep{}: Model saved as {}.".format(epoch, args.save_model))


if __name__ == '__main__':
    main()
