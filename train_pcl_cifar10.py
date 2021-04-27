from __future__ import print_function
import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from proximity import Proximity
from contrastive_proximity import Con_Proximity
from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from trades import *
from utils import Logger
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--network', type=str, default='smallCNN')
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--stats-dir', default='./stats-cifar-smallCNN/pcl',
                    help='directory of stas for saving checkpoint')
parser.add_argument('--model-dir', default='./model-cifar-smallCNN/pcl',
                    help='directory of model for saving checkpoint')
parser.add_argument('--base-dir', default='./model-cifar-smallCNN/softmax',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-model', default='smallCNN_cifar10_pcl_advPGD',
                    help='directory of model for saving checkpoint')
parser.add_argument('--base-model', default='softmax_01',
                    help='directory of model for saving checkpoint')
parser.add_argument('--checkpoint', type=int, default=100, metavar='N',
                    help='')
parser.add_argument('--save-freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--lr-prox', type=float, default=0.5,
                    help="learning rate for Proximity Loss")
parser.add_argument('--weight-prox', type=float, default=1,
                    help="weight for Proximity Loss")
parser.add_argument('--lr-conprox', type=float, default=0.00001,
                    help="learning rate for Con-Proximity Loss")
parser.add_argument('--weight-conprox', type=float, default=0.00001,
                    help="weight for Con-Proximity Loss")
parser.add_argument('--weight-xent', type=float, default=1,
                    help="weight for Cross-Entropy Loss")
parser.add_argument('--sub-sample', action='store_true')
parser.add_argument('--sub-size', type=int, default=5000)
parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--only-adv', action='store_true')
parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--log-dir', default='./log/pcl',
                    help='directory of model for saving checkpoint')
parser.add_argument('--log-file', default='tct.log',
                    help='directory of model for saving checkpoint')
parser.add_argument('--feat-size', type=int, default=256)
parser.add_argument('--adv-train-iters', type=int, default=10)
parser.add_argument('--adv-eval-iters', type=int, default=7)

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
trainset = torchvision.datasets.CIFAR10(root='./data_attack/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root='./data_attack/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)


def train(model, device, train_loader, optimizer,
          criterion_prox, optimizer_prox,
          criterion_conprox, optimizer_conprox, epoch):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model.eval()
        adv_data = generate_adv_data(model=model,
                                     x_natural=data,
                                     y=target,
                                     optimizer=optimizer,
                                     step_size=args.adv_train_iters,
                                     epsilon=args.epsilon,
                                     perturb_steps=args.num_steps,
                                     beta=args.beta)
        if args.only_adv:
            data = adv_data
            labels = target
        else:
            true_labels = target
            data = torch.cat((data, adv_data), 0)
            labels = torch.cat((target, true_labels))
        model.train()
        feats, logits = model(data)
        # print(feats)
        # print(logits)
        loss_xent = F.cross_entropy(logits, labels)
        loss_prox = criterion_prox(feats, labels)
        loss_conprox = criterion_conprox(feats, labels)
        loss = args.weight_xent * loss_xent + args.weight_prox * loss_prox - args.weight_conprox * loss_conprox
        optimizer.zero_grad()
        optimizer_prox.zero_grad()
        optimizer_conprox.zero_grad()

        loss.backward()
        optimizer.step()
        optimizer_prox.step()
        optimizer_conprox.step()

        # print progress
        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} ce: {:.6f} prox: {:6f} conprox: {:6f} takes {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       loss_xent.item(), loss_prox.item(), loss_conprox.item(),
                datetime.timedelta(seconds=round(time.time() - start_time))))
            start_time = time.time()


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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
            output = model(data)
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


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.adv_eval_iters,
                  step_size=args.step_size):
    _, out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    # if args.random:
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            _, logits = model(X_pgd)
            loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    _, output = model(X_pgd)
    err_pgd = (output.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))
    return 1 - natural_err_total / len(test_loader.dataset), 1 - robust_err_total / len(test_loader.dataset)


def main():
    # init model, ResNet18() can be also used here for training
    # model = WideResNet().to(device)
    if args.network == 'smallCNN':
        model = SmallCNN().to(device)
    elif args.network == 'wideResNet':
        model = WideResNet().to(device)
    elif args.network == 'resnet':
        model = ResNet().to(device)
    sys.stdout = Logger(os.path.join(args.log_dir, args.log_file))
    print(model)
    criterion_prox = Proximity(10, args.feat_size, True)
    criterion_conprox = Con_Proximity(10, args.feat_size, True)
    optimizer_prox = optim.SGD(criterion_prox.parameters(), lr=args.lr_prox)
    optimizer_conprox = optim.SGD(criterion_conprox.parameters(), lr=args.lr_conprox)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.fine_tune:
        base_dir = args.base_dir
        state_dict = torch.load("{}/{}_ep{}.pt".format(base_dir, args.base_model, args.checkpoint))
        opt = torch.load("{}/opt-{}_ep{}.tar".format(base_dir, args.base_model, args.checkpoint))
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(opt)


    natural_acc = []
    robust_acc = []

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer_prox, epoch)
        adjust_learning_rate(optimizer_conprox, epoch)

        start_time = time.time()

        # adversarial training
        train(model, device, train_loader, optimizer,
              criterion_prox, optimizer_prox,
              criterion_conprox, optimizer_conprox, epoch)

        # evaluation on natural examples
        print('================================================================')
        print("Current time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        # eval_train(model, device, train_loader)
        # eval_test(model, device, test_loader)
        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader)
        with open(os.path.join(stats_dir, '{}.txt'.format(args.save_model)), "a") as f:
            f.write("{} {} {}\n".format(epoch, natural_err_total, robust_err_total))

        print('using time:', datetime.timedelta(seconds=round(time.time() - start_time)))

        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)


        file_name = os.path.join(stats_dir, '{}_stat{}.npy'.format(args.save_model, epoch))
        # np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss),
        #                              np.array(self.train_acc), np.array(self.test_acc),
        #                              np.array(self.elasticity), np.array(self.x_grads),
        #                              np.array(self.fgsms), np.array(self.pgds),
        #                              np.array(self.cws))))
        np.save(file_name, np.stack((np.array(natural_acc), np.array(robust_acc))))

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '{}_ep{}.pt'.format(args.save_model, epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}_ep{}.tar'.format(args.save_model, epoch)))
            print("Ep{}: Model saved as {}.".format(epoch, args.save_model))
        print('================================================================')


if __name__ == '__main__':
    main()
