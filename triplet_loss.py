import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=1024, use_gpu=True):
        super(TripletLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        # if self.use_gpu:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        # else:
        #     self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, margin):
        batch_size = x.size(0) // 2
        # x = F.normalize(x, p=2, dim=1)
        ori, adv = x.chunk(2, dim=0)
        # centers = F.normalize(self.centers, p=2, dim=1)
        distmat = torch.pow(adv, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size) + \
                  torch.pow(ori, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size).t()
        distmat.addmm_(1, -2, adv, ori.t())

        # classes = torch.arange(self.num_classes).long()
        # if self.use_gpu:
        #     classes = classes.cuda()
        print("ori labels: {}".format(labels))
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        print("now labels: {}".format(labels))
        mask = labels.eq(labels.expand(batch_size, self.num_classes))
        print("mask: {}".format(mask))
        zero = torch.tensor([0.]).cuda()

        dist = []
        for i in range(batch_size):
            congener_dist = distmat[i][i]
            congener_marks = mask[i].clone()
            inhomogen_marks = (-1 * congener_marks + 1).bool()
            nearst_inhomogen_dist = torch.min(distmat[i][inhomogen_marks])
            congener_dist = congener_dist.clamp(min=1e-12, max=1e+12)
            nearst_inhomogen_dist = nearst_inhomogen_dist.clamp(min=1e-12, max=1e+12)
            dist.append(max(congener_dist-nearst_inhomogen_dist+margin, zero))
        # print(dist)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss





