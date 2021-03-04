from collections import OrderedDict
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(SmallCNN, self).__init__()

        self.num_channels = 3
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 64, 3, padding=1)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(64, 64, 3, padding=1)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2)),           
            ('conv3', nn.Conv2d(64, 128, 3, padding=1)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(128, 128, 3, padding=1)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2)),
            ('conv5', nn.Conv2d(128, 196, 3, padding=1)),
            ('bn5', nn.BatchNorm2d(196)),
            ('relu5', activ),
            ('conv6', nn.Conv2d(196, 196, 3, padding=1)),
            ('bn6', nn.BatchNorm2d(196)),
            ('relu6', activ),
            ('maxpool3', nn.MaxPool2d(2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(196 * 4 * 4, 256))]))
        self.classifier_bn = nn.Sequential(OrderedDict([('bn1', nn.BatchNorm1d(256))]))
        self.classifier_relu = nn.Sequential(OrderedDict([('relu1', activ)]))
        self.classifier_fc = nn.Sequential(OrderedDict([('fc3', nn.Linear(256, self.num_labels))]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # nn.init.constant_(self.classifier.fc1.weight, 0)
        # nn.init.constant_(self.classifier.fc1.bias, 0)
        nn.init.constant_(self.classifier_fc.fc3.weight, 0)
        nn.init.constant_(self.classifier_fc.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        f = self.classifier(features.view(-1, 196 * 4 * 4))
        bn = self.classifier_bn(f)
        relu = self.classifier_relu(bn)
        fc = self.classifier_fc(relu)
        print("features {}\n"
              "f {}"
              "bn {}"
              "relu {}"
              "fc {}".format(features.size(), f, bn, relu, fc))
        return features, fc