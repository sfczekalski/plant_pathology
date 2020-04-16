import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        fc_in_shape = self.resnet.fc.in_features

        self.logit = nn.Linear(fc_in_shape, 4)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x
