import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG19_Weights


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module( str(x), vgg_pretrained_features[x] )
        for x in range(2, 7):
            self.slice2.add_module( str(x), vgg_pretrained_features[x] )
        for x in range(7, 12):
            self.slice3.add_module( str(x), vgg_pretrained_features[x] )
        for x in range(12, 21):
            self.slice4.add_module( str(x), vgg_pretrained_features[x] )
        for x in range(21, 30):
            self.slice5.add_module( str(x), vgg_pretrained_features[x] )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)

        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module):
    def __init__(self, use_cuda=True, layids=None):
        super().__init__()
        self.vgg = Vgg19()
        if use_cuda:
            self.vgg.cuda()

        self.criterion = nn.L1Loss()
        self.weights = [ 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0 ]
        self.layids = layids

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)

        loss = 0

        if self.layids is None:
            self.layids = list(range(len(x_vgg)))

        for i in self.layids:
            loss += self.weights[i] * self.criterion( x_vgg[i], y_vgg[i].detach() )

        return loss

