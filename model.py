import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
color = 3
filter_k = 64
embedding_len = 128
def CONV1_1(inplanes,planes,s = 1, p = 0):
        return nn.Conv2d(inplanes, planes, 1, stride = s, padding = p, bias=False)

class BottleNeck(nn.Module):
    """docstring for BottleNeck"""
    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out, 0.1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = F.leaky_relu(out, 0.1)

        return out

#####Generator input batch_size*4*24*24  and  output batch_size*3*8*8######
class G(nn.Module):
    """docstring for G"""
    def __init__(self):
        super(G, self).__init__()

        self.conv_1 = nn.Conv2d(color+1, filter_k, 7, 1, bias=False)
        self.BN1 = nn.BatchNorm2d(filter_k)
        self.conv_7 = nn.Conv2d(filter_k,filter_k, 3, 1, bias=False)
        self.layer_1 = BottleNeck(64, 64)
        self.layer_2 = BottleNeck(64, 64)
        self.conv_2 = CONV1_1(64,128)
        self.BN2 = nn.BatchNorm2d(128)
        self.layer_3 = BottleNeck(128, 128)
        # self.layer_4 = BottleNeck(128, 128)
        self.conv_3 = CONV1_1(128,256)
        self.BN3 = nn.BatchNorm2d(256)
        self.layer_5 = BottleNeck(256,256)
        # self.layer_6 = BottleNeck(256,256)
        # self.layer_7 = BottleNeck(256,256)
        # self.layer_8 = BottleNeck(256,256)
        # self.conv_4 = CONV1_1(256,512)
        # self.BN4 = nn.BatchNorm2d(512)
        # self.layer_9 = BottleNeck(512,512)
        # self.layer_10 = BottleNeck(512,512)
        # self.conv_5 = CONV1_1(512,1024)
        # self.BN5 = nn.BatchNorm2d(1024)
        # self.layer_11 = BottleNeck(1024,1024)
        self.conv_6 = CONV1_1(256,color)



    def forward(self, x):
        x = self.conv_1(x)
        x = self.BN1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.conv_7(x)
        x = self.BN1(x)
        x = F.leaky_relu(x, 0.1)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = F.leaky_relu(self.BN2(self.conv_2(x)), 0.1)

        x = self.layer_3(x)
        # x = self.layer_4(x)
        x = F.leaky_relu(self.BN3(self.conv_3(x)), 0.1)
        x = self.layer_5(x)
        # x = self.layer_6(x)
        # x = self.layer_7(x)
        # x = self.layer_8(x)
        # x = F.leaky_relu(self.BN4(self.conv_4(x)), 0.1)
        # x = self.layer_9(x)
        # x = self.layer_10(x)
        # x = F.leaky_relu(self.BN5(self.conv_5(x)), 0.1)
        # x = self.layer_11(x)
        x = self.conv_6(x)

        return F.sigmoid(x)

#####Generator input batch_size*3*8*8  and  output batch_size*512*1*1######
class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(3, 64, 4, stride = 1, bias=False),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 5, stride=2, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 128, 3, stride=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 128, 3, stride=1, bias=False),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 128, 3, stride=1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),


    )

  def forward(self, x):
    output = self.main(x)
    return output



class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(128, 1024, 2, stride=1, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 1024, 2, stride=1, bias=False),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 1, 1),
    )


  def forward(self, x):
    output = self.main(x).view(-1, 1)
    return output


class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()
    self.fc1 = nn.Linear(128 * 3 * 3, 1024)
    self.fc_q = nn.Linear(1024,embedding_len)

    self.bn1 = nn.BatchNorm1d(1024)
    self.bn_q = nn.BatchNorm1d(embedding_len)
    # self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    # self.bn = nn.BatchNorm2d(128)
    # self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    # self.conv_disc = nn.Conv2d(128, 4, 1)
    # # self.conv_mu = nn.Conv2d(128, 2, 1)
    # # self.conv_var = nn.Conv2d(128, 2, 1)

  def forward(self, x):
      x = F.leaky_relu(self.bn1(self.fc1(x.view(-1,128*3*3))))
      return F.leaky_relu(self.bn_q(self.fc_q(x)))

# def weight_init(m):
#     if isinstance(m, nn.Conv2d):
#         size = m.weight.size()
#         fan_out = size[0] # number of rows
#         fan_in = size[1] # number of columns
#         variance = np.sqrt(2.0/(fan_in + fan_out))
#         m.weight.data.normal_(0.0, variance)

########################################GAN_init#########################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
