from numpy.lib.histograms import histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

batch_size = 16

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
#         self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(18432, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

# ====================================================================================================================
import numpy as np

class NNBackbone2(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, num_classes),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out

class ZXCNN2(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.back_bone = NNBackbone2(in_features, num_classes, dim)
        self.fc = nn.Linear(28*28, 28*28)

    def forward(self, x):
        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.reshape(shape_)

        out = self.back_bone(x)
        # out = self.fc(out)
        return out

class ZXCNN3(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        # super(Mclr_Logistic, self).__init__()
        super(ZXCNN3, self).__init__()
        # self.back_bone = NNBackbone2(in_features, num_classes, dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, num_classes),
            # nn.ReLU(inplace=True)
        )



        self.fc = nn.Linear(28*28, 28*28)

    def forward(self, x):
        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.reshape(shape_)

        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


class ZXCNN4(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        # super(Mclr_Logistic, self).__init__()
        super(ZXCNN4, self).__init__()
        # self.back_bone = NNBackbone2(in_features, num_classes, dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes)
        )



        self.fc = nn.Linear(28*28, 28*28)

    def forward(self, x):
        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.reshape(shape_)

        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out



# zengxiang
class NNBackbone(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 28*28),
            nn.ReLU(inplace=True)
            # nn.Linear(28*28, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return out


class ZXCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super(ZXCNN, self).__init__()
        self.back_bone = NNBackbone(in_features, num_classes, dim)
        self.fc = nn.Linear(28*28, num_classes)

    def forward(self, x):
        out = self.back_bone(x)
        out = self.fc(out)
        return out

class ZXCNN_V1(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super(ZXCNN_V1, self).__init__()
        self.back_bone = ZXCNN(in_features, num_classes, dim)
        self.base_fc = nn.Linear(28*28, 28*28)

    def forward(self, x):
        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.base_fc(x)
        x = x.reshape(shape_)

        out = self.back_bone(x)
        return out

class My_Logistic():
    def __init__(self, input_dim=1 * 28 * 28, num_labels=10):
        self.global_theta = np.zeros((input_dim + 1, num_labels))

class Linear_recognation(nn.Module):
    def __init__(self, input_dim=100, num_labels=10):
        super(Linear_recognation, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=100, num_labels=10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_labels)
        # 2.25 test matrix transform
        # self.fn = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)

        # # 2.25
        # x = self.fn(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


# 2.25
class Mclr_FN(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_labels=10):
        super(Mclr_FN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_labels)

        self.fn_base = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fn_base(x)

        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_labels=10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs


# ====================================================================================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100, num_labels=10):
        super(DNN, self).__init__()
        # self.fc1 = nn.Linear(input_dim, mid_dim)
        # # 添加一层
        # self.fc_tmp = nn.Linear(mid_dim, mid_dim)
        # self.fc2 = nn.Linear(mid_dim, num_labels)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 784),
            nn.Sigmoid(),
            # # nn.Sigmoid()
            # nn.Dropout(0.1),
            # nn.Linear(300, 300),
            # nn.Sigmoid(),
            # nn.Dropout(0.1),
            # nn.Linear(300, num_labels),
            # nn.Sigmoid()
            nn.Linear(784, 300),
            nn.LeakyReLU(inplace=True),
            nn.Linear(300, 300),
            nn.LeakyReLU(inplace=True),
            nn.Linear(300, 300),
            nn.LeakyReLU(inplace=True),
            nn.Linear(300, 300),
            nn.LeakyReLU(inplace=True),
            nn.Linear(300, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.model(x)
        return x

    # def forward(self, x):
    #     x = torch.flatten(x, 1)
    #     # x = F.relu(self.fc1(x))
    #
    #     x = self.fc1(x)
    #     x = F.softmax(x)
    #
    #     x = F.dropout(x, 0.1)
    #
    #     # 添加一层
    #     x = self.fc_tmp(x)
    #     x = F.softmax(x)
    #     x = F.dropout(x, 0.1)
    #
    #     x = self.fc2(x)
    #     # x = F.log_softmax(x, dim=1)
    #     x = F.softmax(x, dim=1)
    #     return x


class DNNbase(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, mid_dim=100):
        super(DNNbase, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x


class DNNClassifier(nn.Module):
    def __init__(self, mid_dim=100, num_labels=10):
        super(DNNClassifier, self).__init__()
        self.fc2 = nn.Linear(mid_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc2(x)

        # x = F.dropout(x, 0.5)

        x = F.log_softmax(x, dim=1)
        return x


class My_DNN(nn.Module):
    def __init__(self, mid_dim=100, num_labels=10):
        super(My_DNN, self).__init__()
        self.fc1 = nn.Linear(mid_dim, 30)
        # self.fc2 = nn.Linear(50, 100)
        # self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(30, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.batch_norm(x, x, x)
        x = F.relu(x)

        # x = self.fc2(x)
        # # # x = F.batch_norm(x, x, x)
        # x = F.relu(x)
        #
        # x = self.fc3(x)
        # # x = F.batch_norm(x, x, x)
        # x = F.relu(x)

        x = self.fc4(x)

        # x = F.log_softmax(x, dim=1)
        return x


class My_DNN_fn(nn.Module):
    def __init__(self, mid_dim=100, num_labels=10):
        super(My_DNN_fn, self).__init__()
        self.fc1 = nn.Linear(mid_dim, 30)
        self.fc4 = nn.Linear(30, num_labels)

        self.base_fc = nn.Linear(mid_dim, mid_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.base_fc(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc4(x)

        return x


class My_DNN_NET(nn.Module):
    def __init__(self, in_dim, hidden_1, hidden_2, hidden_3, num_labels=10):
        super(My_DNN_NET, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_1, bias=True), nn.BatchNorm1d(hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(hidden_1, hidden_2, bias=True), nn.BatchNorm1d(hidden_2))
        # self.layer3 = nn.Sequential(nn.Linear(hidden_2, hidden_3, bias=True), nn.BatchNorm1d(hidden_3))
        self.layer4 = nn.Sequential(nn.Linear(hidden_2, num_labels))

        # self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_1, bias=True))
        # self.layer2 = nn.Sequential(nn.Linear(hidden_1, hidden_2, bias=True))
        # self.layer3 = nn.Sequential(nn.Linear(hidden_2, hidden_3, bias=True))
        # self.layer4 = nn.Sequential(nn.Linear(hidden_3, num_labels))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        x = self.layer4(x)
        # x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

class CifarNet(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

class CifarNet_test(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNet_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, num_labels)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

class CifarNet_fn_test(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNet_fn_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, num_labels)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_labels)

        self.base_fc = nn.Linear(3*32*32, 3*32*32)

    def forward(self, x):

        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.base_fc(x)
        x = x.reshape(shape_)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = self.fc1(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


# class CifarNetHead(nn.Module):
#     def __init__(self):
#         super(CifarNetHead, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return x

# class CifarNetBase(nn.Module):
#     def __init__(self):
#         super(CifarNetBase, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)

#     def forward(self, x):
#         x = self.pool(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

class CifarNetBase(nn.Module):
    def __init__(self):
        super(CifarNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class CifarNetClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNetClassifier, self).__init__()
        self.fc = nn.Linear(84, num_labels)

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ====================================================================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class MnistLeNetSmall(nn.Module):
    def __init__(self):
        super(MnistLeNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class LeNet(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_labels=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class Cifar10LeNetSmall(nn.Module):
    def __init__(self, num_labels=10):
        super(Cifar10LeNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.fc1 = nn.Linear(12*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_labels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet_fn(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_labels=10, iswn=None):
        super(LeNet_fn, self).__init__()

        self.conv_params = nn.Sequential(
            # nn.Conv2d(1, 20, kernel_size=5),
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        # self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck = nn.Linear(1250, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

        self.fn_base = nn.Linear(3*32*32, 3*32*32)

    def forward(self, x):

        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fn_base(x)
        x = x.reshape(shape_)

        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LeNet_fn_zx(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256, num_labels=10, iswn=None):
        super(LeNet_fn_zx, self).__init__()

        self.fn_base = nn.Linear(3*32*32, 3*32*32)

        self.conv_params = nn.Sequential(
            # nn.Conv2d(1, 20, kernel_size=5),
            # nn.Conv2d(3, 6, kernel_size=5),
            nn.Conv2d(3, 20, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        # self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck = nn.Linear(1250, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)



    def forward(self, x):

        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fn_base(x)
        x = x.reshape(shape_)

        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)

        x = self.fc(x)
        return x


class LeNetBase(nn.Module):
    def __init__(self, feature_dim=50 * 4 * 4, bottleneck_dim=256):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class LeNetClassifier(nn.Module):
    def __init__(self, num_labels=10, bottleneck_dim=256, iswn=None):
        super(LeNetClassifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


# class LeNetBase(nn.Module):
#     def __init__(self):
#         super(LeNetBase, self).__init__()
#         self.conv_params = nn.Sequential(
#                 nn.Conv2d(1, 20, kernel_size=5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 nn.Conv2d(20, 50, kernel_size=5),
#                 nn.Dropout2d(p=0.5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 )
#         self.in_features = 50*4*4

#     def forward(self, x):
#         x = self.conv_params(x)
#         x = x.view(x.size(0), -1)
#         return x

# class LeNet_bootleneck(nn.Module):
#     def __init__(self, feature_dim, bottleneck_dim=256, iswn="bn"):
#         super(LeNet_bootleneck, self).__init__()
#         self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
#         # self.bottleneck.apply(init_weights)
#         self.iswn = iswn

#     def forward(self, x):
#         x = self.bottleneck(x)
#         if self.iswn == "bn":
#             x = self.bn(x)
#             x = self.dropout(x)
#         return x

# ====================================================================================================================

class CNNCifar(nn.Module):
    def __init__(self, num_labels=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, num_labels)

        # self.weight_keys = [['fc1.weight', 'fc1.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CNNMnist(nn.Module):
    def __init__(self, num_labels=10):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, num_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x


class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 1, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 1)
        x = self.dense(x)
        return x

class MyCNN_fn(torch.nn.Module):
    def __init__(self):
        super(MyCNN_fn, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 1, 10)
        )

        self.fn = nn.Linear(784, 784)

    def forward(self, x):
        shape_ = x.size()
        x = torch.flatten(x, 1)
        x = self.fn(x)
        x = x.reshape(shape_)

        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 1)
        x = self.dense(x)
        return x


# ====================================================================================================================

class ResNetClassifier(nn.Module):
    def __init__(self, input_dim=512, num_labels=10):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ====================================================================================================================

# model for sent140
class NetMLP(nn.Module):
    '''
        PyTorch nn.Module for a Multilayer Perceptron.
    '''

    def __init__(self, input_size, layer_sizes, activation=nn.ReLU(), dropout=0.1):
        '''
            Constructor for neural network; defines all layers and sets attributes for optimization.
        '''
        super(NetMLP, self).__init__()

        # NN layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_sizes[0])])
        self.layers.extend(nn.Linear(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, len(layer_sizes)))
        self.layers.append(nn.Linear(layer_sizes[-1], 1))

        # activation functions
        self.activation = activation
        self.finalActivation = nn.Sigmoid()

        # # optimization attributes
        # self.epochs = epochs
        # self.learning_rate = learning_rate
        # self.l2reg = l2reg

        # loss and optimizer
        self.criterion = nn.BCELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
        #                                   weight_decay=self.l2reg)

    def forward(self, x):
        '''
            Forward pass through MLP.
        '''
        out = x
        # loop through all layers except for last one
        for layer in self.layers[:-1]:
            out = self.activation(layer(self.dropout(out)))
        out = self.finalActivation(self.layers[-1](out))
        return out


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        # _x = self.one_hot_encode(_x, 65)

        x, (_, h) = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # s, h = x.shape
        # b = 1

        # x = x.reshape(x.size()[0] * x.size()[1], x.size()[2])

        x = self.linear1(x)
        # x = x.view(s, b, -1)
        # x = x.view(s, -1)
        return x

    def one_hot_encode(self, arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.int().flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return torch.from_numpy(one_hot)


class LstmRNN_single(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        seq_len = 50

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size * seq_len, output_size)  # 全连接层

    def forward(self, _x):
        # _x = self.one_hot_encode(_x, 65)

        x, (_, h) = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # s, h = x.shape
        # b = 1

        x = x.reshape(x.size()[0], x.size()[1] * x.size()[2])

        x = self.linear1(x)
        # x = x.view(s, b, -1)
        # x = x.view(s, -1)
        return x

    def one_hot_encode(self, arr, n_labels):
        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.int().flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return torch.from_numpy(one_hot)


class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635, num_labels=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_labels)

    def forward(self, x):
        text = x
        text_lengths = len(text[0])

        text = torch.flatten(text, )

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))

        return out


class LSTMNetBase(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

    def forward(self, x):
        text, text_lengths = x

        embedded = self.embedding(text)

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)

        return out


class LSTM_class(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2,
                 padding_idx=0, vocab_size=98635):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=32,
                            num_layers=1,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

    def forward(self, x):
        text = x
        # text_lengths = len(x)
        #
        # embedded = self.embedding(text)
        #
        # # pack sequence
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
        #                                                     enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(x)

        # unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:, -1, :])
        out = self.dropout(out)

        return out


class LSTMNetClassifier(nn.Module):
    def __init__(self, hidden_dim, bidirectional=False, num_labels=10):
        super().__init__()

        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_labels)

    def forward(self, out):
        out = torch.sigmoid(self.fc(out))

        return out


# ====================================================================================================================

class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_labels=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc2 = nn.Linear(hidden_dim, num_labels)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        return self.softmax(z)


# ====================================================================================================================

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.8,
                 padding_idx=0, vocab_size=98635, num_labels=10):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=50, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_labels)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        # text, text_lengths = x

        # embedded_sent = self.embedding(text).permute(0,2,1)
        # embedded_sent = self.embedding(x.int()).permute(0, 2, 1, 3)

        # conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out1 = self.conv1(x).squeeze(2)

        conv_out2 = self.conv2(x).squeeze(2)
        conv_out3 = self.conv3(x).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)

# ====================================================================================================================


# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input

#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output
