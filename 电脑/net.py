# coding=utf-8

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class ANet(nn.Module):
    def __init__(self, s_info_dim=2, a_dim=2):
        super(ANet, self).__init__()
        # bachNorm && drop
        # self.bn = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(.5)
        flat_size = 48 * 80 * 60
        # flat_size = 48 * 40 * 30

        self.sigm = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.lr = nn.LeakyReLU()
        self.conv1 = nn.Sequential(  # input shape (3,640,480)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=24,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (24, 640, 480)
            nn.LeakyReLU(),    # nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (24, 320, 240)
            nn.BatchNorm2d(24)
        )
        self.conv2 = nn.Sequential(  # input shape (24, 320, 240)
            nn.Conv2d(24, 36, 5, 1, 2),  # output shape (36, 320, 240)
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (36, 160, 120)
            nn.BatchNorm2d(36)
        )
        self.conv3 = nn.Sequential(  # input shape (36, 160, 120)
            nn.Conv2d(36, 48, 5, 1, 2),  # output shape (48, 160, 120)
            nn.LeakyReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 80, 60)
            nn.BatchNorm2d(48)
        )

        # ------------fully connected layer---------------

        self.linCNN = nn.Linear(flat_size, 64)    # 512
        self.linI1 = nn.Linear(s_info_dim, 64)    # 512    input shape (1, 2, 2)
        # self.linI2 = nn.Linear(512, 128)    # 512   input shape (1, 1, 512)

        self.lin1 = nn.Linear(64 * s_info_dim + 64, a_dim)       # 512 + 128, 128
        # self.lin2 = nn.Linear(128, a_dim)

        # ---------------optimizer-----------
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, img, info):
        # to be modified
        # img, info = state.split([1, 5], dim=0)

        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = img.view(img.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        img = self.lr(self.linCNN(img))

        info = self.lr(self.linI1(info))
        # info = self.linI2(info)
        info = info.view(info.size(0), -1)

        output = self.lin1(torch.cat([img, info], 1))

        # output = self.lin2(output)
        output[:, 0] = 0.3 * self.sigm(output[:, 0].clone())
        output[:, 1] = 0.5 * self.tanh(output[:, 1].clone())
        # print(output)

        return output

    # def get_action(self, state):
    #     state = torch.FloatTensor(state).unsqueeze(0).to(device)
    #
    #     return self.forward(state).detach().cpu().numpy()[0, 0]

    def get_action(self, img, info):
        img = torch.FloatTensor(img).unsqueeze(0).to(device)
        info = torch.FloatTensor(info).unsqueeze(0).to(device)

        return self.forward(img, info).detach().cpu().numpy()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class CNet(nn.Module):
    def __init__(self, s_info_dim=2, a_dim=2):
        super(CNet, self).__init__()
        # bachNorm && drop
        # self.bn = nn.BatchNorm2d(32)
        # self.dropout = nn.Dropout(.5)
        flat_size = 48 * 80 * 60
        # flat_size = 48 * 40 * 30

        self.lr = nn.LeakyReLU()
        self.conv1 = nn.Sequential(  # input shape (3,640,480)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=24,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (24, 640, 480)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (24, 320, 240)
            nn.BatchNorm2d(24),
        )
        self.conv2 = nn.Sequential(  # input shape (24, 320, 240)
            nn.Conv2d(24, 36, 5, 1, 2),  # output shape (36, 320, 240)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (36, 160, 120)
            nn.BatchNorm2d(36),
        )
        self.conv3 = nn.Sequential(  # input shape (36, 160, 120)
            nn.Conv2d(36, 48, 5, 1, 2),  # output shape (48, 160, 120)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 80, 60)
            nn.BatchNorm2d(48),
        )

        # ------------fully connected layer---------------

        self.linCNN = nn.Linear(flat_size, 128)        # 512
        self.linI1 = nn.Linear(s_info_dim, 64)   # 512 # input shape (1, 3, 2)
        # self.linI2 = nn.Linear(512, 128)    # 256

        self.lin1 = nn.Linear(64 * (s_info_dim + 1) + 128, 128)  # 512 + 256, 256
        # self.lin2 = nn.Linear(256, 128)
        # self.lin3 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(128, 1)

        # ---------------optimizer-----------
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, img, info, action):
        # to be modified
        # img, info = state.split([1, 5], dim=0)

        img = self.conv1(img)
        img = self.conv2(img)
        img = self.conv3(img)
        img = img.view(img.size(0), -1)
        img = self.lr(self.linCNN(img))
        action = action.unsqueeze(0)

        infoA = self.lr(self.linI1(torch.cat([info, action], 1)))      # 2
        # infoA = self.linI2(infoA)

        infoA = infoA.view(infoA.size(0), -1)
        output = self.lr(self.lin1(torch.cat([img, infoA], 1)))
        output = self.lin2(output)
        # output = self.lin3(output)

        return output

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
