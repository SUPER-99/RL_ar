# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cv2
import pickle
from net import ANet, CNet
import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


class CriticNode11(nn.Module):
    def __init__(self):
        super(CriticNode11, self).__init__()
        # 定义完成功能所需的模�?
        self.actor = ANet().to(device)
        self.actor_target = ANet().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = CNet().to(device)
        self.critic_target = CNet().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # 要初始化网络，就注释掉load
        # 若要在之前的dict上训练，就加上load
        self.load("dict/dict11", "criticDict", "actorDict")

        self.discount = 0.99
        self.tau = 0.001
        self.actorDict = {}
        self.criticDict = {}
        print("初始化网络11完成")

    def optimize(self, byteData):
        print("开始优化网络11...")
        # -----------self.critic(data)----------------
        # to be modified:  model.train()
        for _, sample in byteData.items():
            img = torch.FloatTensor(sample["lastImg"]).unsqueeze(0).to(device)
            info = torch.FloatTensor(sample["lastInfo"]).unsqueeze(0).to(device)
            action = torch.FloatTensor(sample["a"]).to(device)
            img_next = torch.FloatTensor(sample["img"]).unsqueeze(0).to(device)
            info_next = torch.FloatTensor(sample["info"]).unsqueeze(0).to(device)
            reward = torch.FloatTensor(sample["r"]).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(img_next, info_next, self.actor_target(img_next, info_next))
            target_Q = reward + (self.discount * target_Q).detach()
            # print("target_Q", target_Q)

            # Get current Q estimate
            current_Q = self.critic(img, info, action)
            # print("current_Q", current_Q)

            # Compute critic loss and Optimize the critic
            critic_loss = F.mse_loss(current_Q, target_Q)
            # print("critic_loss", critic_loss)
            self.critic.optimize(critic_loss)

            # Compute actor loss and Optimize the actor
            actor_loss = -self.critic(img, info, self.actor(img, info)).mean()
            # print("actor_loss", actor_loss)
            self.actor.optimize(actor_loss)
            # -----------self.critic(data)----------------

            # 读取并保存神经网络参数
            self.actorDict = self.actor.state_dict()
            # print("actor网络参数已获取", len(pickle.dumps(self.actorDict)))
            self.criticDict = self.critic.state_dict()
            torch.save(self.criticDict, "{}/{}.dict".format("dict/dict11", "criticDict"))
            torch.save(self.actorDict, "{}/{}.dict".format("dict/dict11", "actorDict"))

        # 更新target网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            

    def load(self, directory, filenameC, filenameA):
        self.critic.load_state_dict(torch.load("{}/{}.dict".format(directory, filenameC), map_location=device))
        self.actor.load_state_dict(torch.load("{}/{}.dict".format(directory, filenameA), map_location=device))
        self.critic.eval()
        print("Critic网络参数已加载")
