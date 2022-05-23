#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from __future__ import division

# ------------ 定义必要模块 ------------
import rospy
import torch
import torch.nn as nn
import message_filters
import numpy as np
from gym import spaces
import cv2
import time
import cv_bridge
import pickle
from net import ANet, CNet
import platform
import random
import os
from get_reward import Reward
from math import *

# ------------ 定义需要的传感器信息类型以及发布信息类 ------------
from sensor_msgs.msg import Image
from simple_follower.msg import position as PositionMsg
from std_msgs.msg import String
from std_msgs.msg import Int8
from dynamic_reconfigure.server import Server
from sensor_msgs.msg import Joy, LaserScan
from geometry_msgs.msg import Twist

device = "cpu"
store_path = "/home/wheeltec/wheeltec_robot/src/wang/scripts/var/"
print(platform.python_version())


# ------------ 定义ActorNode------------
class ActorNode(nn.Module):
    def __init__(self):
        super(ActorNode, self).__init__()
        # 定义完成功能所需的模块
        self.actor = ANet().to(device)

        # 定义所需的sub对象
        im_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
        # self.dep_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        position_sub = message_filters.Subscriber('/object_tracker/current_position', PositionMsg)
        scan_sub = message_filters.Subscriber('/scan', LaserScan)
        coord_sub = message_filters.Subscriber('camera_coord', String)
        self.ts = message_filters.ApproximateTimeSynchronizer([im_sub, position_sub, coord_sub], 10, 0.1,
                                                              allow_headerless=True)
        self.ts.registerCallback(self.recipe_ready)
        self.optimize_ready_sub = rospy.Subscriber('/train/optimizeready', String, self.updatedict)

        # 定义所需的pub对象
        self.transmission_ready_pub = rospy.Publisher('/train/transmissionready', String, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.twist = Twist()

        # 定义一些需要使用到的变量
        self.bridge = cv_bridge.CvBridge()
        self.store_path = store_path
        self.store_count = 0
        # 图像信息
        self.img_arr = 0
        self.last_img = 0
        # 雷达信息
        self.info_arr = 0
        self.last_info = 0
        # 动作
        self.a = [[0, 0]]
        self.last_a = [[0, 0]]
        # 坐标
        self.coord = [[[[0, 0, 0], [0, 0, 0]], 0]]
        self.last_coord = [[[[0, 0, 0], [0, 0, 0]], 0]]
        self.x = 0
        self.y = 0
        # 动作空间，范围
        self.action_space = spaces.Box(
            low=np.array([0, -2.0], dtype=np.float32),
            high=np.array([0.5, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        self.all_info = {}
        self.storeTime = 0
        self.angle = 0
        self.total_angle = 0
        self.g = 0

    def recipe_ready(self, image_data, position_data, coord_data):
        # 根据当前时刻的信息去对上一时刻的状态动作做评估
        # 传感器信息同步后触发的回调函数
        print("信息过滤器启动...")
        print("初始内存...")
        os.system('free -h')
        
        s_time = time.time()
        # print('开始时间', s_time)
        # 处理传感器信息
        self.last_img = self.img_arr
        self.last_info = self.info_arr
        self.last_coord = self.coord
        self.last_a = self.a

        # 图像信息
        img_arr = self.bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')  # shape: (480, 640, 3)
        self.img_arr = np.transpose(img_arr, (2, 1, 0))  # shape: (3, 640, 480)
        # print(self.img_arr.shape)

        # 坐标信息
        self.coord = eval(coord_data.data)
        print('actor', self.coord)
        print('last', self.last_coord)
        if not self.coord[0][0][0]:
            self.x = self.last_coord[0][0][0][0]
            self.y = self.last_coord[0][0][0][1]
            self.coord[0][0][0] = self.last_coord[0][0][0]
        else:
            self.x = self.coord[0][0][0][0]
            self.y = self.coord[0][0][0][1]
            
        if not self.coord[0][0][1]:
            self.coord[0][0][1] = self.last_coord[0][0][1]
        print(self.coord)

        # 雷达信息
        angle = position_data.angleX
        distance = position_data.distance
        self.info_arr = np.array([[angle, distance], [self.x, self.y]])  # shape: (2, 2)
        # print(self.info_arr.shape)

        a_dim = 2  # [线速度, 角速度]

        self.storeTime = rospy.get_time()

        # 推理动作
        self.a = self.actor.get_action(self.img_arr, self.info_arr)
        # print('动作', self.a)
        self.total_angle += self.a[0][1]
        # 避免车子做出掉头的行为
        if self.total_angle * (180 / pi) > 90 or self.total_angle * (180 / pi) < -90:
            self.a[0][1] = -self.a[0][1]
            self.total_angle += 2 * self.a[0][1] 

        reward = Reward(self.coord, self.last_coord, self.last_a, self.a, self.total_angle)

        # 收集四条传一次
        if self.store_count != 0:
            r, self.g = reward.reward_sum()
            # print('r=', r)
            info = {"storeTime": self.storeTime, "img": self.img_arr, "info": self.info_arr, "a": self.last_a,
                    "lastImg": self.last_img, "lastInfo": self.last_info, "r": r}
            self.all_info[self.storeTime] = info
            if self.store_count % 4 == 0 and self.store_count != 0:
                store_path = "{}{}.info".format(self.store_path, self.storeTime)
                with open(store_path, "wb") as f:
                    pickle.dump(self.all_info, f)
                    print("info数据已储存...")
                f.close()
                self.transmission_ready_pub.publish(store_path)
                self.all_info = {}

        #     # 设置buffer
        #     if len(self.all_info) <= 18:
        #         self.all_info[self.storeTime] = info
        #     else:
        #         # 超出则随机删除一条，并添加新的
        #         # key = random.sample(self.all_info.keys(), 1)
        #         # key获取的值是['key'],key[0]的值才是key
        #         # def self.all_info[key[0]]
        #         del self.all_info[random.sample(self.all_info.keys(), 1)[0]]
        #         self.all_info[self.storeTime] = info
        #
        # # 储存变量并通知（NOTE：此处存储变量的方式以及通知的时间可以另行决定）
        # # 随机挑选100条要进行训练的数据，并发布
        # if self.store_count % 4 == 0 and self.store_count != 0:
        #     sample = self.randomSample(self.all_info, 4)
        #     store_path = "{}{}.info".format(self.store_path, self.storeTime)
        #     with open(store_path, "wb") as f:
        #         pickle.dump(sample, f)
        #         # del sample
        #         print("info数据已储存...")
        #     f.close()
        #     self.transmission_ready_pub.publish(store_path)
        
        # 执行动作
        if self.g == 1:
            self.twist.linear.x = 0
            self.twist.angular.z = 0
        else:    
            self.twist.linear.x = self.a[0][0]
            self.twist.angular.z = self.a[0][1]
        self.cmd_vel_pub.publish(self.twist)
        
        time.sleep(0.5)
        self.twist.linear.x = 0
        self.twist.angular.z = self.a[0][1]
        # print(self.twist)
        self.cmd_vel_pub.publish(self.twist)

        self.store_count += 1

    # actor神经网络参数更新
    def updatedict(self, store_net_path):
        # ---------loading------------
        self.actor.load_state_dict(torch.load(store_net_path.data))
        self.actor.eval()
        print("Actor网络参数已加载")

    def randomSample(self, diction, n):
        newDict = {}
        keys = random.sample(diction.keys(), n)
        for key in keys:
            newDict[key] = diction[key]
        return newDict


########################################################


rospy.init_node("learning_actor", anonymous=True)
actor = ActorNode()
rospy.spin()
