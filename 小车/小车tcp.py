#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import json
import sys

import rospy
from multiprocessing import Process, Manager
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import socket
import time
import threading
from socket import *
import torch
import os
import multiprocessing

net_path = "/home/wheeltec/wheeltec_robot/src/wang/scripts/dict"
TCP_Recv_Port = ("10.42.0.12", 9999)  # 小车
TCP_Port = ("10.42.0.207", 8888)  # PC


class ListenSend:
    def __init__(self):
        self.train_sub = rospy.Subscriber("/train/transmissionready", String, self.train_callback)
        self.train_pub = rospy.Publisher("/train/optimizeready", String, queue_size=10)

        self.store_net_path = net_path
        self.count = 0

        # TCP接收
        self.TCP_Recv_Socket = socket(AF_INET, SOCK_STREAM)
        self.TCP_Recv_Socket.bind(TCP_Recv_Port)
        self.TCP_Recv_Socket.listen(128)
        # self.TCP_Socket = socket(AF_INET, SOCK_STREAM)

    def train_callback(self, data):
        print('触发')
        p1 = multiprocessing.Process(self.sendinfo(data))
        p2 = multiprocessing.Process(self.recvdict())
        p1.start()
        p2.start()

    def sendinfo(self, data):
        # TCP发送
        # info仍然保留字节串的形式
        print(1111111111111111)
        with open(data.data, 'rb') as f:
            info = f.read()
        TCP_Socket = socket(AF_INET, SOCK_STREAM)

        # 连接PC
        TCP_Socket.connect(TCP_Port)
        # 发送数据
        print('TCP开始发送数据')
        segSum = len(info) // 10240
        for index in range(segSum):
            TCP_Socket.send(info[index * 10240:10240 * (1 + index)])
        TCP_Socket.send(info[segSum * 10240:])
        print('发送长度为', len(info))
        TCP_Socket.close()
    
    def recvdict(self):
        # 接收数据
        store_net_path = "{}/{}.dict".format(self.store_net_path, "actorDict1")
        while True:
            print('TCP开始接收数据')
            client, addr = self.TCP_Recv_Socket.accept()
            print('ssssss')
            d = client.recv(5)
            data = pickle.loads(d)
            if data == 2:
                self.tcp_recv_file(client, store_net_path)
                self.train_pub.publish(store_net_path)
                print("神经网络已保存")
                break
            else:
                break
                
        
    def tcp_recv_file(self, client, store_net_path):
        fp = open(store_net_path, 'wb')
        allData = b''
        time_start = time.time()
        while True:
            data = client.recv(10240000)
            fp.write(data)
            allData += data
            if not data:
                print('数据接收完毕，总字节数为', len(allData), "用时为", time.time() - time_start)
                client.close()
                fp.close()
                break


rospy.init_node("learning_tcp", anonymous=True)
listenSend = ListenSend()
rospy.spin()
