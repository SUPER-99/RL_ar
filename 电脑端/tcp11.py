# coding=utf-8

import multiprocessing
import pickle
import socket
import threading
import cv2
# 自行修改
from critic11 import CriticNode11
import random
import os
import time
import torch

# IP地址部分自行修改
# 主机的IP地址(接收info的TCP)
PCPort = ("10.42.0.157", 2222)
# 小车1的IP地址(发送网络参数的TCP)
CarPort = ("10.42.0.11", 9999)
# 联邦服务器的IP地址
Flport = ("10.42.0.157", 1111)
# 传控制命令的IP地址(定位信息的TCP)
CarPort1 = ("10.42.0.157", 7777)


class TCP11:
    # 在该类中，有三个功能
    # 一：接收来自小车的info信息，并根据接收到的数量来判断什么时候开始网络训练
    # 二：返回最新的网络模型，包括强化学习训练得出的，也包括联邦学习得出的
    # 三：向定位信息TCP发送小车控制信息，当网络在训练的时候，发送控制命令使小车停止，当网络模型传输完毕后，发送控制命令使小车运行
    def __init__(self):
        # 自行修改
        self.Critic = CriticNode11()
        # infoTCP接收
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(PCPort)
        self.server.listen(128)
        # 自行修改存放地址
        # dict存放地址
        self.dict_loc = 'dict/dict11/actorDict.dict'
        # info存放地址
        self.location = './info/info11/'
        # 经验池,存放的是每条info的地址
        self.file_list = []
        # 网络训练次数
        self.lr_count = 0

    def listen_and_send(self, client, fl):
        p1 = multiprocessing.Process(target=self.receive_info(client))
        p2 = multiprocessing.Process(target=self.net_train(fl))
        p1.start()
        p2.start()

    def receive_info(self, client):
        # 读取并保存数据
        Info_Data = b''
        store_time = time.time()
        while True:
            info_data = client.recv(2000000)
            store_path = "{}{}.info".format(self.location, store_time)
            if len(info_data):
                Info_Data += info_data
            else:
                with open(store_path, 'wb') as f:
                    f.write(Info_Data)
                f.close()
                self.memory_pool(store_path)
                break

    def memory_pool(self, path):
        # 维护经验池，超过250条就删除最开始的数据
        if len(self.file_list) <= 250:
            self.file_list.append(path)
        else:
            self.file_list.pop()
            self.file_list.append(path)
        print('经验池条数', len(self.file_list))

    def net_train(self, fl):
        # 向定位信息tcp发送小车控制信息，即当网络在训练时，小车需要停止
        cclient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 判断联邦学习是否传回新的网络参数
        if fl > 0:
            self.send_dict_car()

        # 当经验池中有了25条info地址后开始训练
        elif 25 < len(self.file_list):
            print('开始网络训练')
            stime = time.time()

            # 二进制的长度代表车辆信息，长度为1代表小车1，以此类推，c的值为3代表小车停止，为4代表小车运行
            c = int(3).to_bytes(1, 'little')
            cclient.sendto(c, CarPort1)

            train_data = {}
            # 看收敛性，reward之和
            r_sum = 0
            file_random = random.sample(self.file_list, 16)
            for file_ob in file_random:
                with open(file_ob, 'rb') as f:
                    d = f.read()
                    da = pickle.loads(d)
                    keys = da.keys()
                    for key in keys:
                        train_data[key] = da[key]
                    f.close()
            for k, v in train_data.items():
                r = v['r']
                r_sum = r[0] + r_sum
            # print('reward', r_sum/64)
            self.Critic.optimize(train_data)
            self.lr_count += 1
            self.send_dict_car()
            etime = time.time()
            print('用时', etime - stime)

        # 经验池还没达到阈值时，继续添加info信息
        else:
            c = int(4).to_bytes(1, 'little')
            cclient.sendto(c, CarPort1)

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(CarPort)
            # d:给小车发回信息，避免TCP堵塞
            d = pickle.dumps(1)
            client.send(d)
            client.close()

        # 当新的网络参数发送完毕后，需要像小车发送一个运行指令
        c = int(4).to_bytes(1, 'little')
        cclient.sendto(c, CarPort1)
        cclient.close()

        print('网络训练次数', self.lr_count)
        # 当网络进行了20次训练后，发给联邦服务器
        if self.lr_count > 20:
            self.send_dict_fl()

    def send_dict_fl(self):
        # 向联邦服务器发送网络参数
        fl_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        fl_client.connect(Flport)
        actor_dict = torch.load(self.dict_loc)
        while True:
            print('发送网络到FL')
            # 此处的p是为了标识ip端口，方便联邦学习返回优化后的网络参数
            p = int(2222).to_bytes(2, 'little')
            fl_client.send(p)
            dict_data = pickle.dumps(actor_dict)
            print('网络大小', len(dict_data))
            fl_client.send(dict_data)
            break

    def send_dict_car(self):
        # 向小车发送网络参数
        car_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        car_client.connect(CarPort)
        a_dict = open(self.dict_loc, 'rb')
        Dict_data = b''
        while True:
            print('发送网络到小车')
            d = pickle.dumps(2)
            car_client.send(d)
            dict_data = a_dict.read(10240000)
            Dict_data += dict_data
            car_client.send(dict_data)
            if not len(dict_data):
                print('传输成功，网络大小', len(Dict_data))
                a_dict.close()
                break
        car_client.close()


if __name__ == '__main__':
    # 类名称自行修改
    tcp = TCP11()
    # 联邦学习标志
    Fl = 0
    while True:
        print('listening...')
        tclient, addr = tcp.server.accept()
        print("Connection from :", addr)

        # 来自联邦服务器的TCP连接请求，接收网络参数
        if addr[0] == '10.42.0.157':
            all = b''
            while True:
                rcvData = tclient.recv(102400)
                if len(rcvData):
                    all += rcvData
                else:
                    print('接收FL的网络，大小', len(all))
                    data = pickle.loads(all)
                    torch.save(data, tcp.dict_loc)
                    break
            Fl += 1
            tclient.close()

        # 来自小车端的TCP连接请求，接收info
        if addr[0] == '10.42.0.86':
            print('flcount', Fl)
            tcp.listen_and_send(tclient, Fl)
            Fl = 0
            tclient.close()
