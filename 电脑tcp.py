# coding=utf-8

import multiprocessing
import pickle
import socket
import threading
import time
import cv2
from critic11 import CriticNode11
from critic12 import CriticNode12
import random
import os

tcpPort = ("10.42.0.207", 8888)  # 本机
tcpSendPort11 = ("10.42.0.11", 9999)  # 小车1
tcpSendPort12 = ("10.42.0.12", 9999)  # 小车2
# info存放地址
location11 = './info11/'
location12 = './info12/'
# dict存放地址
dict_loc11 = 'dict11/actorDict.dict'
dict_loc12 = 'dict12/actorDict.dict'


def listenAndSend(CriticClass, tcp_send_port, location, dict_loc, Client, Count):
    p1 = multiprocessing.Process(target=Receive_info(location, Client))
    p2 = multiprocessing.Process(target=train_data(location, CriticClass, tcp_send_port, dict_loc, Count))
    p1.start()
    p2.start()


def Receive_info(Location, Client):
    # 读取并存放数据
    all_Data = b''
    store_time = time.time()
    while True:
        data = Client.recv(2000000)
        store_path = "{}{}.info".format(Location, store_time)
        if len(data):
            all_Data += data
        else:
            with open(store_path, 'wb') as f:
                f.write(all_Data)
            f.close()
            print('长度：', len(all_Data))
            break
    Client.close()
    print('关闭套接字')


def train_data(Location, CriticClass, tcp_send_port, dict_loc, Count):
    # 判断数据条数，并训练
    file_names = os.listdir(Location)
    file_ob_list = []

    for file_name in file_names:
        fileob = Location + '/' + file_name
        file_ob_list.append(fileob)
    print('未开始训练')
    if Count % 60 == 0 and Count != 0:
        print('开始训练')
        data = {}
        file_random = random.sample(file_ob_list, 3)
        for file_ob in file_random:
            with open(file_ob, 'rb') as f:
                d = f.read()
                da = pickle.loads(d)
                keys = da.keys()
                for key in keys:
                    data[key] = da[key]
                f.close()
        CriticClass.optimize(data)
        Send_dict(tcp_send_port, dict_loc)


def Send_dict(tcp_send_port, dict_loc):
    # 发送网络参数
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(tcp_send_port)
    fp = open(dict_loc, 'rb')
    allData = b''
    while True:
        print('开始传输')
        data = fp.read(10240000)
        allData += data
        client.send(data)
        if not len(data):
            print("传输成功", len(allData))
            fp.close()
            break
    client.close()


if __name__ == '__main__':
    # CriticClass11 = CriticNode11()
    CriticClass12 = CriticNode12()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(tcpPort)
    server.listen(128)
    count12 = 0
    count11 = 0
    while True:
        client, addr = server.accept()
        print("Connection from :", addr)
        if addr[0] == '10.42.0.187':
            print('car12')
            count12 += 1
            listenAndSend(CriticClass12, tcpSendPort12, location12, dict_loc12, client, count12)
            print('count', count12)
            client.close()
            print('ss')

        # if addr[0] == '10.42.0.187':
        #     print(addr)
        #     client.recv(2000000)
        #     client.close()


        if addr[0] == '10.42.0.11':
            print('car11')
            count11 += 1
            listenAndSend(CriticClass11, tcpSendPort11, location11, dict_loc11, client, count11)
            # car1 = multiprocessing.Process(target=listenAndSend(CriticClass11, tcpSendPort11, location11, dict_loc11))
            # car1.start()

