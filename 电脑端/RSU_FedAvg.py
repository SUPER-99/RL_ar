import socket
import time
import threading
import json
import copy
import torch
import pickle


def fedAvg(rcvMedols):  # 聚合函数
    wg = copy.deepcopy(rcvMedols[0])
    for k in rcvMedols[0].keys():
        for i in range(1, len(rcvMedols)):
            wg[k] += rcvMedols[i][k]   # 求和
        wg[k] = torch.div(wg[k], len(rcvMedols))   # 平均
    return wg


def acceptClient(client):
    data = b''
    p = client.recv(2)
    port = int.from_bytes(p, 'little')
    rcvdip = (addr[0], port)
    print('rcvdip', rcvdip)
    if rcvdip not in rcvdIP:
        rcvdIP.append((addr[0], port))
        print('rcvdIP', rcvdIP)
        while True:
            d = client.recv(10240000)
            if not d:
                realMsg = pickle.loads(data)
                rcvdMsg.append(realMsg)
                print("Add" + str(addr) + "into list")
                break
            data += d
        client.close()
    else:
        index = rcvdIP.index(rcvdip)
        while True:
            d = client.recv(10240000)
            if not d:
                realMsg = pickle.loads(data)
                rcvdMsg[index] = realMsg
                print("Add" + str(addr) + "new dict into list")
                break
            data += d
        client.close()


if __name__ == '__main__':
    # 定义通信相关参数
    rcvdMsg = []  # 记录来自IV的报文
    rcvdIP = []  # 记录IV的IP地址
    HOST = '10.42.0.157'  # 本地的地址
    PORT = 1111  # 本地TCP监听端口
    epoch = 0

    while True:  # 重复多轮执行监听-收集-处理-发回 的流程
        epoch += 1
        print(epoch)
        pickedIV = 3  # 每轮被挑选的IV数量

        # 建立TCP监听
        stcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stcp.bind((HOST, PORT))
        stcp.listen(128)
        print("ready for accpet model...")

        while len(rcvdIP) < pickedIV:
            client, addr = stcp.accept()
            print('addr', addr)
            acceptClient(client)

        rcvMedol = []  # 所有收到的模型参数
        for t in range(len(rcvdMsg)):
            rcvMedol.append(rcvdMsg[t])

        wg = fedAvg(rcvMedol)
        sndMsg = ' '  # 定义要发回的消息，即聚合后的全局模型参数
        sndMsg = wg

        realsndMsg = pickle.dumps(sndMsg)

        while True:
            for ip in range(len(rcvdIP)):
                ctcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 发送给IV
                print('fasong')
                print(rcvdIP[ip])
                ctcp.connect(rcvdIP[ip])
                ctcp.send(realsndMsg)  # 由于sndMsg跟随阶段进行变化，只需发出即可
                ctcp.close()
            break
        rcvdIP = []
