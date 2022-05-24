import socket
import time
import threading
import json
import copy
import torch


def fedAvg(rcvMedols):  # 聚合函数
    wg = copy.deepcopy(rcvMedols[0])
    for k in rcvMedols.keys():     
        for i in range(1, len(rcvMedols)):
            wg[k] += rcvMedols[i][k]    #求和
        wg[k] = torch.div(wg[k], len(rcvMedols))    #平均
    return wg

if __name__ == '__main__':
    # 定义通信相关参数
    rcvdMsg = []  # 记录来自IV的报文
    rcvdIP = []  # 记录IV的IP地址
    HOST = '127.0.0.1'  # 本地的地址
    PORT = 8088  # 本地UDP监听端口
    BUFSIZE = 4096  # 报文缓冲大小

    TIME_LIMIT = 20 #每轮等待车辆的最大时延

    while True:  # 重复多轮执行监听-收集-处理-发回 的流程
        rcvNum = 0  # 当前收到的报文数
        pickedIV = 3  # 每轮被挑选的IV数量

        # 建立UDP监听
        sudp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sudp.bind((HOST, PORT))
        print("ready for accpet model...")

        def acceptClient(d, addr):
            if addr not in rcvdIP:
                rcvdIP.append(addr)
                realMsg = json.loads(d.decode('utf-8'))
                rcvdMsg.append(realMsg)
                print("Add" + str(addr) + "into list")

        def timeCounter(): #计时器
            timeudp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            time.sleep(TIME_LIMIT)
            signal = '-1'
            sndSig = signal.encode('utf-8')
            timeudp.sendto(sndSig, (HOST, PORT))

        timeCounter()
        while rcvNum < pickedIV:
            d, addr = sudp.recvfrom(BUFSIZE)
            if d.decode('utf-8') == '-1':
                print("waiting time out!")
                break
            rcvNum += 1
            print("recive Msg...,this is No" + " " + str(rcvNum))
            t = threading.Thread(target=acceptClient, args=(d, addr))
            t.start()

        rcvMedols = [] #所有收到的模型参数
        for t in range(len(rcvdMsg)):
            rcvMedols.append(rcvdMsg[t][0])

        wg = fedAvg(rcvMedols)
        sndMsg = ' '  # 定义要发回的消息，即聚合后的全局模型参数
        sndMsg = wg

        realsndMsg = json.dumps(sndMsg).encode('utf-8')
        target_port = 8089  # 指定发送到客户端的端口

        cudp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 发送给IV
        for ip in range(len(rcvdIP)):
            cudp.sendto(realsndMsg, rcvdIP[ip])  # 由于sndMsg跟随阶段进行变化，只需发出即可
        cudp.close()