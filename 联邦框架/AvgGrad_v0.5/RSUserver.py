import socket
import time
import threading
import json

if __name__ == '__main__':
    #定义通信相关参数
    rcvdMsg = [] #记录来自IV的报文
    rcvdIP = [] #记录IV的IP地址
    # gridient = []
    # PI_gridient = []
    wg = [] #全局参数
    # PIg = 0
    HOST = '127.0.0.1' #本地的地址
    PORT = 8088 #本地UDP监听端口
    BUFSIZE = 4096 #报文缓冲大小

    ## 报文信息应该以元组（tuple）形式存储，其具体格式为[msgType,Data,PI],分别对应信息类型（梯度信息或是模型信息）、
    ## 数据本身（梯度数据或是模型数据）以及

    #联邦相关参数
    # TIMELIMIT = 20 #最大等待训练时间
    PILIMIT = 80 #模型表现阈值
    HASMATURE = False #当前联邦状态指示 False表示为梯度聚合状态，True表示已进入模型分享状态

    while True: #重复多轮执行监听-收集-处理-发回 的流程
        rcvNum = 0 #当前收到的报文数
        pickedIV = 3 #每轮被挑选的IV数量

        #建立UDP监听
        sudp= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        # sudp.settimeout(20)
        sudp.bind((HOST,PORT))
        print("ready for accpet grident or model...")
        
        def acceptClient(d,addr):
            if addr not in rcvdIP:
                rcvdIP.append(addr)
                realMsg = json.loads(d.decode('utf-8'))
                rcvdMsg.append(realMsg)
                print("Add" + str(addr) +"into list")

        def timeCounter():
            timeudp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            time.sleep(20)
            signal = '-1'
            sndSig = signal.encode('utf-8')
            timeudp.sendto(sndSig ,(HOST,PORT))

        timeCounter()
        while rcvNum < pickedIV:
            d,addr = sudp.recvfrom(BUFSIZE)
            if d.decode('utf-8') == '-1':
                print("waiting time out!")
                break
            rcvNum += 1
            print("recive Msg...,this is No" + " " + str(rcvNum))
            t = threading.Thread(target = acceptClient,args=(d,addr))
            t.start()
            
        for t in range(len(rcvdMsg)):
            rcvMedols = []
            rcvMedols.append(rcvdMsg[t][0])

        wg = FedAvg(rcvMedols)
        
        # for t in range(len(rcvdMsg)):
        #     if HASMATURE: #如果已经进入成熟阶段，那么就直接
        #         break
        #     if rcvdMsg[t][0] == "M" and rcvdMsg[t][2] > PILIMIT: #如果收到模型且模型表现大于初始阈值
        #         wn = rcvdMsg[t][1]
        #         PIn = rcvdMsg[t][2]
        #         HASMATURE = True
        #     else:
        #         gridient.append(rcvdMsg[t][1][0])
        #         PI_gridient.append(rcvdMsg[t][1][1])
        
        

        sndMsg = ' ' #定义要发回的消息，平均梯度或是模型信息
        # if HASMATURE: #成熟阶段，执行模型更新
        #     (wg, PIg) = FedOP.modelUpdate(wg,PIg,wn,PIn)
        #     sndMsg = wg
        # else: # 聚合阶段，执行梯度聚合
        #     avgG = FedOP.gradientUpdate(gridient,PI_gridient)
        #     sndMsg = avgG
        sndMsg = wg

        realsndMsg = json.dumps(sndMsg).encode('utf-8')
        target_port = 8089 # 指定发送到客户端的端口

        cudp = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #发送给IV
        for ip in range(len(rcvdIP)):
            cudp.sendto(realsndMsg,rcvdIP[ip])  #由于sndMsg跟随阶段进行变化，只需发出即可
        cudp.close()


