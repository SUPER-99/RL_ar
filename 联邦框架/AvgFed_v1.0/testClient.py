import socket
import json

if __name__ == '__main__':
    HOST = '127.0.0.1'
    PORT = 8088
    RSU_addr = (HOST,PORT)

    #注意：
    MsgW = (state.dict()) #待发送的模型参数
    #这里向RSU发送的数据应该为Actor的网络参数，具体格式待补充
    MsgSend = json.dumps(MsgW).encode('utf-8')

    udp_client = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    udp_client.sendto(MsgSend,RSU_addr)
    rcvData = udp_client.recvfrom(4096)
    print(rcvData)
    udp_client.close()
