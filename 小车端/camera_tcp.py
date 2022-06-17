#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from socket import *
import time
import rospy
from std_msgs.msg import String
import numpy as np

TCP_Recv_Port = ("10.42.0.11", 7777)  # 小车的ip地址


def camera_tcp():
    coord_pub = rospy.Publisher("camera_coord", String, queue_size=10)
    rospy.init_node("learningcamera_tcp", anonymous=True)
    print('启动摄像头tcp')
    tcp_recv_socket = socket(AF_INET, SOCK_STREAM)
    tcp_recv_socket.bind(TCP_Recv_Port)
    tcp_recv_socket.listen(128)
    client, addr = tcp_recv_socket.accept()
    print(type(addr))
    print("Connection from :", addr)
    while True:
        data = client.recv(1024)
        # location_data = camera_recv(client, coord_pub)
        location_data = data.decode("utf-8")
        client.send(b'finish')  # .encode("utf-8")
        # if location_data[0][0][0] is not None:
            # print('tcp', location_data)
        coord_pub.publish(location_data)
        #print('该次位置信息长度:')



if __name__ == '__main__':
    camera_tcp()
