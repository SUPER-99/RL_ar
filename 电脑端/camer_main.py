#! coding=utf-8
import ctypes
import socket
import pickle
import threading
import time
import cv2
import cv2.aruco as aruco
from camera_aruco import detectTarget, judgeWarning, estimateCameraPose, parameterPrepare, ThreadedCamera
import multiprocessing


class camera():
    def __init__(self):
        self.label1 = 4
        self.label2 = 4
        self.label3 = 4
        self.label4 = 4

    def cameraAruco(self, locationInfo, lserver):
        target1Point, target2Point = [], []

        mtx, dist, rMatrix, tvec, refMarkerArray, targetMarker = parameterPrepare()

        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        vc.set(cv2.CAP_PROP_FPS, 30)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        Local = []
        tcpPort1 = ('10.42.0.11', 7777)  # 小车11IP
        tcpPort2 = ('10.42.0.12', 7777)  # 小车12IP
        tcpPort4 = ('10.42.0.14', 7777)  # 小车14IP

        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.connect(tcpPort1)
        s2.connect(tcpPort2)
        s4.connect(tcpPort4)
        lserver.settimeout(0.05)
        while True:
            # 获取小车停止还是运行的信息
            try:
                receive_data, client = lserver.recvfrom(5)
                print('receive_data', len(receive_data))
                if len(receive_data) == 1:
                    self.label1 = int.from_bytes(receive_data, 'little')
                elif len(receive_data) == 2:
                    self.label2 = int.from_bytes(receive_data, 'little')
                elif len(receive_data) == 3:
                    self.label3 = int.from_bytes(receive_data, 'little')
                elif len(receive_data) == 4:
                    self.label4 = int.from_bytes(receive_data, 'little')
            except socket.timeout:
                print('...')

            rval, frame = vc.read()
            start_time = time.time()
            if frame is not None:
                # 1. 估計camera pose
                # 1.1 detect aruco markers
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                parameters = aruco.DetectorParameters_create()

                corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=parameters)

                img = frame.copy()
                aruco.drawDetectedMarkers(img, corners, ids)
                cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('detect', 1280, 720)
                cv2.imshow("detect", img)
                # 一毫秒刷新一次
                cv2.waitKey(1)

                # 1.2 estimate camera pose
                gotCameraPose, rMatrixTemp, tvecTemp = estimateCameraPose(mtx, dist, refMarkerArray, corners, ids)

                # 1.3 update R, T to static value
                if gotCameraPose:
                    rMatrix = rMatrixTemp
                    tvec = tvecTemp

                # 2. 根據目標的marker來計算世界坐標系坐標
                locationInfo = detectTarget(mtx, dist, rMatrix, tvec, targetMarker, corners, ids)
                Local.append([locationInfo, start_time, self.label1, self.label2, self.label3, self.label4])
                print(Local)

                print('发送')
                data = str(Local).encode("utf-8")
                S1 = multiprocessing.Process(target=self.send(data, s1))
                S2 = multiprocessing.Process(target=self.send(data, s2))
                S4 = multiprocessing.Process(target=self.send(data, s4))
                S1.start()
                S2.start()
                S4.start()
                Local = []

    def send(self, data, s):
        s.send(data)
        msg = s.recv(1024)
        if msg:
            print('该次传输结束')

    def main(self, label_server):
        locationInfo = multiprocessing.Manager().Value(ctypes.c_char_p, '00-9.9')
        p = multiprocessing.Process(target=self.cameraAruco, args=(locationInfo, label_server))
        p.start()


if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("10.42.0.157", 7777))
    ca = camera()
    ca.main(server)

