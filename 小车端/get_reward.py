import numpy
from math import *


# Reward = R_Forward + R_CAvo + R_VKeep

class Reward:
    def __init__(self, coordinate, last_coordinate, last_a, a, all_angle):
        self.rf = 0
        self.rv = 0
        self.rc1 = 0
        self.rc2 = 0
        self.rb = 0
        self.re = 0
        # 相邻两次位置信息
        self.coord = coordinate
        self.last_coord = last_coordinate
        # 小车两次速度和最新时刻的角度
        self.vu = a[0][0]
        self.last_vu = last_a[0][0]
        self.angle = a[0][1]
        # 障碍物速度
        self.v_obs = 0
        # 总的偏转角度
        self.total_angle = all_angle * (180 / pi)
        # 两次的时间
        self.time = coordinate[0][1]
        self.last_time = last_coordinate[0][1]
        # 小车加速度
        self.au = (self.vu - self.last_vu) / (self.time - self.last_time)

    def get_rf(self):
        goal = 0
        angle = 40
        xie = 0.295
        xie1 = 0.32
        xie2 = 0.26
        xie3 = 0.3
        # 计算车辆正向性奖励
        car1 = self.coord[0][0][0]
        car2 = self.coord[0][0][1]
        car3 = self.coord[0][0][2]
        car4 = self.coord[0][0][3]

        # 四个轮胎
        # 右转
        # print('total_angle', self.total_angle)
        if self.total_angle <= 0:
            # print('右转')
            total_angle = - self.total_angle
            # 右上
            if self.total_angle + angle < 90:
                ru = [car1[0] + sin((angle + total_angle) * (pi / 180)) * xie,
                      car1[1] + cos((angle + total_angle) * (pi / 180)) * xie]
            else:
                ru = [car1[0] + sin((total_angle - 50) * (pi / 180)) * xie,
                      car1[1] - cos((total_angle - 50) * (pi / 180)) * xie]
            # 左上
            if self.total_angle < angle:
                lu = [car1[0] - sin((angle - total_angle) * (pi / 180)) * xie1,
                      car1[1] + cos((angle - total_angle) * (pi / 180)) * xie1]
            else:
                lu = [car1[0] + sin((total_angle - angle) * (pi / 180)) * xie1,
                      car1[1] + cos((total_angle - angle) * (pi / 180)) * xie1]

            # 右下
            if self.total_angle < angle:
                rd = [car1[0] + xie2 * sin((angle - total_angle) * (pi / 180)),
                      car1[1] - xie2 * cos((angle - total_angle) * (pi / 180))]
            else:
                rd = [car1[0] - xie2 * sin((total_angle - angle) * (pi / 180)),
                      car1[1] - xie2 * cos((total_angle - angle) * (pi / 180))]

            # 左下
            if self.total_angle + angle < 90:
                ld = [car1[0] - xie3 * sin((angle + total_angle) * (pi / 180)),
                      car1[1] - xie3 * cos((angle + total_angle) * (pi / 180))]
            else:
                ld = [car1[0] - xie3 * cos((total_angle - 50) * (pi / 180)),
                      car1[1] + xie3 * sin((total_angle - 50) * (pi / 180))]
        # 左转
        else:
            # print('左转')
            total_angle = self.total_angle
            # 右上
            if self.total_angle < angle:
                ru = [car1[0] + sin((angle - total_angle) * (pi / 180)) * xie,
                      car1[1] + cos((angle - total_angle) * (pi / 180)) * xie]
            else:
                ru = [car1[0] - sin((total_angle - angle) * (pi / 180)) * xie,
                      car1[1] + cos((total_angle - angle) * (pi / 180)) * xie]
            # 左上
            if self.total_angle + angle < 90:
                lu = [car1[0] - sin((angle + total_angle) * (pi / 180)) * xie1,
                      car1[1] + cos((angle + total_angle) * (pi / 180)) * xie1]
            else:
                lu = [car1[0] - cos((total_angle - 50) * (pi / 180)) * xie1,
                      car1[1] - sin((total_angle - 50) * (pi / 180)) * xie1]

            # 右下
            if self.total_angle + angle < 90:
                rd = [car1[0] + xie2 * sin((total_angle + angle) * (pi / 180)),
                      car1[1] - xie2 * cos((total_angle + angle) * (pi / 180))]
            else:
                rd = [car1[0] + xie2 * cos((total_angle - 50) * (pi / 180)),
                      car1[1] + xie2 * sin((total_angle - 50) * (pi / 180))]

            # 左下
            if self.total_angle < angle:
                ld = [car1[0] - xie3 * sin((angle - total_angle) * (pi / 180)),
                      car1[1] - xie3 * cos((angle - total_angle) * (pi / 180))]
            else:
                ld = [car1[0] + xie3 * sin((total_angle - angle) * (pi / 180)),
                      car1[1] - xie3 * cos((total_angle - angle) * (pi / 180))]

        # distance5,6,7代表和其他车之间的距离
        obs = [0.17027012237122907, 0.8706936682868397, 0.08]
        distance1 = ((car1[0] - obs[0]) ** 2 + (car1[1] - obs[1]) ** 2) ** 0.5
        distance2 = ((car1[0] - car2[0]) ** 2 + (car1[1] - car2[1]) ** 2) ** 0.5
        distance3 = ((car1[0] - car3[0]) ** 2 + (car1[1] - car3[1]) ** 2) ** 0.5
        distance4 = ((car1[0] - car4[0]) ** 2 + (car1[1] - car4[1]) ** 2) ** 0.5
        # 碰撞奖励
        print('车2', distance2)
        print('车3', distance3)
        print('车4', distance4)
        if min(distance2, distance3, distance4) < 0.55:
            self.rc1 = -1
            goal = 1
        elif 0.5 < min(distance2, distance3, distance4) < 0.9:
            self.rc1 = 0.5 * (min(distance2, distance3, distance4) / 0.9)

        if distance1 < 0.35:
            self.rc2 = -1
            goal = 1
        elif 0.35 < distance1 < 0.7:
            self.rc2 = 0.5 * (distance1 / 0.7)

        # 边界奖励, x轴
        b_l = -1.22
        b_r = 1.22
        if ru[0] > b_r or lu[0] < b_l:
            self.rf = -1
            goal = 1

        # 沿车道中心线行走
        k2 = 0.5
        if b_l < lu[0] < ru[0] < b_r:
            if -1.22 < car1[0] < -0.61:
                self.rb = k2 * (1 - (abs(car1[0] + 0.61) / 0.61))
            elif -0.61 < car1[0] < 0:
                self.rb = k2 * (1 - (abs(car1[0] + 0.305) / 0.305))
            elif 0 < car1[0] < 0.61:
                self.rb = k2 * (1 - (abs(car1[0] - 0.305) / 0.305))
            elif 0.61 < car1[0] < 1.22:
                self.rb = k2 * (1 - (abs(car1[0] - 0.61) / 0.61))
        else:
            self.rb = 0

        # 到达终点
        if car1[1] >= 1.8:
            self.re = 1
            goal = 1
        elif car1[1] >= 1.6 and 0.61 < car1[0] < 1.22:
            self.re = 1
            goal = 1
        else:
            self.re = 0

        return self.rf, self.rc1, self.rb, goal, self.rc2, self.re

    # def get_rv(self):
    #     # 计算车辆速度保持奖励
    #     # goal 是否达到目的地
    #     # goal = 0
    #     # 该道路的最大速度
    #     vmax = 1
    #     # w自定义
    #     w = 5
    #     if self.vu < vmax:
    #         self.rv = w * self.vu
    #     elif self.vu > vmax:
    #         self.rv = -0.5
    #     return self.rv
    #
    # def get_rc(self):
    #     # 避障奖励
    #     # 小车反应时间,自定义
    #     react_time = 0.01
    #     # 自定义
    #     k = 1
    #     # 最小安全距离
    #     d_safe = 0
    #     # 折损因子
    #     k1 = 0.1
    #
    #     # 车与障碍物之间的距离
    #     if not self.coord:
    #         loc_user_e = self.last_coord[0][0][0]
    #     else:
    #         loc_user_e = self.coord[0][0][0]
    #
    #     loc_obs = [0.26977618581712104, 0.7077182808255185]
    #     distance = ((loc_obs[0] - loc_user_e[0]) ** 2 + (loc_obs[1] + 0.3 - loc_user_e[1]) ** 2) ** 0.5
    #
    #     # 理论上的安全距离
    #     losd = (((self.vu * cos(self.total_angle)) ** 2) / (2 * self.au * cos(self.total_angle))) + (
    #                 self.v_obs * react_time) + d_safe
    #     print('加速度', self.au)
    #     print('losd', losd)
    #     v_hengxiang = self.vu * sin(self.total_angle)
    #     print('total_angle', self.total_angle)
    #     print('横向速度', v_hengxiang)
    #     lasd = v_hengxiang ** 2 / self.au * sin(self.total_angle)
    #     print('lasd', lasd)
    #     safe_distance = (((losd ** 2) + (lasd ** 2)) ** 1 / 2)
    #     print(distance, safe_distance)
    #     if distance - 0.3 < safe_distance:
    #         self.rc = - k * (safe_distance - distance) / safe_distance
    #     if distance == 0:
    #         self.rc = -0.5
    #     else:
    #         self.rc = 0.1
    #     return self.rc

    def reward_sum(self):
        rf, rc1, rb, g, rc2, re = self.get_rf()
        # print('rc1', rc1)
        # print('rc2', rc2)
        # print('rf', rf)
        # print('rb', rb)
        # print('re', re)
        # print('总奖励',rc+rf+rb)
        return [rc1 + rf + rb + rc2 + re], g
