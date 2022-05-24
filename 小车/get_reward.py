import numpy
from math import *


# Reward = R_Forward + R_CAvo + R_VKeep

class Reward:
    def __init__(self, coordinate, last_coordinate, last_a, a, all_angle):
        self.rf = 0
        self.rv = 0
        self.rc = 0
        self.rb = 0
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
        xie = 0.25
        # 计算车辆正向性奖励
        loc_user_e = self.coord[0][0][0]
        other_car = self.coord[0][0][1]
        '''
        if not self.coord:
            loc_user_e = self.last_coord[0][0][0]
        else:
            loc_user_e = self.coord[0][0][0]
        '''
        # 四个轮胎
        # 右转
        # print('total_angle', self.total_angle)
        if self.total_angle <= 0:
            # print('右转')
            xie1 = 0.3
            total_angle = - self.total_angle
            # 右上
            if self.total_angle + angle < 90:
                ru = [loc_user_e[0] + sin((angle + total_angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((angle + total_angle) * (pi / 180)) * xie]
            else:
                ru = [loc_user_e[0] + sin((total_angle - 50) * (pi / 180)) * xie,
                      loc_user_e[1] - cos((total_angle - 50) * (pi / 180)) * xie]
            # 左上
            if self.total_angle < angle:
                lu = [loc_user_e[0] - sin((angle - total_angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((angle - total_angle) * (pi / 180)) * xie]
            else:
                lu = [loc_user_e[0] + sin((total_angle - angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((total_angle - angle) * (pi / 180)) * xie]

            # 右下
            if self.total_angle < angle:
                rb = [loc_user_e[0] + xie1 * sin((angle - total_angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((angle - total_angle) * (pi / 180))]
            else:
                rb = [loc_user_e[0] - xie1 * sin((total_angle - angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((total_angle - angle) * (pi / 180))]

            # 左下
            if self.total_angle + angle < 90:
                lb = [loc_user_e[0] - xie1 * sin((angle + total_angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((angle + total_angle) * (pi / 180))]
            else:
                lb = [loc_user_e[0] - xie1 * cos((total_angle - 50) * (pi / 180)),
                      loc_user_e[1] + xie1 * sin((total_angle - 50) * (pi / 180))]
        # 左转
        else:
            # print('左转')
            xie1 = 0.3
            total_angle = self.total_angle                      
          # 右上
            if self.total_angle < angle:
                ru = [loc_user_e[0] + sin((angle - total_angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((angle - total_angle) * (pi / 180)) * xie]
            else:
                ru = [loc_user_e[0] - sin((total_angle - angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((total_angle - angle) * (pi / 180)) * xie]
            # 左上
            if self.total_angle + angle < 90:
                lu = [loc_user_e[0] - sin((angle + total_angle) * (pi / 180)) * xie,
                      loc_user_e[1] + cos((angle + total_angle) * (pi / 180)) * xie]
            else:
                lu = [loc_user_e[0] - cos((total_angle - 50) * (pi / 180)) * xie,
                      loc_user_e[1] - sin((total_angle - 50) * (pi / 180)) * xie]

            # 右下
            if self.total_angle + angle < 90:
                rb = [loc_user_e[0] + xie1 * sin((total_angle + angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((total_angle + angle) * (pi / 180))]
            else:
                rb = [loc_user_e[0] + xie1 * cos((total_angle - 50) * (pi / 180)),
                      loc_user_e[1] + xie1 * sin((total_angle - 50) * (pi / 180))]

            # 左下
            if self.total_angle < angle:
                lb = [loc_user_e[0] - xie1 * sin((angle - total_angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((angle - total_angle) * (pi / 180))]
            else:
                lb = [loc_user_e[0] + xie1 * sin((total_angle - angle) * (pi / 180)),
                      loc_user_e[1] - xie1 * cos((total_angle - angle) * (pi / 180))]

        # 障碍物位置
        loc_obs = [0.3233132809043666, 0.6688269915604095]
        distance1 = ((loc_obs[0] - ru[0]) ** 2 + (loc_obs[1] - ru[1]) ** 2) ** 0.5
        distance2 = ((loc_obs[0] - lu[0]) ** 2 + (loc_obs[1] - lu[1]) ** 2) ** 0.5
        distance3 = ((loc_obs[0] - rb[0]) ** 2 + (loc_obs[1] - rb[1]) ** 2) ** 0.5
        distance4 = ((loc_obs[0] - lb[0]) ** 2 + (loc_obs[1] - lb[1]) ** 2) ** 0.5
        distance5 = ((loc_obs[0] - other_car[0]) ** 2 + (loc_obs[1] - other_car[1]) ** 2) ** 0.5
        # print(distance1, distance2)
        # 碰撞奖励
        if distance1 <= 0.2 or distance2 <= 0.2 or distance3 <= 0.2 or distance4 <= 0.2:
            self.rc = -1
            goal = 1
        else:
            self.rc = 0

        # 边界奖励, x轴
        b_l = -0.6
        b_r = 0.6
        # print('obs', loc_obs)
        # print('angle', total_angle)
        # print('car', loc_user_e)
        # print('ru', ru)
        # print('lu', lu)
        # print('rb', rb)
        # print('lb', lb)
        if rb[0] > b_r or ru[0] > b_r or lb[0] < b_l or lu[0] < b_l:
            self.rf = -1
        else:
            self.rf = 0.1

        # 沿车道中心线行走
        k2 = 0.5
        if b_l < lu[0] < ru[0] < b_r:
            if loc_user_e[0] < 0:
                self.rb = k2 * (1 - (abs(loc_user_e[0] + 0.3) / 0.3))
            elif 0 < loc_user_e[0]:
                self.rb = k2 * (1 - (abs(loc_user_e[0] - 0.3) / 0.3))
        else:
            self.rb = 0

        return self.rf, self.rc, self.rb, goal

    def get_rv(self):
        # 计算车辆速度保持奖励
        # goal 是否达到目的地
        # goal = 0
        # 该道路的最大速度
        vmax = 1
        # w自定义
        w = 5
        if self.vu < vmax:
            self.rv = w * self.vu
        elif self.vu > vmax:
            self.rv = -0.5
        return self.rv

    def get_rc(self):
        # 避障奖励
        # 小车反应时间,自定义
        react_time = 0.01
        # 自定义
        k = 1
        # 最小安全距离
        d_safe = 0
        # 折损因子
        k1 = 0.1

        # 车与障碍物之间的距离
        if not self.coord:
            loc_user_e = self.last_coord[0][0][0]
        else:
            loc_user_e = self.coord[0][0][0]
        
        loc_obs = [0.26977618581712104, 0.7077182808255185]
        distance = ((loc_obs[0] - loc_user_e[0]) ** 2 + (loc_obs[1] + 0.3 - loc_user_e[1]) ** 2) ** 0.5

        # 理论上的安全距离
        losd = (((self.vu * cos(self.total_angle)) ** 2) / (2 * self.au * cos(self.total_angle))) + (self.v_obs * react_time) + d_safe
        print('加速度', self.au)
        print('losd', losd)
        v_hengxiang = self.vu * sin(self.total_angle)
        print('total_angle', self.total_angle)
        print('横向速度', v_hengxiang)
        lasd = v_hengxiang ** 2 / self.au * sin(self.total_angle)
        print('lasd', lasd)
        safe_distance = (((losd ** 2) + (lasd ** 2)) ** 1 / 2)
        print(distance, safe_distance)
        if distance - 0.3 < safe_distance:
            self.rc = - k * (safe_distance - distance) / safe_distance
        if distance == 0:
            self.rc = -0.5
        else:
            self.rc = 0.1
        return self.rc

    def reward_sum(self):
        rv = self.get_rv()
        rf, rc, rb, g= self.get_rf()
        print('rc', rc)
        # print('rv', rv)
        print('rf', rf)
        print('rb', rb)
        return [rc+rf+rb], g





