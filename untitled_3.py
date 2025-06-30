# Untitled - By: Thomas - Sun Jun 29 2025

import sensor
import time
from pyb import UART,Pin, ExtInt
from ulab import numpy as np


Mode = 3

blob_max =None
p = 0
black = (43, 0, -128, 127, -128, 127)
red = (36, 0, 13, 127, -80, 124)

uart = UART(3, 9600)

p_out = Pin('P7', Pin.OUT_PP)
p_out.low()

date = None
roi1 = None

TS = 1/60
last_frame_location = [0 for _ in range(4)]
last_frame_rect = [0 for _ in range(4)]
x = 0 #左顶点x坐标
y = 0 #左顶点y坐标
last_frame_x = x #上一帧左顶点x坐标
last_frame_y = y #上一帧左顶点y坐标

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_auto_whitebal(False)
sensor.set_auto_gain(False)
sensor.skip_frames(time=2000)

class KalmanFilter:
    """
    卡尔曼滤波器初始化
    :param initial_state: 初始状态向量 [x, y, w, h, dx, dy]
    """
    def __init__(self, initial_state):

        self.A = np.array([
            [1, 0, 0, 0, TS, 0], # x位置 = 前次x + 速度x*Δt
            [0, 1, 0, 0, 0, TS], # y位置 = 前次y + 速度y*Δt
            [0, 0, 1, 0, 0, 0], # 宽度保持不变（假设目标尺寸稳定）
            [0, 0, 0, 1, 0, 0], # 高度保持不变
            [0, 0, 0, 0, 1, 0], # x方向速度保持不变（匀速模型）
            [0, 0, 0, 0, 0, 1] # y方向速度保持不变
            ])
        self.C = np.eye(6) # 观测矩阵（直接观测位置和尺寸）
        self.Q = np.diag([1e-6]*6) # 过程噪声（系统不确定性，值越小滤波越敏感）
        self.R = np.diag([1e-6]*6) # 观测噪声（测量误差，值越小越信任传感器数据）
        # 初始化状态
        self.x_hat = initial_state # 状态估计向量 [x, y, w, h, dx, dy]
        self.p = np.diag([10]*6) # 误差协方差矩阵（初始不确定性较大）

    """
    卡尔曼滤波更新步骤
    :param Z: 观测值向量 [x, y, w, h, dx, dy]
    :return: 更新后的状态估计
    """
    def update(self, Z):
    # ---------- 预测阶段 ----------
        x_hat_minus = np.dot(self.A, self.x_hat) # 状态预测
        p_minus = np.dot(self.A, np.dot(self.p, self.A.T)) + self.Q # 协方差预测
        # ---------- 更新阶段 ----------
        S = np.dot(self.C, np.dot(p_minus, self.C.T)) + self.R # 新息协方差
        S_inv = np.linalg.inv(S + 1e-4*np.eye(6)) # 正则化逆矩阵（防止奇异）
        K = np.dot(np.dot(p_minus, self.C.T), S_inv) # 卡尔曼增益计算
        innovation = Z - np.dot(self.C, x_hat_minus) # 测量残差
        self.x_hat = x_hat_minus + np.dot(K, innovation) # 状态修正
        self.p = np.dot((np.eye(6) - np.dot(K, self.C)), p_minus) # 协方差更新
        return self.x_hat

def callback(line):
    """
    外部中断函数
    """
    global Mode
    Mode += 1

clock = time.clock()
ext = ExtInt(Pin('P9'), ExtInt.IRQ_FALLING, Pin.PULL_UP, callback)
kf = KalmanFilter(np.array([80, 60, 30, 30, 2, 2]))
while True:
    clock.tick()
    if Mode == 0:
        img = sensor.snapshot()
        Rects = img.find_rects(threshold=20000)
        for Rect in Rects:
            img.draw_rectangle(Rect.x(),Rect.y(),Rect.w(),Rect.h())
            roi1 = Rect.rect()
            print(roi1)
    #print(Mode)
    if Mode == 1:
        img = sensor.snapshot()
        blobs = img.find_blobs(red)
        for blob in blobs:
            img.draw_cross(blob.cx(),blob.cy())
            Date = (blob.cx(),blob.cy())
            print(Date)
            uart.write(Date +'\n')


    if Mode ==2:
        img = sensor.snapshot()
        # blobs = img.find_blobs(black)
        # for blob in blobs:
        #     img.draw_rectangle(blob.rect())
        for r1 in img.find_rects(threshold=60000):
            img.draw_rectangle(r1.rect())
            roi2 = r1.rect()
            for c in r1.corners():
                img.draw_circle(c[0],c[1],5)
            for r2 in img.find_rects(roi2,threshold=50000):
                img.draw_rectangle(r2.rect())
                roi3 = r2.rect()
                if roi2[2] > (roi3[2]+10) and roi2[3] > (roi3[3]+10):
                    for c2 in r2.corners():
                        img.draw_circle(c2[0],c2[1],5,color=(0, 255, 0))
                #roi3 = Rect.rect()
        #         print(roi2,roi3)
    if Mode == 3:
        blob_max = None
        p=0
        img = sensor.snapshot()
        #blobs1 = img.find_blobs([black])
        blobs2 = img.find_blobs([red],merge=True)
        #for blob in blobs1:
            #img.draw_circle(blob.cx(),blob.cy(),(int(blob.w()/2)) )
        for blob in blobs2:
            if blob.pixels()>p:
                p = blob.pixels()
                blob_max = blob
                if blob_max:
                    img.draw_circle(blob_max.cx(),blob_max.cy(),(int(blob_max.w()/2)), color=(0, 255, 0))
                    x = blob_max.cx()
                    y = blob_max.cy()
                    w = blob_max.w()
                    h = blob_max.h()
                    dx = (x - last_frame_x) / TS
                    dy = (y - last_frame_y) / TS

                    Z = np.array([x,y,w,h,dx,dy])
                    x_hat = kf.update(Z)
                    last_frame_x, last_frame_y = x, y
                    img.draw_circle(int(x_hat[0]),int(x_hat[1]),int (x_hat[2]/2),color=(255, 0, 0))
