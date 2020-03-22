#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import time
import math
import serial
import struct
import numpy as np
import setting


class Switch_play_zone:
    if setting.zone.lower() == "red":
        coefficient_val = 1
        color = (0, 0, 255)
        lower_limit = [np.array([170, 155, 25]), np.array([0, 200, 25])]
        upper_limit = [np.array([180, 255, 255]), np.array([2, 255, 255])]
    elif setting.zone.lower() == "blue":
        coefficient_val = -1
        color = (255, 0, 0)
        lower_limit = [np.array([70, 155, 60])] * 2
        upper_limit = [np.array([105, 255, 255])] * 2
    else:
        sys.exit("Please check setting file!")


class Cal:
    R2 = 2**-0.5

    def coordinate_transformation(self, x, y):
        return [(y+x), (y-x)] * self.R2


class Communication:
    mode = setting.mode
    status = False

    def __init__(self):
        if self.mode == 0:
            try:
                self.device = serial.Serial("/dev/mbed", 115200)
            except serial.serialutil.SerialException:
                sys.exit("Serial Device is not conect!")
            else:
                self.status = True

    def start(self):
        if self.status:
            start = time.time()
            self.device.writable
            while not (self.device.writable()):
                pass
            execution_time = time.time() - start
            if execution_time < 3.7:
                return 3.7 - execution_time
            else:
                return 0
        else:
            return 3.7

    def write(self, val):
        if self.status and self.device.writable():
            self.device.write(struct.pack("b", val))

    def __del__(self):
        if self.status:
            self.device.close()
