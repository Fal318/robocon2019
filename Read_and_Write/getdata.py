#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time, os
import cv2
import numpy as np
import pyrealsense2 as rs

height, width, fps = 480, 848, 60
align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

current_time = time.localtime()
dirs_path = os.getcwd()
file_name = "/d435.bag"
dirs_path += str.format('./Data/Bag-file/{0}-{1}-{2}_{3}-{4}', *current_time)
os.makedirs(dirs_path, exist_ok=True)
config.enable_record_to_file(dirs_path + file_name)

pipeline = rs.pipeline()
pipeline.start(config)
start = time.time()
try:
    while time.time() - start <= 235:
        frames = pipeline.wait_for_frames()
        align_frames = align.process(frames)
        color_frame = align_frames.get_color_frame()
        depth_frame = align_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = depth_frame.as_depth_frame()
        depth_color_frame = rs.colorizer().colorize(depth_frame)
        color_image = np.array(color_frame.get_data())
        depth_image = np.array(depth_color_frame.get_data())
        cv2.putText(color_image, str(int(time.time() - start)), (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=5)
        
        cv2.imshow("color-depth", np.vstack((color_image, depth_image)))
        if cv2.waitKey(1) & 0xFF == (ord('q') and ord('e')):
            break
finally:
    try:
        pipeline.stop()
        
        cv2.destroyAllWindows()
    except:
        pass