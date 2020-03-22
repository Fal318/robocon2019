#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import pyrealsense2 as rs
import numpy as np
import cv2

height, width, fps = 480, 848, 60
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
name = input()
path = str.format('../Data/Bag-file/{0}/d435.bag', name)
config.enable_device_from_file(path)
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
out = cv2.VideoWriter("../Data/Video/" + name + ".mp4", fourcc, float(fps), (width, height))

pipeline.start(config)
start = time.time()
while time.time() - start <= 235:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        print('end file')
        break
    depth_color_frame = rs.colorizer().colorize(depth_frame)
    color_image = np.array(color_frame.get_data())
    depth_image = np.array(depth_color_frame.get_data())
    out.write(color_image)
    cv2.putText(color_image, str(int(time.time() - start)), (0, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=5)
    cv2.imshow("color-depth", np.hstack((color_image, depth_image)))
    if cv2.waitKey(1) & 0xFF == (ord('q') and ord('e')):
        pipeline.stop()
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
pipeline.stop()
out.release()