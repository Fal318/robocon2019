#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2

height, width, fps = 480, 848, 60
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_device_from_file(input())

threshold = rs.threshold_filter()
threshold.set_option(rs.option.filter_option.min_distance, 0.1)
threshold.set_option(rs.option.filter_option.max_distance, 12.0)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 0)
temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)
temporal.set_option(rs.option.holes_fill, 3)
hole_filling = rs.hole_filling_filter()
hole_filling.set_option(rs.option.holes_fill, 1)

pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        print('end file')
        break
    depth_frame = spatial.process(depth_frame)
    depth_frame = threshold.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)
    depth_color_frame = rs.colorizer().colorize(depth_frame)
    color_image = np.array(color_frame.get_data())
    depth_image = np.array(depth_color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    cv2.imshow("color-depth", np.hstack((color_image, depth_image)))
    if cv2.waitKey(1) & 0xFF == (ord('q') and ord('e')):
        pipeline.stop()
        cv2.destroyAllWindows()
        break