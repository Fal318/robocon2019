#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
import function as func


count = 0
zone = func.Switch_play_zone()
cal = func.Cal()
height, width, fps = 480, 848, 60
limit_x, limit_y, limit_z = 0.025, 0.9, 0.83
pix_limit = [100, 150, 190, 233, 273]

align = rs.align(rs.stream.color)
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
"""
current_time = time.localtime()
dirs_path = os.getcwd()
file_name = "/d435.bag"
dirs_path += str.format("/Data/Bag-file/{0}-{1}-{2}_{3}-{4}", *current_time)
os.makedirs(dirs_path, exist_ok=True)
config.enable_record_to_file(dirs_path + file_name)
"""

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 1)
spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 0)
hole_filling = rs.hole_filling_filter()
hole_filling.set_option(rs.option.holes_fill, 1)

com = func.Communication()
try:
    pipeline = rs.pipeline()
    pipeline.start(config)
    time.sleep(com.start())
    start = time.time()

    while True:
        input_frame = pipeline.wait_for_frames()
        align_frames = align.process(input_frame)
        color_frame = align_frames.get_color_frame()
        depth_frame = align_frames.get_depth_frame()
        count += 1

        if not color_frame or not depth_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()
        depth_stream = depth_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.array(color_frame.get_data())
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = [
            cv2.inRange(hsv, zone.lower_limit[0], zone.upper_limit[0]),
            cv2.inRange(hsv, zone.lower_limit[1], zone.upper_limit[1]),
        ]
        mask_image = cv2.bitwise_or(mask[0], mask[1])
        frame = cv2.bitwise_or(color_image, color_image, mask=mask_image)
        contours = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        areas = list(map(cv2.contourArea, contours))
        moments = list(map(cv2.moments, contours))
        area_data = []
        for moment, area, contour in zip(moments, areas, contours):
            if moment["m00"] == 0:
                x, y = 0, 0
            else:
                x = int(moment["m10"] / moment["m00"])
                y = int(moment["m01"] / moment["m00"])
            length = depth_frame.get_distance(x, y)
            processed_area = math.sqrt(area) * length
            area_data.append([processed_area, [x, y], length, contour])
            # pix_arr, point_arr, length_arr, contour
        
        area_data = [i for i in area_data if (i[0] > pix_limit[0]) and (i[0] < pix_limit[4])]
        processed_data = []
        for d in area_data:
            arg = np.argmax(area_data, axis=0)
            if d[0] <= pix_limit[1]:
                x, y = d[1]
                processed_data.append([x, y, d[2]])
            elif processed_area[arg][0] < pix_limit[2] and processed_area[arg][0] > pix_limit[1]:
                x, y = d[1]
                contour = d[3]
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                difference = (rightmost[0] - leftmost[0]) // 4
                truth_x = [x - difference, x + difference]
                for i in range(2):
                    length = depth_frame.get_distance(x, y)
                    processed_data.append([truth_x[i], length])

        draw_image = [np.zeros((height, width, 3), np.uint8)]*4

        if len(processed_data) == 0:
            com.write(0)
        else:
            processed_data = sorted(
                processed_data, key=lambda x: abs(x[0])
            )
            x, y, length = processed_data.pop(0)
            distance, before_y, before_z = rs.rs2_deproject_pixel_to_point(
                depth_stream, [x, y], length
            )
            aftter_y, aftter_z = cal.coordinate_transformation(
                before_y, before_z)

            if abs(aftter_y) > limit_y or abs(aftter_z) > limit_z:
                distance *= int(zone.coefficient_val * 1000)
                cv2.circle(color_image, (x, y), 8, zone.color, -1)
                cv2.putText(
                    draw_image[0],
                    str(math.floor(distance)),
                    (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    10,
                    zone.color,
                    20,
                )
                cv2.putText(
                    draw_image[1],
                    str(math.floor(abs(aftter_y * 1000))),
                    (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    10,
                    zone.color,
                    20,
                )
                cv2.putText(
                    draw_image[2],
                    str(math.floor(abs(aftter_z * 1000))),
                    (10, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    10,
                    zone.color,
                    20,
                )

            if abs(distance) < limit_x:
                draw_image = np.tile(
                    np.uint8([84, 255, 159]), (height, width, 1))
                com.write(1)
            elif distance < 0:
                draw_image = np.zeros((height, width, 3), np.uint8)
                cv2.line(draw_image, (0, height),
                         (width, height), zone.color, 80)
                com.write(0)
            else:
                draw_image = np.zeros((height, width, 3), np.uint8)
                cv2.line(draw_image, (0, 0), (width, 0), zone.color, 80)
                com.write(0)
            distance *= int(zone.coefficient_val * 1000)
            cv2.circle(color_image, (x, y), 8, zone.color, -1)

            cv2.putText(
                draw_image[0],
                str(math.floor(distance)),
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                10,
                zone.color,
                20,
            )
            cv2.putText(
                draw_image[1],
                str(math.floor(abs(aftter_y * 1000))),
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                10,
                zone.color,
                20,
            )
            cv2.putText(
                draw_image[2],
                str(math.floor(abs(aftter_z * 1000))),
                (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                10,
                zone.color,
                20,
            )

        image1 = [
            np.hstack((color_image, frame)),
            np.hstack((draw_image[0], draw_image[1])),
            np.hstack((draw_image[2], draw_image[3]))
        ]

        image1 = np.hstack((color_image, frame))
        image2 = np.hstack((draw_image[0], draw_image[1]))
        image3 = np.hstack((draw_image[2], draw_image[3]))
        image = np.vstack((image1, image2))
        cv2.imshow("main", image)
        cv2.imshow("z-status", image3)

        if cv2.waitKey(1) & 0xFF == (ord("q") and ord("e")):
            break

except KeyboardInterrupt:
    cv2.destroyAllWindows()
    pipeline.stop()
    del com


else:
    try:
        cv2.destroyAllWindows()
        pipeline.stop()
        del com
    except RuntimeError:
        print(os.sys.exc_info()[0])
        print(os.sys.exc_info()[1])
        print("RealSense is not conect!")
    else:
        print(str(math.floor(count / (time.time() - start))) + "fps")
