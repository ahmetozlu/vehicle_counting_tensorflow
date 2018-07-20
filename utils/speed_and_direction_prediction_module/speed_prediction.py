#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------
from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
bottom_position_of_detected_vehicle = [0]


def predict_speed(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    ):
    speed = 'n.a.'  # means not available, it is just initialization
    direction = 'n.a.'  # means not available, it is just initialization
    scale_constant = 1  # manual scaling because we did not performed camera calibration
    isInROI = True  # is the object that is inside Region Of Interest
    update_csv = False

    if bottom < 250:
        scale_constant = 1  # scale_constant is used for manual scaling because we did not performed camera calibration
    elif bottom > 250 and bottom < 320:
        scale_constant = 2  # scale_constant is used for manual scaling because we did not performed camera calibration
    else:
        isInROI = False

    if len(bottom_position_of_detected_vehicle) != 0 and bottom \
        - bottom_position_of_detected_vehicle[0] > 0 and 205 \
        < bottom_position_of_detected_vehicle[0] \
        and bottom_position_of_detected_vehicle[0] < 210 \
        and roi_position < bottom:
        is_vehicle_detected.insert(0, 1)
        update_csv = True
        image_saver.save_image(crop_img)  # save detected vehicle image

    # for debugging
    # print("bottom_position_of_detected_vehicle[0]: " + str(bottom_position_of_detected_vehicle[0]))
    # print("bottom: " + str(bottom))
    if bottom > bottom_position_of_detected_vehicle[0]:
        direction = 'down'
    else:
        direction = 'up'

    if isInROI:
        pixel_length = bottom - bottom_position_of_detected_vehicle[0]
        scale_real_length = pixel_length * 44  # multiplied by 44 to convert pixel length to real length in meters (chenge 44 to get length in meters for your case)
        total_time_passed = current_frame_number - current_frame_number_list[0]
        scale_real_time_passed = total_time_passed * 24  # get the elapsed total time for a vehicle to pass through ROI area (24 = fps)
        if scale_real_time_passed != 0:
            speed = scale_real_length / scale_real_time_passed / scale_constant  # performing manual scaling because we have not performed camera calibration
            speed = speed / 6 * 40  # use reference constant to get vehicle speed prediction in kilometer unit
            current_frame_number_list.insert(0, current_frame_number)
            bottom_position_of_detected_vehicle.insert(0, bottom)

    return (direction, speed, is_vehicle_detected, update_csv)
