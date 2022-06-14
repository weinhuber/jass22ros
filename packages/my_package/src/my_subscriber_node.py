#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from duckietown_msgs.msg import Twist2DStamped, SegmentList
from sensor_msgs.msg import CompressedImage, CameraInfo
import cv2
import numpy as np
import math

class MySubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
        # self.sub = rospy.Subscriber(f'/{os.environ["VEHICLE_NAME"]}/camera_node/image/compressed', CompressedImage, self.callback, queue_size=1)
        # self.sub4 = rospy.Subscriber(f'/{self.ACQ_DEVICE_NAME}/{self.ACQ_TOPIC_RAW}', CompressedImage, self.callback3, queue_size=1)

    def callback(self, data):
        rospy.loginfo("starting image processing")
        foo = self.process_duckie_img(data.data)
        rospy.loginfo(foo.shape)

    def process_duckie_img(self, data):
        self.CUTTING_MULT = 3 / 8
        # Min number of votes for line detection
        self.MIN_THRESHOLD = 10
        self.MIN_LINELENGTH = 8
        self.MAX_LINEGAP = 4
        cropped_edges = self.region_of_interest(self.detect_edges(data))
        lane_lines = self.average_slope_intercept(data, self.detect_line_segments(cropped_edges))
        lane_lines_image = self.display_lines(data, lane_lines)
        heading_image = self.display_heading_line(lane_lines_image, steering_angle=90)
        return heading_image

    def detect_edges(self, frame):
        # filter for blue lane lines
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow(f"Picture{index}", hsv)

        sensitivity = 70
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # cv2.imshow(f"Picture{index} white mask", mask)

        edges = cv2.Canny(mask, 200, 400)
        # cv2.imshow(f"Picture{index} canny", edges)

        return edges

    def region_of_interest(self, edges):
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # only focus bottom half of the screen
        polygon = np.array([[
            (0, height * self.CUTTING_MULT),
            (width, height * self.CUTTING_MULT),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges

    def detect_line_segments(self, cropped_edges):
        kernel1 = np.ones((3, 5), np.uint8)
        kernel2 = np.ones((9, 9), np.uint8)
        # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
        img1 = cv2.erode(cropped_edges, kernel1, iterations=1)
        img2 = cv2.dilate(img1, kernel2, iterations=3)
        img3 = cv2.bitwise_and(cropped_edges, img2)
        img3 = cv2.bitwise_not(img3)
        img4 = cv2.bitwise_and(cropped_edges, cropped_edges, mask=img3)

        rho = 1  # distance precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, self.MIN_THRESHOLD,
                                        np.array([]), minLineLength=self.MIN_LINELENGTH, maxLineGap=self.MAX_LINEGAP)
        return line_segments

    def average_slope_intercept(self, frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            print('No line_segment segments detected')
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1 / 3
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    # print('skipping vertical line segment (slope=inf): %s' % line_segment)
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

        return lane_lines

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]

    def detect_lane(self, frame):
        edges = self.detect_edges(frame)
        cropped_edges = self.region_of_interest(edges)
        line_segments = self.detect_line_segments(cropped_edges)
        lane_lines = self.average_slope_intercept(frame, line_segments)

        return lane_lines

    def display_lines(self, frame, lines, line_color=(0, 255, 0), line_width=10):
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return line_image

    def display_heading_line(self, frame, steering_angle, line_color=(0, 0, 255), line_width=5):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape

        # figure out the heading line from steering angle
        # heading line (x1,y1) is always center bottom of the screen
        # (x2, y2) requires a bit of trigonometry

        # Note: the steering angle of:
        # 0-89 degree: turn left
        # 90 degree: going straight
        # 91-180 degree: turn right
        steering_angle_radian = steering_angle / 180.0 * math.pi
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 2)

        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image

    def stabilize_steering_angle(
            self,
            curr_steering_angle,
            new_steering_angle,
            num_of_lane_lines,
            max_angle_deviation_two_lines=5,
            max_angle_deviation_one_lane=1):
        """
        Using last steering angle to stabilize the steering angle
        if new angle is too different from current angle,
        only turn by max_angle_deviation degrees
        """
        if num_of_lane_lines == 2:
            # if both lane lines detected, then we can deviate more
            max_angle_deviation = max_angle_deviation_two_lines
        else:
            # if only one lane detected, don't deviate too much
            max_angle_deviation = max_angle_deviation_one_lane

        angle_deviation = new_steering_angle - curr_steering_angle
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle
                                            + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle
        return stabilized_steering_angle


if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='my_subscriber_node')
    # keep spinning
    rospy.spin()