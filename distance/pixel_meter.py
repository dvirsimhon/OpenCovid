import math
import numpy as np
import cv2
import lib.config as globals
from lib.config import *


mouse_pressed = False
lengths = np.array([])
temp_data = []

class DrawLineWidget(object):
    def __init__(self, img, scaling_factor):
        self.original_image = img
        self.clone = self.original_image.copy()
        self.scaling_factor = scaling_factor
        cv2.namedWindow('Pixel-Meter')
        cv2.setWindowProperty('Pixel-Meter', cv2.WND_PROP_TOPMOST, 2)  # set window always on top
        w_x, w_y, w_w, w_h = cv2.getWindowImageRect(globals.project)
        #font_scale = img.shape[1] / w_w
        font_scale = 1
        bottom_left_corner_of_text = (10, img.shape[0] - 10)
        cv2.rectangle(img, (0, img.shape[0] - 50), (int(font_scale * 1e4), img.shape[0]), (0, 0, 0),
                      cv2.FILLED)
        cv2.putText(img, "Mark object and than specify length", bottom_left_corner_of_text, cv2.FONT_ITALIC, font_scale, stats_color, 2)
        cv2.setMouseCallback('Pixel-Meter', self.extract_coordinates)
        self.dist = 0
        self.pixel_as_cm = 0
        # List to store start/end points
        self.image_coordinates = []

    def is_positive_numeric(self, str):
        """ Returns True is string is a number. """
        return str.replace('.', '', 1).isdigit() and float(str) > 0

    def extract_coordinates(self, event, x, y, flags, parameters):
        global mouse_pressed
        # Record ending (x,y) coordinates on left mouse bottom release
        if event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            end = len(self.image_coordinates) - 1
            # print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[end]))
            x1 = self.image_coordinates[0][0]
            x2 = self.image_coordinates[end][0]
            y1 = self.image_coordinates[0][1]
            y2 = self.image_coordinates[end][1]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            self.dist = dist
            # print('Distance: ' + str(dist))
            print(Cyan + "SETUP\nEnter Object of Reference size in CM (greater than 0):" + Bold + White)
            size_in_cm = input()
            while not self.is_positive_numeric(size_in_cm):
                print(Cyan + "SETUP\nInvalid size was inserted. Please enter size in CM greater than 0:" + Bold + White)
                size_in_cm = input()
            try:
                if self.pixel_as_cm == 0:
                    self.pixel_as_cm = float(size_in_cm) / float(self.dist)
                # print('one pixel as cm: {:0.3f}'.format(self.pixel_as_cm))
            except ZeroDivisionError:
                self.pixel_as_cm = 100

            # Draw line
            self.clone = self.original_image.copy()
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[end], (107, 209, 67), 2, cv2.LINE_AA)
            cv2.imshow("Pixel-Meter", self.clone)

            pixels_in_meter = (dist / float(size_in_cm)) * 100 / self.scaling_factor

            global lengths
            global temp_data

            lengths = np.append(lengths, pixels_in_meter)
            temp_data.append((x1, y1))
            print(RESET+40*'=')
            print(f'\t\t{Magenta}{pixels_in_meter:.1f}{Bold} pixel/meter{RESET}')
            print(40*'=')

        if mouse_pressed:
            if event == cv2.EVENT_MOUSEMOVE:
                self.image_coordinates.append((x, y))
                # Draw line
                end = len(self.image_coordinates) - 1
                self.clone = self.original_image.copy()
                cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[end], (67, 76, 209), 2, cv2.LINE_AA)
                cv2.imshow("Pixel-Meter", self.clone)

        # Record starting (x,y) coordinates on left mouse button click
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
            self.image_coordinates.append((x, y))
            mouse_pressed = True

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone


def convert(frame):
    # if temp_data:
    #     data = np.array([*temp_data])
    #     return lengths, data

    img = frame.img
    height, width = img.shape[:2]
    max_height = 900
    max_width = 900
    scaling_factor = 1
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    draw_line_widget = DrawLineWidget(img, scaling_factor)
    while True:
        cv2.imshow('Pixel-Meter', draw_line_widget.show_image())
        key = cv2.waitKey(10)
        # Close program with keyboard 'q'
        if key == 27 or key == ord('q'):
            cv2.destroyWindow("Pixel-Meter")
            data = np.array([*temp_data])
            return lengths, data
