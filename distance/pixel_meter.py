import math

import cv2

mouse_pressed = False
pixel_meter = -1


class DrawLineWidget(object):
    def __init__(self, size_in_cm, img):
        self.original_image = img
        self.clone = self.original_image.copy()
        cv2.namedWindow('Pixel-Meter')
        cv2.setMouseCallback('Pixel-Meter', self.extract_coordinates)
        self.dist = 0
        self.size_in_cm = size_in_cm
        self.pixel_as_cm = 0
        # List to store start/end points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        global mouse_pressed
        # Record ending (x,y) coordintes on left mouse bottom release
        if event == cv2.EVENT_LBUTTONUP:
            mouse_pressed = False
            end = len(self.image_coordinates) - 1
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[end]))
            x1 = self.image_coordinates[0][0]
            x2 = self.image_coordinates[end][0]
            y1 = self.image_coordinates[0][1]
            y2 = self.image_coordinates[end][1]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            self.dist = dist
            print('Distance: ' + str(dist))
            if self.pixel_as_cm == 0:
                self.pixel_as_cm = float(self.size_in_cm) / float(self.dist)
                print('one pixel as cm: {:0.3f}'.format(self.pixel_as_cm))

            real_distance = dist * self.pixel_as_cm
            print("Real distance: " + str(real_distance))

            # Draw line
            self.clone = self.original_image.copy()
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[end], (36, 255, 12), 2)
            cv2.imshow("Pixel-Meter", self.clone)

            pixels_in_meter = (dist / float(self.size_in_cm)) * 100
            print("Pixels in one meter: " + str(pixels_in_meter))
            global pixel_meter
            pixel_meter = pixels_in_meter

        if mouse_pressed:
            if event == cv2.EVENT_MOUSEMOVE:
                self.image_coordinates.append((x, y))
                # Draw line
                end = len(self.image_coordinates) - 1
                self.clone = self.original_image.copy()
                cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[end], (36, 255, 12), 2)
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
    if pixel_meter != -1:
        return pixel_meter
    size_in_cm = input("Enter the size of object you'll mark in cm: ")
    img = frame.img
    height, width = img.shape[:2]
    max_height = 900
    max_width = 900
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    draw_line_widget = DrawLineWidget(size_in_cm, img)

    while True:
        cv2.imshow('Pixel-Meter', draw_line_widget.show_image())
        key = cv2.waitKey(1)
        # Close program with keyboard 'q'
        if key == 27 or key == ord('q'):
            # cv2.destroyAllWindows()
            cv2.destroyWindow("Pixel-Meter")
            print(pixel_meter)
            return pixel_meter
