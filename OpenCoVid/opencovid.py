import os
import cv2

class OpenCoVid:
    def sub_image(self, img, x,y,w,h):
        return img[y:y+h, x:x+w].copy()

    def load_image(self,f_path):
        if not os.path.isfile(f_path):
            raise Exception("img file not found")
        return cv2.imread(f_path)

    def yolo_to_x_y(self,x_center, y_center, x_width, y_height, width, height):
        x_center *= width
        y_center *= height
        x_width *= width
        y_height *= height
        x_width /= 2.0
        y_height /= 2.0
        return int(x_center - x_width), int(y_center - y_height), int(x_center + x_width), int(y_center + y_height)

    def get_data_pairs_yolo(self,img_path,label_f_path):
        # validate and read data from file
        if not os.path.isfile(label_f_path):
            raise Exception("label file not found")
        img = self.load_image(img_path)

        img_h = img.shape[0]
        img_w = img.shape[1]

        labeled_objects = []
        with open(label_f_path) as f:
            content = f.readlines()
        for line in content:
            values_str = line.split()
            class_index, x_center, y_center, x_width, y_height = map(float, values_str)
            class_index = int(class_index)
            x1, y1, x2, y2 = self.yolo_to_x_y(x_center, y_center, x_width, y_height, img_w, img_h)
            labeled_objects.append((class_index,self.sub_image(img,x1,y1,abs(x2 - x1),abs(y2 - y1))))

        return labeled_objects

print("cv version: " + cv2.__version__)

i_path = "_111550872_gettyimages-1128162568.jpg"
l_path = "_111550872_gettyimages-1128162568.txt"

ocv = OpenCoVid()
labeled_objects = ocv.get_data_pairs_yolo(i_path,l_path)
for pair in labeled_objects:
    label, img = pair
    print()

print()