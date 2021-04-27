import cv2
import numpy as np

class YoloPerson:
    """
    Class for detecting bounding box of persons in img with YOLOv3
    """
    def __init__(self,
                 weights_file='yolo-coco/yolov3.weights',
                 cfg_file='yolo-coco/yolov3.cfg',
                 min_confidence=0.5,
                 threshold=0.3):
        """
        Constructor
        :param weights_file: yolo weights file path, default weights trained on COCO DB will be used
        :param cfg_file: yolo configuration file path, default cfg will be YOLOv3
        :param min_confidence: minimum confidence of the model to allow bounding boxes, default will be 0.5
        :param threshold: threshold for non-maxima suppression, default will be 0.3
        """
        self.PEOPLE_LABEL = 0

        self.confidence = min_confidence
        self.threshold = threshold

        self.yolo_net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

        self.ln = self.yolo_net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

    def load_frame(self, frame):
        """
        load frame to the dnn model and prepare for prediction
        :param frame: given frame input for the dnn model
        :return: true if the model is ready for prediction, false otherwise
        """
        valid = True

        try:
            blob = cv2.dnn.blobFromImage(frame.img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            self.yolo_net.setInput(blob)
        except:
            valid = False

        return valid

    def detect(self, frame=None):
        """
        Predict BBox of person object in a given frame and update the frame with the analyzed information
        :param frame: a given frame with img attribute
        :return:
            update frame object with attribute frame.persons:
            list of tuples = [ ( ( start_x, start_y, end_x, end_y ) , confidence ) ]
        """
        # validate input
        valid = self.load_frame(frame)

        if valid:
            # init
            frame.persons = []
            W = frame.img.shape[1]
            H = frame.img.shape[0]
            boxes = []
            confidences = []

            # predict
            layer_outputs = self.yolo_net.forward(self.ln)

            for output in layer_outputs:
                for detection in output:
                    # extract the class ID and probability of the current detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # only allow Person Class Detection with high confidence
                    if classID == self.PEOPLE_LABEL and confidence > self.confidence:
                        # convert to the dimensions of the given frame and from Yolo format
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # add to the pre-suppression temp lists
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))

            # apply non-maxima suppression to suppress weak, overlapping bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,self.threshold)

            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    x1 = boxes[i][0]
                    y1 = boxes[i][1]
                    x2 = x1 + boxes[i][2] # bbox width
                    y2 = y1 + boxes[i][3] # bbox height
                    # add to frame
                    info = ((x1, y1, x2, y2),confidences[i])
                    frame.persons.append(info)

        return frame
