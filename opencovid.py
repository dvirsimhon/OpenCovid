import cv2
import sys
sys.path.insert(0, 'yolomask/')
from yolomask import inference


#from frcnn.face_mask_estimator_faster_rcnn import face_mask_estimator_faster_rcnn

class Frame:
    def __init__(self, img):
        self.img = img

class FrameStream:
    def __init__(self, video_src=0):
        self.cap = cv2.VideoCapture(video_src)
        # check if we succeeded
        if not self.cap.isOpened():
            deviceID = video_src  # 0 = open default camera
            apiID = cv2.CAP_ANY  # 0 = autodetect default API
            self.cap.open(deviceID, apiID)

    def getStreamInfo(self,propId):
        return self.cap.get(propId)

    def setStreamInfo(self,propId, val):
        return self.cap.set(propId, val)

    def nextFrame(self):
        # Capture frame-by-frame, return ret (if has next frame) , frame (if exists)
        ret, f = self.cap.read()
        return ret, Frame(f)

class FaceMaskEstimator:
    def detect(self, frame):
        #frame.img

        frame.masks = [ ( ( 10, 10, 30, 30 ) , 0.6 , 'mask' ),
                        ( ( 50, 50, 80, 90 ) , 0.5 , 'no_mask' ),
                        ( ( 100, 10, 120, 30 ) , 0.3 , 'mask' ),
                        ( ( 300, 30, 330, 60 ) , 0.4 , 'mask' ) ]


        pass

class OpenCoVid:
    def __init__(self,callback, video_src=0, fps=60):
        self.set_frame_src(video_src)
        self.callback = callback
        self.fps = fps
        self.reset()

    def reset(self):
        self.f_count = 0
        self.pipeline_filters = []

        #self.add_analyze_filter(FaceMaskEstimator())
        self.add_analyze_filter(inference.YoloMask())
        #self.add_analyze_filter(face_mask_estimator_faster_rcnn())

        self.stopAnalze()

    def set_frame_src(self, video_src):
        self.frame_src = FrameStream(video_src)

    def add_analyze_filter(self,filter):
        self.pipeline_filters.append(filter)

    def stopAnalze(self):
        self.analyzing = False

    def analyze(self):
        self.analyzing = True

        while self.analyzing:
            #self.frame_src.setStreamInfo(cv2.CAP_PROP_POS_MSEC,(self.f_count * 100))  # not extract every frame, limit that one frame every second
            ret, frame = self.frame_src.nextFrame()

            if not ret: # frame src closed/no more frames
                break

            for f in self.pipeline_filters:
                f.detect(frame)

            self.f_count = self.f_count + 1
            self.callback(frame)


## TEST ZONE ############################################################################################################

