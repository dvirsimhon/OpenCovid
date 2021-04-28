import cv2
import sys
# sys.path.insert(0, 'yolomask/')

# from yolomask import mask_inference
# from frcnn.face_mask_estimator_faster_rcnn import face_mask_estimator_faster_rcnn

class Frame:

    def __init__(self, img):
        """ Constructor to the frame object that holds all the analyzed information and the source frame

        Parameters:
        img (ndarray): source img in numpy ndarray representation

        """
        self.img = img

class FrameStream:

    def __init__(self, video_src=0):
        """ Constructor to the frame Stream object that generate a stream of frames given a source

        Parameters:
        video_src (?): frame stream source path (the same argument as cv2.VideoCapture(video_src))
                        Options:
                            * 0 = default camera on the device
                            * mp4 file path

        """
        self.cap = cv2.VideoCapture(video_src)
        # check if we succeeded
        if not self.cap.isOpened():
            deviceID = video_src  # 0 = open default camera
            apiID = cv2.CAP_ANY  # 0 = autodetect default API
            self.cap.open(deviceID, apiID)

    def getStreamInfo(self,propId):
        """ Get Property information on the frame stream

        Parameters:
        propId (string): property id the same as VideoCapture.get(propId) method.

        Returns:
        ?: Property value

        """
        return self.cap.get(propId)

    def setStreamInfo(self,propId, val):
        """ Set Property information on the frame stream

        Parameters:
        propId (string): property id the same as VideoCapture.set(propId) method.
        val (?): property value to set

        """
        return self.cap.set(propId, val)

    def nextFrame(self):
        """ Get the next frame in the stream if exist one.

        Returns:
        ret (bool): is there a next frame or not
        frame (Frame): the frame object that holds the frame img

        """
        ret, f = self.cap.read()
        return ret, Frame(img=f)

class OpenCoVid:
    def __init__(self,callback, video_src=0, fps_limit=60):
        """ Constructor to the OpenCoVid Analyze pipeline object

        Parameters:
        callback (callable): callback method that takes a single parameter 'frame'
                             this method will be called at the end of the analyze pipeline.
        video_src (?): frame stream source path (the same argument as cv2.VideoCapture(video_src))
        fps_limit (int): maximum number of frames to analyze per second

        """
        self.set_frame_src(video_src)
        self.callback = callback
        self.fps_limit = fps_limit

        self.reset()

    def reset(self):
        """ Stop and reset the analyze object, this method must be called before analyzing a second time """
        # reset vars
        self.stopAnalze()
        self.f_count = 0
        self.pipeline_filters = []

        # Populate Pipeline with basic filters
        #self.add_analyze_filter(inference.YoloMask())
        # self.add_analyze_filter(face_mask_estimator_faster_rcnn())


    def set_frame_src(self, video_src):
        """ Set the Frame Stream source to the OpenCoVid Analyze pipeline object

        Parameters:
        video_src (?): frame stream source path (the same argument as cv2.VideoCapture(video_src))

        """
        self.frame_src = FrameStream(video_src)

    def add_analyze_filter(self,filter):
        """ Add a new filter to the end of the pipeline (before callback), if reset() is called the filter will be deleted

        Parameters:
        filter (class): class that implements the method 'detect(frame)' that have a 'Frame' argument.
                        this method will be called in the pipeline process

        """
        detect_op = getattr(filter, "detect", None)
        if callable(detect_op):
            self.pipeline_filters.append(filter)

    def stopAnalze(self):
        """ This Method Stop analyzing the frame stream """
        self.analyzing = False

    def apply_pipeline(self, frame):
        """ Analyze a single frame, process the frame in the pipeline and pass it to the callback

        Parameters:
        frame (Frame): the frame object that holds the frame img (in attribute frame.img)

        """
        for f in self.pipeline_filters:
            f.detect(frame)

        self.callback(frame)

        return frame

    def analyze(self):
        """ Start analyzing the frame stream frame by frame until stream is over or stop/reset method is called """
        self.analyzing = True

        while self.analyzing:
            #self.frame_src.setStreamInfo(cv2.CAP_PROP_POS_MSEC,(self.f_count * 100))  # not extract every frame, limit that one frame every second
            ret, frame = self.frame_src.nextFrame()

            if not ret: # frame src closed/no more frames
                break

            self.apply_pipeline(frame)
            self.f_count = self.f_count + 1


