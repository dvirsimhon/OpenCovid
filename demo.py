import cv2
from opencovid import OpenCoVid

print("cv version: " + cv2.__version__)

mask_color = (0,255,0)
no_mask_color = (0,0,255)

font = cv2.FONT_HERSHEY_SIMPLEX
#fontScale = 0.5
fontColor = (170,50,200)
lineType = 2
from utils.plots import plot_one_box
def displayAnalyze(frame):

    w_x, w_y, w_w, w_h = cv2.getWindowImageRect(WINDOW_NAME)
    fontScale = frame.img.shape[1] / w_w

    # Draw Bbox on img
    for info in frame.masks:
        bbox, confidence, label = info
        x1, y1, x2, y2 = bbox
        print(bbox)

        x1 = int(x1)#int((x1 / 640) * frame.img.shape[1])#
        x2 = int(x2)#int((x2 / 640) * frame.img.shape[1])#
        y1 = int(y1)#int((y1 / 640) * frame.img.shape[0])#
        y2 = int(y2)#int((y2 / 640) * frame.img.shape[0])#

        start_p = (x1,y1)
        end_p = (x2,y2)

        color = no_mask_color
        if label == 0.0:
            label = 'no mask'
        elif label == 1.0:
            label = 'mask'

        if label == 'mask':
            color = mask_color

        cv2.rectangle(frame.img,start_p,end_p,(0,0,0) ,int(2 * fontScale))
        cv2.rectangle(frame.img, start_p, end_p, color, int(1 * fontScale))
        #plot_one_box(bbox, frame.img, label=label,color=color, line_thickness=1)

        txt_info = "{} {}".format(label, round(confidence, 3))

        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, (fontScale * 0.4), (0,0,0), int(2 * fontScale))
        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, (fontScale * 0.4), color, int(1 * fontScale))

    # Draw Mask Counter info
    bottomLeftCornerOfText = (10, frame.img.shape[0] - 10)
    text = "No-Mask Count: {}  Mask Count: {}".format(frame.mask_off_count, frame.mask_on_count)
    cv2.putText(frame.img, text,bottomLeftCornerOfText,font,fontScale,(0,0,0),int(5 * fontScale))
    cv2.putText(frame.img, text, bottomLeftCornerOfText, font, fontScale, (255,255,255), int(1 * fontScale))

    # Display img
    cv2.imshow(WINDOW_NAME, frame.img)
    #cv2.resizeWindow(WINDOW_NAME, frame.img.shape[1], frame.img.shape[0] + 30)

    pressed_key = cv2.waitKey(display_speed)
    if pressed_key == ord('q'):
        ocv.stopAnalze()

class MaskCounter:
    def detect(self, frame):

        frame.mask_on_count = 0
        frame.mask_off_count = 0

        for info in frame.masks:
            bbox, confidence, label = info

            if label == 0.0:
                label = 'no mask'
            elif label == 1.0:
                label = 'mask'

            if label == 'mask':
                frame.mask_on_count = frame.mask_on_count + 1
            else:
                frame.mask_off_count = frame.mask_off_count + 1


WINDOW_NAME = "OpenCoVid Demo"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#cv2.moveWindow(WINDOW_NAME, 20,20)
#cv2.resizeWindow(WINDOW_NAME, 640,640)

#cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

#video_src = "1.mp4"
video_src = "3.mp4"
#video_src = "videoplayback.mp4"

display_speed = round(1000 / 60)

# === OpenCoVid Lib Use =========================
ocv = OpenCoVid(callback=displayAnalyze, video_src=video_src)

ocv.add_analyze_filter(MaskCounter())

#ocv.analyzeImg('3.jpg')
ocv.analyze()

cv2.destroyAllWindows()