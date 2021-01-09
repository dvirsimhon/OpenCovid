import cv2
from opencovid import OpenCoVid

print("cv version: " + cv2.__version__)

mask_color = (0,255,0)
no_mask_color = (0,0,255)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (170,50,200)
lineType = 2

def displayAnalyze(frame):

    # Draw Bbox on img
    for info in frame.masks:
        bbox, confidence, label = info
        x1, y1, x2, y2 = bbox

        start_p = (x1,y1)
        end_p = (x2,y2)

        color = no_mask_color
        if label == 'mask':
            color = mask_color

        cv2.rectangle(frame.img,start_p,end_p,(0,0,0),2)
        cv2.rectangle(frame.img, start_p, end_p, color, 1)

        txt_info = "{} {}".format(label, confidence)

        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, 0.4, (0,0,0), 2)
        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, 0.4, color, 1)

    # Draw Mask Counter info
    bottomLeftCornerOfText = (10, frame.img.shape[0] - 10)
    text = "No-Mask Count: {}  Mask Count: {}".format(frame.mask_off_count, frame.mask_on_count)
    cv2.putText(frame.img, text,bottomLeftCornerOfText,font,fontScale,(0,0,0),5)
    cv2.putText(frame.img, text, bottomLeftCornerOfText, font, fontScale, (255,255,255), 1)

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
            if label == 'mask':
                frame.mask_on_count = frame.mask_on_count + 1
            else:
                frame.mask_off_count = frame.mask_off_count + 1


WINDOW_NAME = "OpenCoVid Demo"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

#cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

video_src = "/home/serfata/OpenCovid/yolomaski/data/videos/sda.mp4"
display_speed = round(1000 / 60)

# === OpenCoVid Lib Use =========================
ocv = OpenCoVid(callback=displayAnalyze, video_src=video_src)

ocv.add_analyze_filter(MaskCounter())

ocv.analyze()

cv2.destroyAllWindows()