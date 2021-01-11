import cv2
from opencovid import OpenCoVid

########################################################################################################################
#                                                                                                                      #
# Demo - How To Use 'OpenCoVid' Object, This Demo will Display the analyzed frame and will be able to save the result  #
#        as a video file                                                                                               #
#                                                                                                                      #
########################################################################################################################

# == Demo Parameters =================================================
WINDOW_NAME = "OpenCoVid Demo"          # App Window name
display_speed = round(1000 / 60)        # App update window speed

video_src = "videoplayback.mp4"         # The Video Path to analyze

mask_color = (0,255,0)                  # Mask On Display Color
no_mask_color = (0,0,255)               # No Mask Display Color

font = cv2.FONT_HERSHEY_SIMPLEX         # Display Font
# ====================================================================

def displayAnalyze(frame):
    """
    Callback Method That is passed to the 'OpenCoVid' Object
    This method will visualize the analyzed information on the given frame and will be able to save the result
    """
    w_x, w_y, w_w, w_h = cv2.getWindowImageRect(WINDOW_NAME)
    fontScale = frame.img.shape[1] / w_w

    # Draw Bbox on img
    for info in frame.masks:
        bbox, confidence, label = info
        x1, y1, x2, y2 = bbox

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

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

    pressed_key = cv2.waitKey(display_speed)
    if pressed_key == ord('q'):  # stop and close app
        ocv.stopAnalze()

class MaskCounter:
    """ This Class Is An Implementation Of A Analyze Filter that will be added to the pipeline """

    def detect(self, frame):
        """ Analyze the frame to have a statistical information on mask count

        This filter will add the given attribute to the Frame object:
        frame.mask_on_count (int): number of people with masks in the frame
        frame.mask_off_count (int): number of people with out masks in the frame

        """
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

####################################################################################################

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# === OpenCoVid Lib Use =========================
ocv = OpenCoVid(callback=displayAnalyze, video_src=video_src)

ocv.add_analyze_filter(MaskCounter())

ocv.analyze()
# ===============================================

cv2.destroyAllWindows()