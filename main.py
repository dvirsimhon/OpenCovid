import sys

import cv2

sys.path.insert(0, 'yolomask/')

from yolomask.mask_inference import YoloMask
from yolomask.utils.plots import plot_one_box
from yolomask.person_inference import YoloPerson
from opencovid import OpenCoVid
from distance.social_distance import SocialDistance

########################################################################################################################
#                                                                                                                      #
# Demo - How To Use 'OpenCoVid' Object, This Demo will Display the analyzed frame                                      #
#                                                                                                                      #
########################################################################################################################

# == Demo Parameters =================================================
WINDOW_NAME = "OpenCoVid"  # App Window name
display_speed = round(1000 / 60)  # App update window speed

video_src = 0  # The Video Path to analyze

mask_color = (169, 203, 145)  # Mask On Display Color
no_mask_color = (108, 108, 199)  # No Mask Display Color

font = cv2.FONT_HERSHEY_SIMPLEX  # Display Font


# ====================================================================

def displayAnalyze(frame):
    """
    Callback Method That is passed to the 'OpenCoVid' Object
    This method will visualize the analyzed information on the given frame and will be able to save the result
    """
    w_x, w_y, w_w, w_h = cv2.getWindowImageRect(WINDOW_NAME)
    font_scale = frame.img.shape[1] / w_w
    frame_cpy = frame.img.copy()
    bottom_left_corner_of_text = (10, frame.img.shape[0] - 10)
    middle_up_position = (round(frame.img.shape[1] / 2) - 10, 20)
    cv2.rectangle(frame.img, (0, frame.img.shape[0] - 50), (int(font_scale * 1e4), frame.img.shape[0]), (0, 0, 0),
                  cv2.FILLED)
    alpha = 0.4
    frame.img = cv2.addWeighted(frame.img, alpha, frame_cpy, 1 - alpha, gamma=0)

    # Draw Bbox on img
    for info in frame.masks:
        bbox, confidence, label = info
        color = no_mask_color
        if label == 0.0:
            label = 'no mask'
        elif label == 1.0:
            label = 'mask'

        if label == 'mask':
            color = mask_color

        if not hasattr(frame, 'violations'):
            frame.violations = 0
        label = "{} {}%".format(label, round(confidence * 100))
        plot_one_box(bbox, frame.img, label=label, color=color, line_thickness=2)

    # Draw Mask Counter info

    text = "no-masks: {} ; masks: {} ; social distancing violations: {}".format(frame.mask_off_count, frame.mask_on_count, frame.violations)
    cv2.putText(frame.img, text, bottom_left_corner_of_text, cv2.FONT_ITALIC, round(font_scale / 2), (197, 197, 197), 2)

    try:
        ratio = frame.mask_off_count / (frame.mask_on_count + frame.mask_off_count)
    except:
        ratio = 1

    if ratio >= 0.51 and frame.mask_off_count >= 3:
        text = "Danger!"
        cv2.putText(
            frame.img, text, middle_up_position, cv2.FONT_HERSHEY_SIMPLEX, (font_scale * 0.5), [108, 108, 199],
            int(2 * font_scale))

    elif ratio != 0:
        text = "Warning!"
        cv2.putText(
            frame.img, text, middle_up_position, cv2.FONT_HERSHEY_SIMPLEX, (font_scale * 0.5), [117, 209, 230],
            int(2 * font_scale))
    else:
        text = "Safe"
        cv2.putText(frame.img, text, middle_up_position,
                    cv2.FONT_HERSHEY_SIMPLEX, (font_scale * 0.5), [169, 203, 145], int(2 * font_scale))

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

ocv.add_analyze_filter(YoloPerson())
ocv.add_analyze_filter(SocialDistance())
ocv.add_analyze_filter(YoloMask())
ocv.add_analyze_filter(MaskCounter())

ocv.analyze()
# ===============================================

cv2.destroyAllWindows()
