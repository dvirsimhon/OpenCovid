from lib.config import *
import lib.config as globals
import random


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def display_analyze(frame):
    """
    Callback Method That is passed to the 'OpenCoVid' Object
    This method will visualize the analyzed information on the given frame and will be able to save the result
    """
    #TODO ZeroDivisionError
    w_x, w_y, w_w, w_h = cv2.getWindowImageRect(globals.project)
    font_scale = frame.img.shape[1] / w_w
    frame_cpy = frame.img.copy()
    bottom_left_corner_of_text = (10, frame.img.shape[0] - 10)
    right_bottom_position = (round(frame.img.shape[1]) - 150, frame.img.shape[0] - 10)
    cv2.rectangle(frame.img, (0, frame.img.shape[0] - 50), (int(font_scale * 1e4), frame.img.shape[0]), (0, 0, 0),
                  cv2.FILLED)

    frame.img = cv2.addWeighted(frame.img, alpha, frame_cpy, 1 - alpha, gamma=0)
    frame.mask_on_count = 0
    frame.mask_off_count = 0

    if not hasattr(frame, 'violations'):
        frame.violations = 0

    if hasattr(frame, 'masks'):
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
            if label == 'mask':
                frame.mask_on_count += 1
            else:
                frame.mask_off_count += 1
            label = "{} {}%".format(label, round(confidence * 100))
            plot_one_box(bbox, frame.img, label=label, color=color, line_thickness=line_thickness)

    # Draw Mask Counter info

    text = "no-masks: {} ; masks: {} ; distance violations: {}".format(frame.mask_off_count,
                                                                         frame.mask_on_count, frame.violations)
    cv2.putText(frame.img, text, bottom_left_corner_of_text, cv2.FONT_ITALIC, round(font_scale / 2), stats_color, 2)

    try:
        ratio = frame.mask_off_count / (frame.mask_on_count + frame.mask_off_count)
    except ZeroDivisionError:
        ratio = 1

    if ratio >= 0.51 and frame.mask_off_count >= 3:
        text = "Danger!"
        cv2.putText(
            frame.img, text, right_bottom_position, font, (font_scale * 0.5), danger_color,
            int(2 * font_scale))

    elif ratio != 0:
        text = "Warning!"
        cv2.putText(
            frame.img, text, right_bottom_position, font, (font_scale * 0.5), warning_color,
            int(2 * font_scale))
    else:
        text = "Safe"
        cv2.putText(frame.img, text, right_bottom_position,
                    font, (font_scale * 0.5), safe_color, int(2 * font_scale))

    # Display img
    cv2.imshow(globals.project, frame.img)
    pressed_key = cv2.waitKey(globals.rate)
    if pressed_key == ord('q'):  # stop and close app
        print(shutdown_ascii)
        sys.exit(1)
