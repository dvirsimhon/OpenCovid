import cv2
import sys

# == Stream Parameters =================================================
WINDOW_NAME = "OpenCoVid"  # App Window name
display_speed = 1  # App update window speed in ms
video_src = 0  # The Video Path to analyze
# ====================================================================

# == Visuals Parameters =================================================
mask_color = (169, 203, 145)  # Mask On Display Color
no_mask_color = (108, 108, 199)  # No Mask Display Color
stats_color = (197, 197, 197)  # Status bar Display Color
safe_color = [169, 203, 145]
danger_color = [108, 108, 199]
warning_color = [117, 209, 230]
font = cv2.FONT_HERSHEY_SIMPLEX  # Display Font
line_thickness = 2
alpha = 0.4
# ====================================================================

