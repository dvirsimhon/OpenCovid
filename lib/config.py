import cv2
import sys
# == Stream Parameters =================================================
WINDOW_NAME = "OpenCoVid"  # App Window name
display_speed = 1  # App update window speed in ms
video_src = 0  # The Video Path to analyze
# ====================================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # open main window
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)  # set window always on top

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

# == ASCII Colors =================================================
Black = '\u001b[30;1m'
Red = '\u001b[31;1m'
Green = '\u001b[32;1m'
Yellow = '\u001b[33;1m'
Blue = '\u001b[34;1m'
Magenta = '\u001b[35;1m'
Cyan = '\u001b[36;1m'
White = '\u001b[37;1m'
Bold = '\u001b[1m'
BgWhite = '\u001b[47;1m'
RESET = '\u001b[0m'


logo = f'''{Green}
____ ___  ____ _  _ ____ ____ _  _ _ ___  
|  | |__] |___ |\ | |    |  | |  | | |  \ 
|__| |    |___ | \| |___ |__|  \/  | |__/ {Red}{Bold}v1.0{RESET}

more-info: https://serfati.github.io/open-covid/                                                                            
'''

analyzing_ascii = f'''{Cyan}\n\n
 ____ __ _ ____ _    _ _ ___  _ __ _ ____
 |--| | \| |--| |___  Y   /__ | | \| |__,
 -----------------------------------------
 {RESET}
'''
print(logo)
