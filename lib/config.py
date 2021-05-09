import cv2
import sys


def initialize(persons_pt_opt='yolomask/weights/yolov5s.pt', masks_pt_opt='yolomask/weights/yolomask.pt',
             project_opt='OpenCoVid', rate_opt=1, source_opt=0, show_inf_opt=1):

    global persons_pt, masks_pt, project, rate, source, show_inf
    # == Model Parameters =================================================
    persons_pt = persons_pt_opt
    masks_pt = masks_pt_opt
    # ====================================================================

    # == Stream Parameters =================================================
    show_inf = show_inf_opt
    rate = rate_opt  # App update window speed in ms
    project = project_opt  # App Window name
    source = source_opt  # The Video Path to analyze
    # video_src = 'runs/vid1.mp4'  # The Video Path to analyze
    # ====================================================================


# == Visuals Parameters =================================================
mask_color = (169, 203, 145)  # Mask On Display Color
no_mask_color = (108, 108, 199)  # No Mask Display Color
stats_color = (197, 197, 197)  # Status bar Display Color
safe_color = [169, 203, 145]  # indicator
danger_color = [108, 108, 199]  # indicator
warning_color = [117, 209, 230]  # indicator
font = cv2.FONT_HERSHEY_SIMPLEX  # Display Font
line_thickness = 2  # bbox line thick
alpha = 0.4  # status bar transparent
dpi = 100  # dpi pixel in inch
# ====================================================================


# == ASCII Colors and Logos ==========================================
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
\n\n\n                                                                           
'''

analyzing_ascii = f'''{Cyan}{Bold}\n\n
 ____ __ _ ____ _    _ _ ___  _ __ _ ____
 |--| | \| |--| |___  Y   /__ | | \| |__,

 {RESET}
'''
print(logo)

shutdown_ascii = f'''{Red}{Bold}\n\n
 __            ___  __   __            
/__` |__| |  |  |  |  \ /  \ |  | |\ | 
.__/ |  | \__/  |  |__/ \__/ |/\| | \| 
                                       
'''
# ====================================================================
