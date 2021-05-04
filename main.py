import argparse
import sys
from argparse import ArgumentParser

from lib import config
from lib.plots import display_analyze
from lib.opencovid import OpenCoVid
from distance.social_distance import SocialDistance

sys.path.insert(0, 'yolomask/')
from yolomask.mask_inference import YoloMask
from yolomask.person_inference import YoloPerson

if __name__ == '__main__':
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--dists', action='store_false', help='analyze social distancing')
    parser.add_argument('--persons', action='store_false', help='analyze persons on frame')
    parser.add_argument('--masks', action='store_false', help='analyze face mask wearing')
    parser.add_argument('--show-inf', action='store_false', help='show inference on frame')
    parser.add_argument('--mask-pt', nargs='+', type=str, default='yolomask/weights/yolomask.pt', help='model.pt path(s)')
    parser.add_argument('--person-pt', nargs='+', type=str, default='yolomask/weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--rate', nargs='+', type=int, default=1, help='display rate speed in ms')
    parser.add_argument('--project', default='OpenCoVid', help='main window and project name')

    opt = parser.parse_args()
    print(opt)
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(
            ('rtsp://', 'https://'))
    config.initialize(opt.person_pt, opt.mask_pt, opt.project, opt.rate, [opt.source, 0][webcam], opt.show_inf)

    # === OpenCoVid Lib Use =========================
    ocv = OpenCoVid(callback=display_analyze, video_src=config.source)
    # filters
    if opt.persons:
        ocv.add_analyze_filter(YoloPerson())
    if opt.dists:
        ocv.add_analyze_filter(SocialDistance())
    if opt.masks:
        ocv.add_analyze_filter(YoloMask())
    # run
    ocv.analyze()
    # ===============================================
