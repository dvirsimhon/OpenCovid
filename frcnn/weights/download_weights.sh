#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Usage:
#    $ bash weights/download_weights.sh

# python - <<EOF
# from utils.google_utils import attempt_download

# for x in ['m']:
#     attempt_download(f'yolov5{x}.pt')

# EOF

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZxGjMsfogaUGaWc0zuYCbOexJPbFmISv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZxGjMsfogaUGaWc0zuYCbOexJPbFmISv" -O frcnn_mask.pth && rm -rf /tmp/cookies.txt