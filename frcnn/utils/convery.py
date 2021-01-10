# parsing directly.
import os

path = "/storage/users/Ise4thYear/OpenCoVid/files/rcnn/data/train"
files = os.listdir(path)
for file in files:
    if file[-3:] == "png":
        #print(file[:-3] + "jpg")
        os.rename(path + "/" + file, path + "/" + file[:-3] + "jpg")

