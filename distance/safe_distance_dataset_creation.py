import cv2, os, sys
# from detect_people import people_yolo_detector
sys.path.insert(0, '../')
from lib.config import initialize
sys.path.insert(0, '../yolomask/')
from OpenCovid.yolomask.person_inference import YoloPerson
from datetime import datetime
initialize()

source_folder_path = "D:\\University\\FourthYear\\Final Project\\Program\\DetectPersons\\detect_people\\demo\\distance_db\\Data - Safe distance\\final\\batch2"
res_folder_path = "D:\\University\\FourthYear\\Final Project\\Program\\DetectPersons\\detect_people\\dataset"
in_idx = 19

WINDOW_NAME = "Create Distance DataSet"          # App Window name

n_img_per_vid = 1

display_speed = round(1000 / 60)        # App update window speed

font = cv2.FONT_HERSHEY_SIMPLEX

def drawPersons(frame,fontScale):

    # Draw Person Bbox on img
    if not hasattr(frame, 'persons'):
        return

    #fontScale *= 2
    i = 1
    for person in frame.persons:
        bbox, confidence = person
        x1, y1, x2, y2 = bbox
        # print(str(x1) + "," + str(y1) + "," + str(x2) + "," + str(y2))
        i+=1

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        start_p = (x1, y1)
        end_p = (x2, y2)

        cv2.rectangle(frame.img, start_p, end_p, (0, 0, 0), int(4 * fontScale))
        cv2.rectangle(frame.img, start_p, end_p, (255, 0, 0), int(2 * fontScale))

        txt_info = "Person - {}%".format(round(confidence * 100))

        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, (fontScale * 0.4), (0, 0, 0), int(3 * fontScale))
        cv2.putText(frame.img, txt_info, (x1, y1 - 5), font, (fontScale * 0.4), (255, 0, 0), int(2 * fontScale))
#
# def displayAnalyze(frame):
#     """
#     Callback Method That is passed to the 'OpenCoVid' Object
#     This method will visualize the analyzed information on the given frame and will be able to save the result
#     """
#     w_x, w_y, w_w, w_h = cv2.getWindowImageRect(WINDOW_NAME)
#     fontScale = frame.img.shape[1] / w_w
#
#     drawPersons(frame,fontScale)
#     drawMasks(frame,fontScale)
#     drawCounter(frame,fontScale)
#
#     # Display img
#     cv2.imshow(WINDOW_NAME, frame.img)
#
#     pressed_key = cv2.waitKey(display_speed)
#     if pressed_key == ord('q'):  # stop and close app
#         ocv.stopAnalze()

class Frame:

    def __init__(self, img):
        """ Constructor to the frame object that holds all the analyzed information and the source frame

        Parameters:
        img (ndarray): source img in numpy ndarray representation

        """
        self.img = img

def extract_img_from_vid(video_src):

    vid = cv2.VideoCapture(video_src)

    img_to_process = []

    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    capture_time = int(video_length / n_img_per_vid)
    count = 1

    ret, frame = vid.read()
    while len(img_to_process) < n_img_per_vid and ret:
        if count % capture_time == 0:
            img_to_process.append(frame)
        ret, frame = vid.read()
        count = count + 1

    vid.release()
    return img_to_process

def detect_stage(detector,frame):

    detector.detect(frame)

    min_confidence = 0.5

    frame.people_centroid = [((int(x2 - (abs(x1 - x2) / 2)), int(y2 - (abs(y1 - y2) / 2))), (x1, y1, x2, y2)) for ((x1, y1, x2, y2),confidence) in frame.persons if confidence > min_confidence]


def display_stage(frame):
    w_x, w_y, w_w, w_h = cv2.getWindowImageRect(WINDOW_NAME)
    fontScale = frame.img.shape[1] / w_w

    frame.pairs = []
    print(">> HELP (COMMANDS, keybord has to be in english!):\n* 't' (or any char not mention here) = Tag pair\n* 'd' = Discard pair\n* 'n' = skip to next img\n* 'q' = finish batch and close")
    for i in range(len(frame.people_centroid)):
        for j in range(i,len(frame.people_centroid)):
            if i != j:
                center1, bbox1,  = frame.people_centroid[i]
                center2, bbox2,  = frame.people_centroid[j]

                drawBBox(frame,bbox1, center1,bbox2, center2,fontScale)
                pressed_key = cv2.waitKey(0)
                if pressed_key == ord('q'):  # stop and close app
                    return len(frame.pairs), True
                if pressed_key == ord('n'):  # stop current frame
                    return len(frame.pairs), False
                if pressed_key == ord('d'):  # stop and close app
                    continue

                dist = float(input("Dist in Cm: "))
                print(dist)
                if isinstance(dist, float):
                    frame.pairs.append((center1,center2,dist))

    return len(frame.pairs), False

def drawBBox(frame,bbox1,center1, bbox2,center2, fontScale):

    radius = 5
    curr_img = frame.img.copy()

    x11, y11, x12, y12 = bbox1
    start_p1 = (int(x11), int(y11))
    end_p1 = (int(x12), int(y12))

    x21, y21, x22, y22 = bbox2
    start_p2 = (int(x21), int(y21))
    end_p2 = (int(x22), int(y22))

    cv2.rectangle(curr_img, start_p1, end_p1, (0, 0, 0), int(2 * fontScale))
    cv2.rectangle(curr_img, start_p1, end_p1, (255, 0, 0), int(fontScale))
    cv2.circle(curr_img,center1,radius,(0, 255, 0),int(4 * fontScale))

    cv2.rectangle(curr_img, start_p2, end_p2, (0, 0, 0), int(2 * fontScale))
    cv2.rectangle(curr_img, start_p2, end_p2, (255, 0, 0), int(fontScale))
    cv2.circle(curr_img, center2, radius, (0, 255, 0), int(4 * fontScale))

    cv2.imshow(WINDOW_NAME, curr_img)

import csv
def create_img_csv_file(frame,dest_img_folder,dest_lbl_folder,file_idx_name):
    print("Name to Save: ",file_idx_name)
    # Format: c1_x, c1_y, c2_x, c2_y, dist

    img_file_path = os.path.join(dest_img_folder,str(file_idx_name)+".jpg")
    lbl_file_path = os.path.join(dest_lbl_folder, str(file_idx_name)+".csv")

    cv2.imwrite(img_file_path, frame.img)  # save frame as JPEG file

    with open(lbl_file_path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for center1,center2,dist in frame.pairs:
            c1_x, c1_y = center1
            c2_x, c2_y = center2
            spamwriter.writerow([c1_x,c1_y,c2_x,c2_y,dist])

    # pass

def start_tag_creation_program(src_folder_path, target_folder_path, initial_idx=0, verbose = True):

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d_%Y %H_%M_%S")
    ds_name = 'dataset - ' + date_time

    target_path = os.path.join(target_folder_path,ds_name)
    target_img_folder_path = os.path.join(target_path,'imgs')
    target_label_folder_path = os.path.join(target_path, 'labels')

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(target_img_folder_path):
        os.makedirs(target_img_folder_path)
    if not os.path.exists(target_label_folder_path):
        os.makedirs(target_label_folder_path)

    # init
    exit_program = False
    current_idx = initial_idx
    n_img = 0 # add initial_idx to each name, counter serves as img generated in dataset
    n_samples = 0

    detector = YoloPerson(weights="../yolomask/weights/yolov5s.pt")


    for sample_file_name in os.listdir(src_folder_path):

        if exit_program:
            break

        sample_file_path = os.path.join(src_folder_path,sample_file_name)

        if os.path.isfile(sample_file_path):

            img = cv2.imread(sample_file_path)

            if verbose:
                print("FileName: ",sample_file_name," (",end="")

            if img is None:
                # video
                img_to_process = extract_img_from_vid(sample_file_path)
                if verbose:
                    print("video) -->",end=" ")
            else:
                img_to_process = [img]
                if verbose:
                    print("img) -->", end=" ")

            if verbose:
                print(len(img_to_process)," frames to process.")

            for curr_img in img_to_process:

                frame = Frame(curr_img)
                # detect bbox
                detect_stage(detector,frame)
                # create pairs
                curr_n_samples, exit_program = display_stage(frame)
                n_samples += curr_n_samples
                # save as format
                create_img_csv_file(frame,target_img_folder_path,target_label_folder_path,current_idx)

                n_img += 1

                if exit_program:
                    break

        current_idx += 1

    return (current_idx - initial_idx), n_img, n_samples



cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

n_files, n_img, n_samples = start_tag_creation_program(source_folder_path,res_folder_path,initial_idx=in_idx)
print("Total #Files In Original Data: ", n_files)
print("Total #Img In Dataset: ", n_img)
print("Total #samples In Dataset: ", n_samples)
cv2.destroyAllWindows()

def summary_data(db_folder,threshold_dist):

    n_imgs = 0

    n_people = 0
    max_people_in_samples = 0
    min_people_in_samples = 1000

    n_samples = 0
    max_samples = 0
    min_samples = 10000
    n_samples_close = 0

    max_dist = 0.0
    min_dist = 100000000.0
    avg_dist = 0.0

    for dataset_folder in os.listdir(db_folder):
        target_label_folder_path = os.path.join(db_folder,dataset_folder, 'labels')
        for lbl_file in os.listdir(target_label_folder_path):
            lbl_file_path = os.path.join(target_label_folder_path,lbl_file)
            with open(lbl_file_path, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

                unique_people = set()
                samples_in_file = 0
                n_imgs += 1
                for row in spamreader:
                    c1 = (row[0], row[1])
                    c2 = (row[2], row[3])
                    dist = float(row[4])

                    samples_in_file += 1
                    if dist < threshold_dist:
                        n_samples_close += 1

                    unique_people.add(c1)
                    unique_people.add(c2)

                    avg_dist += dist
                    if dist > max_dist:
                        max_dist = dist
                    if dist < min_dist:
                        min_dist = dist

                n_samples += samples_in_file
                if samples_in_file > max_samples:
                    max_samples = samples_in_file
                if samples_in_file < min_samples:
                    min_samples = samples_in_file

                n_people_in_img = len(unique_people)
                n_people += n_people_in_img
                if n_people_in_img > max_people_in_samples:
                    max_people_in_samples = n_people_in_img
                if n_people_in_img < min_people_in_samples:
                    min_people_in_samples = n_people_in_img


    return n_imgs, n_people, min_people_in_samples, n_people / n_imgs, max_people_in_samples, n_samples, n_samples_close,min_samples,n_samples / n_imgs,max_samples, min_dist, avg_dist / n_samples, max_dist

threshold_dist = 200 # 2 meters away
n_imgs, n_people, min_people_in_samples, avg_people_in_samples, max_people_in_samples, n_samples, n_samples_close, min_samples,avg_samples,max_samples, min_dist, avg_dist, max_dist = summary_data(res_folder_path,threshold_dist)
print()
print("=========>> DataSet Summary - {} Files (Imgs) <<=========".format(n_imgs))
print("* People Info: {} People in all files, per img = [min={}, avg={}, max={}]".format(n_people,min_people_in_samples,round(avg_people_in_samples,3),max_people_in_samples))
print("* Sample Info: {} pair distance samples in all files, {} samples with distance less than {}cm. samples per img = [min={}, avg={}, max={}]".format(n_samples,n_samples_close,threshold_dist,min_samples,round(avg_samples,3),max_samples))
print("* Distance Info: [min={}, avg={}, max={}] (in cm)".format(min_dist,round(avg_dist,3),max_dist))
print("=========================================================")