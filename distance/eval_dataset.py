import os
import numpy as np
import csv, sys
import cv2

import matplotlib.pyplot as plt
sys.path.insert(0, '..')
from lib.config import initialize
sys.path.insert(0, '../../yolomask/')


# sys.path.insert(0, 'yolomask/')
from yolomask.mask_inference import YoloMask
from yolomask.person_inference import YoloPerson
from distance.social_distance import SocialDistance

initialize()

from OpenCovid.lib.opencovid import OpenCoVid

main_dataset_folder_path = "D:\\University\\FourthYear\\Final Project\\Program\\DetectPersons\\detect_people\\dataset"

def load_dataset(dataset_folder_path):

    data = []

    for dataset_batch_folder in os.listdir(dataset_folder_path):

        batch_folder_path = os.path.join(dataset_folder_path,dataset_batch_folder)

        batch_img_folder_path = os.path.join(batch_folder_path,'imgs')
        batch_lbl_folder_path = os.path.join(batch_folder_path, 'labels')

        batch_data = []

        for lbl_file in os.listdir(batch_lbl_folder_path):

            file_name = lbl_file[:len(lbl_file)-len(".csv")]
            lbl_file_path = os.path.join(batch_lbl_folder_path,lbl_file)
            img_file_path = os.path.join(batch_img_folder_path,file_name) + ".jpg"

            img = cv2.imread(img_file_path)
            img_samples_X = []
            img_samples_y = []
            img_unique_people = set()

            with open(lbl_file_path, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')

                for row in reader:

                    c1 = (row[0], row[1])
                    c2 = (row[2], row[3])
                    dist = float(row[4])

                    img_unique_people.add(c1)
                    img_unique_people.add(c2)
                    img_samples_X.append((c1,c2))
                    img_samples_y.append(dist)

            img_info = (file_name, img, img_unique_people, img_samples_X, img_samples_y)

            batch_data.append(img_info)

        data.append(batch_data)

    return data


class Evaluator:
    def __init__(self,name):
        self.eval_name = name

    def get_name(self):
        return self.eval_name

    def set_current_info(self, info):
        pass

    def eval_pair(self, c1, c2):
        return None


class EuclidEvaluator(Evaluator):
    def __init__(self):
        super(EuclidEvaluator,self).__init__("Euclidean Distance")

    def set_current_info(self, info):
        file_name, img, img_unique_people, X, y_true = info
        self.img = img

    def eval_pair(self, c1, c2):
        c1_x, c1_y = c1
        c2_x, c2_y = c2

        euclid_dist = np.sqrt(np.square(float(c1_x) - float(c2_x)) + np.square(float(c1_y) - float(c2_y)))

        return euclid_dist

class EuclidNoiseEvaluator(Evaluator):
    def __init__(self, noise_power=200):
        super(EuclidNoiseEvaluator,self).__init__("Noisy Euclidean Distance")
        self.noise_power = noise_power

    def set_current_info(self, info):
        file_name, img, img_unique_people, X, y_true = info
        self.img = img

    def eval_pair(self, c1, c2):
        c1_x, c1_y = c1
        c2_x, c2_y = c2

        euclid_dist = np.sqrt(np.square(float(c1_x) - float(c2_x)) + np.square(float(c1_y) - float(c2_y)))

        noise = np.random.uniform(-self.noise_power,self.noise_power)

        return euclid_dist + noise

class IdealEvaluator(Evaluator):
    def __init__(self, noise_power=0):
        super(IdealEvaluator,self).__init__("Ideal")
        self.noise_power = noise_power

    def set_current_info(self, info):
        file_name, img, img_unique_people, X, y_true = info

        self.mapping = {}
        for i in range(len(X)):
            self.mapping[X[i]] = y_true[i]

    def eval_pair(self, c1, c2):
        return self.mapping[(c1, c2)] + np.random.uniform(-self.noise_power,self.noise_power)


class OpenCovidEvaluator(Evaluator):
    def __init__(self, opencovid_object=None, name="OpenCovid"):
        super(OpenCovidEvaluator, self).__init__(name)
        self.oco = opencovid_object
        self.current_frame = None

    def set_current_info(self, info):
        _, img, __, ___, ____ = info

        self.current_frame = self.oco.apply_pipeline_on_img(img)

        self.mapped_dists = {}

        for (perm, dist) in self.current_frame.dists:
            c1 = (perm[0][0],perm[0][1])
            c2 = (perm[1][0], perm[1][1])
            self.mapped_dists[(c1,c2)] = dist
            self.mapped_dists[(c2,c1)] = dist

    def is_c_in_box(self, c, box):
        (c_x,c_y) = c
        (x1,y1,x2,y2) = box
        return c_x >= x1 and c_x <= x2 and c_y >= y1 and c_y <= y2


    def eval_pair(self, c1, c2):
        if self.current_frame is None:
            return None

        # check if both centroids are known in frame (else return None)
        box_c1 = None
        box_c2 = None
        for (bbox,conf) in self.current_frame.persons:
            if box_c1 is None and self.is_c_in_box(c1,bbox):
                box_c1 = bbox
            if box_c2 is None and self.is_c_in_box(c2,bbox):
                box_c2 = bbox

            if box_c1 is not None and box_c2 is not None:
                break

        if box_c1 is not None and box_c2 is not None:
            detected_c1 = self.current_frame.mapping[box_c1]
            detected_c2 = self.current_frame.mapping[box_c2]
            key = (detected_c1,detected_c2)

            return self.mapped_dists[key] if key in self.mapped_dists else None
        else:
            return None


def eval_img_data(y_pred, y_true,error_legal_margin=0.0):

    y_diff = np.subtract(y_true, y_pred)

    # apply error safe margin
    y_diff[np.abs(y_diff) <= error_legal_margin] = 0.0

    MSE = np.square(y_diff).mean()

    return MSE

def clean_result(y_pred, y_true):
    clean_y_true = np.asarray(y_true.copy())
    clean_y_pred = np.asarray(y_pred.copy())

    # clean None vals
    clean_mask = clean_y_pred != None
    clean_y_true = clean_y_true[clean_mask]
    clean_y_pred = clean_y_pred[clean_mask]

    return clean_y_pred, clean_y_true

def eval_results(y_pred_by_eval,y_true_list,error_legal_margin=0.0):

    result = {}

    result_by_batch = []
    # re

    batch_img_idx = []
    img_names = []

    n_samples = {}

    first_eval = True

    for evaluator in y_pred_by_eval.keys():
        result[evaluator] = {}
        n_samples[evaluator] = 0

        img_diff = []
        img_mse = []
        img_acc = []
        img_perfect_estimate = []
        img_num_samples = []

        n_batch = len(y_pred_by_eval[evaluator])

        for batch in range(n_batch):
            if first_eval:
                batch_img_idx.append([])

            for img_name in y_pred_by_eval[evaluator][batch].keys():

                if first_eval:
                    img_id = len(img_names)
                    batch_img_idx[batch].append(img_id)
                    img_names.append(img_name)

                y_pred = y_pred_by_eval[evaluator][batch][img_name]
                y_true = y_true_list[evaluator][batch][img_name]

                clean_y_pred, clean_y_true = clean_result(y_pred, y_true)
                n_samples[evaluator] += len(clean_y_true)

                y_diff = np.subtract(clean_y_true, clean_y_pred)
                y_diff[np.abs(y_diff) <= error_legal_margin] = 0.0  # apply error safe margin

                # batch_mse[batch] = np.abs(y_diff)

                diff_data = y_diff# np.abs(y_diff)
                mse = np.square(y_diff).mean()
                acc = len(y_diff[y_diff == 0.0]) / len(y_diff)

                img_diff.extend(diff_data)
                img_mse.append(mse)
                img_acc.append(acc)
                img_perfect_estimate.append(len(y_diff[y_diff == 0.0]))
                img_num_samples.append(len(y_diff))


        result[evaluator]["Data"] = np.asarray(img_diff)
        result[evaluator]["MSE"] = np.asarray(img_mse)
        result[evaluator]["Accuracy"] = np.asarray(img_acc)
        result[evaluator]["#Hits"] = np.asarray(img_perfect_estimate)
        result[evaluator]["#Samples"] = np.asarray(img_num_samples)

        first_eval = False


    # Compare Results
    print("=" * 9," Test Results With Leagal Error Margin = {} ".format(error_legal_margin),"=" * 4)
    # print("Dataset Size: {}, Number of Imgs in Dataset: {}, Number of Batch (Place) in Dataset: {}".format(len(y_pred),0,0))

    print("Accuracy Comparison:")
    for evaluator in result.keys():
        print("-" * 20)
        name_est = evaluator.get_name()
        n_samples = result[evaluator]["#Samples"].sum()
        total_acc = np.round(result[evaluator]["#Hits"].sum() / n_samples,3)
        acc_per_batch = np.zeros(len(batch_img_idx))
        batch_n_sample = np.zeros(len(batch_img_idx))
        for batch in range(len(batch_img_idx)):
            for i in range(len(batch_img_idx[batch])):
                acc_per_batch[batch] += result[evaluator]["#Hits"][i]
                batch_n_sample[batch] += result[evaluator]["#Samples"][i]
        acc_per_batch /= batch_n_sample

        print("{}:\nTotal Accuracy = {}\n* Per Batch = {}\n* Per Img = {}\nTotal #Samples = {}\n* Per batch = {}\n* Per Img = {}".format(name_est,total_acc,np.round(acc_per_batch,3),np.round(result[evaluator]["Accuracy"],3),n_samples,batch_n_sample,result[evaluator]["#Samples"]))
        print("-" * 20)
    print()
    print("-" * 20)
    print("MSE Comparison:")
    for evaluator in result.keys():
        print("{} = {}".format(evaluator.get_name(), np.round(result[evaluator]["MSE"].sum(),3)))
    print("-" * 20)
    print("=" * 60)
    print()


    data_box_plot = []
    data_bar_plot = []
    names = []
    data_batch_bar_plot = []
    names_batch = []

    for batch in range(len(batch_img_idx)):
        data_batch_bar_plot.append([])
        names_batch.append("batch {}".format(batch))
        for j, evaluator in enumerate(result.keys()):
            data_batch_bar_plot[batch].append(0)
            for i in range(len(batch_img_idx[batch])):
                data_batch_bar_plot[batch][j] += (0 if result[evaluator]["MSE"][i] == 0 else np.log(result[evaluator]["MSE"][i]))
    data_batch_bar_plot = np.asarray(data_batch_bar_plot)

    for i in range(len(img_names)):
        data_bar_plot.append([])
        for evaluator in result.keys():
            data_bar_plot[i].append(0 if result[evaluator]["MSE"][i] == 0 else np.log(result[evaluator]["MSE"][i]))
    data_bar_plot = np.asarray(data_bar_plot)

    for evaluator in result.keys():
        data_box_plot.append(result[evaluator]["Data"])
        names.append(evaluator.get_name())


    # ==== Plot Box - Dist Error ================================================================
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_box_plot)
    ax.set_xticklabels(names)
    # Adding title
    plt.title("Distance Error (cm) Estimator Comparison, Error Margin = {}".format(error_legal_margin))
    # show plot
    plt.show()
    # ==========================================================================================

    # ===== Plot Img - Estimator ===============================================================
    N = len(result)

    ind = np.arange(N)
    width = 0.5

    fig = plt.figure(figsize=(10, 7))
    accum = np.zeros(N)
    for i in range(len(data_bar_plot)):
        if i == 0:
            plt.bar(ind, data_bar_plot[i], width)
        else:
            plt.bar(ind, data_bar_plot[i], width,bottom=accum)
        accum += data_bar_plot[i]
    plt.legend(img_names)

    plt.ylabel('log(MSE)')
    plt.xlabel('Estimator')
    plt.title('log MSE per estimator, Error Margin = {}'.format(error_legal_margin))
    plt.xticks(ind,names)

    plt.show()

    data_bar_plot = data_bar_plot.T
    N = data_bar_plot.shape[1]

    ind = np.arange(N)
    width = 0.5

    fig = plt.figure(figsize=(10, 7))
    accum = np.zeros(N)
    for i in range(len(data_bar_plot)):
        if i == 0:
            plt.bar(ind, data_bar_plot[i], width)
        else:
            plt.bar(ind, data_bar_plot[i], width, bottom=accum)
        accum += data_bar_plot[i]
    plt.legend(names)

    plt.ylabel('log(MSE)')
    plt.xlabel('img name')
    plt.title('log MSE per img, Error Margin = {}'.format(error_legal_margin))
    plt.xticks(ind, img_names)

    plt.show()
    # ==========================================================================================

    # ===== Plot batch - Estimator ===============================================================
    N = len(result)

    ind = np.arange(N)
    width = 0.5

    fig = plt.figure(figsize=(10, 7))
    accum = np.zeros(N)
    for i in range(len(data_batch_bar_plot)):
        if i == 0:
            plt.bar(ind, data_batch_bar_plot[i], width)
        else:
            plt.bar(ind, data_batch_bar_plot[i], width, bottom=accum)
        accum += data_batch_bar_plot[i]
    plt.legend(names_batch)

    plt.ylabel('log(MSE)')
    plt.xlabel('Estimator')
    plt.title('log MSE per estimator, Error Margin = {}'.format(error_legal_margin))
    plt.xticks(ind, names)

    plt.show()

    data_batch_bar_plot = data_batch_bar_plot.T
    N = data_batch_bar_plot.shape[1]

    ind = np.arange(N)
    width = 0.5

    fig = plt.figure(figsize=(10, 7))
    accum = np.zeros(N)
    for i in range(len(data_batch_bar_plot)):
        if i == 0:
            plt.bar(ind, data_batch_bar_plot[i], width)
        else:
            plt.bar(ind, data_batch_bar_plot[i], width, bottom=accum)
        accum += data_batch_bar_plot[i]
    plt.legend(names)

    plt.ylabel('log(MSE)')
    plt.xlabel('batch id')
    plt.title('log MSE per batch, Error Margin = {}'.format(error_legal_margin))
    plt.xticks(ind, names_batch)

    plt.show()
    # ==========================================================================================


def t_estimators(data,evaluators=[IdealEvaluator(),EuclidEvaluator(),OpenCovidEvaluator(OpenCoVid(None),"OpenCovid - 1 origin")], margins=[0.0,30.0,50.0,150.0,400.0],verbose=False):

    # Eval
    y_pred_by_eval = {}
    y_true_list = {}

    for evaluator in evaluators:
        y_pred_by_eval[evaluator] = []
        y_true_list[evaluator] = []

        for batch in range(len(data)):

            y_pred_by_eval[evaluator].append({})
            y_true_list[evaluator].append({})

            for i in range(len(data[batch])):
                file_name, img, img_unique_people, X, y_true = data[batch][i]

                y_pred = []
                evaluator.set_current_info(data[batch][i])

                for (c1, c2) in X:
                    dist = evaluator.eval_pair(c1,c2)
                    y_pred.append(dist)

                y_pred_by_eval[evaluator][batch][file_name] = y_pred
                y_true_list[evaluator][batch][file_name] = y_true

    for margin in margins:
        eval_results(y_pred_by_eval,y_true_list,margin)

data = load_dataset(main_dataset_folder_path)
init_filters = {}

init_filters["person"] = YoloPerson()
init_filters["dists"] = SocialDistance()
init_filters["masks"] = YoloMask()

evaluators=[IdealEvaluator(),EuclidEvaluator(),OpenCovidEvaluator(OpenCoVid(None,init_filters=init_filters),"OpenCovid - 1 origin")]
t_estimators(data,evaluators=evaluators)
print()
print("=" * 50)
print()
