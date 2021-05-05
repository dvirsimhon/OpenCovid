import os
import numpy as np
import csv
import cv2
from OpenCovid.lib.opencovid import OpenCoVid
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

main_dataset_folder_path = "D:\\University\\FourthYear\\Final Project\\Program\\DetectPersons\\detect_people\\dataset"

def load_dataset(dataset_folder_path):

    data_split_to_batch = []
    data = []
    data.append([])

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

            data[0].append(img_info)
            batch_data.append(img_info)

        data_split_to_batch.append(batch_data)

    return data, data_split_to_batch


class Evaluator:
    def __init__(self,name):
        self.eval_name = name

    def get_name(self):
        return self.eval_name

    def set_current_img(self, img, file_name, img_unique_people):
        pass

    def eval_pair(self, c1, c2):
        return None


class EuclidEvaluator(Evaluator):
    def __init__(self):
        super(EuclidEvaluator,self).__init__("Euclidean Estimator")

    def set_current_img(self, img, file_name, img_unique_people):
        self.img = img

    def eval_pair(self, c1, c2):
        c1_x, c1_y = c1
        c2_x, c2_y = c2

        euclid_dist = np.sqrt(np.square(float(c1_x) - float(c2_x)) + np.square(float(c1_y) - float(c2_y)))

        return euclid_dist


class OpenCovidEvaluator(Evaluator):
    def __init__(self, opencovid_object=OpenCoVid()):
        super(OpenCovidEvaluator, self).__init__("OpenCovid Estimator")
        self.oco = opencovid_object
        self.current_frame = None

    def set_current_img(self, img, file_name, img_unique_people):
        self.current_frame = self.oco.apply_pipeline_on_img(img)
        # base on frame.persons BBOX and img_unique_people create dict from centroid -> BBox

    def eval_pair(self, c1, c2):
        if self.current_frame is None:
            return None

        # check if both centroids are known in frame (else return None)
        # base on frame.persons BBOX and img_unique_people

        # return their estimated distance

        return 0.0#self.current_frame.dist[BBox1][BBox2]

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

def eval_results(y_pred_by_eval,y_true,error_legal_margin=0.0):

    result = {}

    for evaluator in y_pred_by_eval.keys():

        result[evaluator] = {}

        y_pred = y_pred_by_eval[evaluator]

        clean_y_pred, clean_y_true = clean_result(y_pred, y_true)

        y_diff = np.subtract(clean_y_true, clean_y_pred)
        y_diff[np.abs(y_diff) <= error_legal_margin] = 0.0 # apply error safe margin

        result[evaluator]["Data"] = np.abs(y_diff)
        result[evaluator]["MSE"] = np.square(y_diff).mean()
        result[evaluator]["Accuracy"] = len(y_diff[y_diff == 0.0]) / len(y_diff)

    # Compare Results
    print("=" * 2," Test Results With Leagal Error Margin = {} ".format(error_legal_margin),"=" * 2)

    print("-" * 20)
    print("Accuracy Comparison:")
    for evaluator in result.keys():
        print("{} = {}".format(evaluator.get_name(),result[evaluator]["Accuracy"]))
    print("-" * 20)
    print("-" * 20)
    print("MSE Comparison:")
    for evaluator in result.keys():
        print("{} = {}".format(evaluator.get_name(), result[evaluator]["MSE"]))
    print("-" * 20)
    print("=" * 47)

    data_box_plot = []
    names = []
    for evaluator in result.keys():
        data_box_plot.append(result[evaluator]["Data"])
        names.append(evaluator.get_name())

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_box_plot, patch_artist=True,notch='True', vert=0)

    # changing color and linewidth of
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#8B008B',
                    linewidth=1.5,
                    linestyle=":")

    # changing color and linewidth of
    # caps
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)

    # changing color and linewidth of
    # medians
    for median in bp['medians']:
        median.set(color='red',
                   linewidth=3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)

    # x-axis labels
    ax.set_yticklabels(names)

    # Adding title
    plt.title("Distance Error (cm) Estimator Comparison, Error Margin = {}".format(error_legal_margin))

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    plt.show()




def test_estimators(data,evaluators=[EuclidEvaluator(),OpenCovidEvaluator()], margins=[0.0,30.0,50.0,150.0,400.0],verbose=False):

    # Eval
    y_pred_by_eval = {}
    for batch in range(len(data)):
        for i in range(len(data[batch])):
            file_name, img, img_unique_people, X, y_true = data[batch][i]

            for evaluator in evaluators:
                y_pred = []
                evaluator.set_current_img(img=img,file_name=file_name, img_unique_people=img_unique_people)

                for (c1, c2) in X:
                    dist = evaluator.eval_pair(c1,c2)
                    y_pred.append(dist)

                y_pred_by_eval[evaluator] = y_pred

    for margin in margins:
        eval_results(y_pred_by_eval,y_true,margin)

    # all_mse = np.zeros(len(data))
    # for i in range(len(data)):
    #     file_name, img, img_unique_people, X, y_true = data[i]
    #     y_pred = []
    #
    #     for (c1, c2) in X:
    #         c1_x, c1_y = c1
    #         c2_x, c2_y = c2
    #
    #         euclid_dist = np.sqrt(np.square(float(c1_x)-float(c2_x))+np.square(float(c1_y)-float(c2_y)))
    #         y_pred.append(euclid_dist)
    #
    #     y_true = np.asarray(y_true)
    #     y_pred = np.asarray(y_pred)
    #
    #     print("Img {}, MSE (format: [margin=mse]) = [".format(file_name),end=' ')
    #     for j in range(len(margins)):
    #         margin = margins[j]
    #
    #         MSE = eval_img_data(y_pred,y_true,margin)
    #         all_mse[i] = MSE
    #
    #         end = ']\n' if j == len(margins) -1 else ', '
    #         print("{}={}".format(margin,round(MSE,4)),end=end)
    #
    # print("Total MSE = {} [min={}, avg={}, max={}]".format(round(all_mse.sum(),4),round(all_mse.min(),4),round(all_mse.mean(),4),round(all_mse.max(),4)))


data, data_split_to_batch = load_dataset(main_dataset_folder_path)

test_estimators(data)
print()
print("=" * 50)
print()
# test_estimators(data_split_to_batch)