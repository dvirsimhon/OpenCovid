import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from random import randrange
import math
import itertools
import cv2
from PIL import Image
from distance import pixel_meter


class SocialDistance:
    def calculate_coord(self, bbox, width, height):
        xmin = bbox[0] * width
        ymin = bbox[1] * height
        xmax = bbox[2] * width
        ymax = bbox[3] * height

        return [xmin, ymin, xmax - xmin, ymax - ymin]


    def calculate_centr(self, coord):
        return (coord[0]+(coord[2]/2), coord[1]+(coord[3]/2))


    def calculate_centr_distances(self, centroid_1, centroid_2):
        return math.sqrt((centroid_2[0]-centroid_1[0])**2 + (centroid_2[1]-centroid_1[1])**2)


    def calculate_perm(self, centroids):
        permutations = []
        for current_permutation in itertools.permutations(centroids, 2):
            if current_permutation[::-1] not in permutations:
                permutations.append(current_permutation)
        return permutations


    def midpoint(self, p1, p2):
        return ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)


    def calculate_slope(self, x1, y1, x2, y2):
        if x2-x1 != 0:
            m = (y2-y1)/(x2-x1)
            return m
        return 0

    def detect(self, frame):

        if not frame.persons:
            return

        # Get width and height
        width, height = 1, 1

        # Pixel per meters
        # In this case, we are considering that 180px approximately is 1 meter
        average_px_meter = pixel_meter.convert(frame)

        # Calculate normalized coordinates for boxes
        centroids = []
        coordinates = []
        for box, conf in frame.persons:
            coord = self.calculate_coord(box, width, height)
            centr = self.calculate_centr(coord)
            centroids.append(centr)
            coordinates.append(coord)

        # Calculate all permutations
        permutations = self.calculate_perm(centroids)

        # Display boxes and centroids
        fig, ax = plt.subplots(figsize=(40, 24), dpi=90, frameon=False)
        plt.axis('off')
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for coord, centr in zip(coordinates, centroids):
            ax.add_patch(patches.Rectangle(
                (coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='y', facecolor='none', zorder=10))
            ax.add_patch(patches.Circle(
                (centr[0], centr[1]), 3, color='yellow', zorder=20))

        # Display lines between centroids
        for perm in permutations:
            dist = self.calculate_centr_distances(perm[0], perm[1])
            dist_m = dist/average_px_meter

            # print("M meters: ", dist_m)
            middle = self.midpoint(perm[0], perm[1])
            # print("Middle point", middle)

            x1 = perm[0][0]
            x2 = perm[1][0]
            y1 = perm[0][1]
            y2 = perm[1][1]

            slope = self.calculate_slope(x1, y1, x2, y2)
            dy = math.sqrt(3**2/(slope**2+1))
            dx = -slope*dy

            # Display randomly the position of our distance text
            if randrange(10) % 2 == 0:
                Dx = middle[0] - dx*10
                Dy = middle[1] - dy*10
            else:
                Dx = middle[0] + dx*10
                Dy = middle[1] + dy*10

            if dist_m < 2.0:
                ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='white', xytext=(Dx, Dy), fontsize=10, arrowprops=dict(
                    arrowstyle='->', lw=1.5, color='yellow'), bbox=dict(facecolor='red', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]),
                        linewidth=2, color='crimson', zorder=15)
            elif 2.0 < dist_m < 3.5:
                ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='black', xytext=(Dx, Dy), fontsize=6, arrowprops=dict(
                    arrowstyle='->', lw=1, color='skyblue'), bbox=dict(facecolor='y', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]),
                        linewidth=0.5, color='skyblue', zorder=15)
            else:
                pass

        ax.imshow(frame.img, interpolation='nearest')
        # This allows you to show the inference
        plt.show()

        ## This allows you to save each frame in a folder
        # fig.savefig("TEST_.png", bbox_inches='tight', pad_inches=0)

        ## Convert figure to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.array(fig.canvas.get_renderer()._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame.img = img
        # cv2.waitKey(0)
        # plt.show()

if __name__ == '__main__':

    boxes = [
        [2014, 2107, 2214, 2498],
        [909, 1403, 1078, 1853],
        [1072, 1428, 1197, 1844],
        [1086, 1904, 1257, 2362]
    ]

    boxes2 = [
        [1928, 1582, 2049, 1930],
        [308, 2268, 573, 2804],
        [772, 2201, 985, 2737],
        [2314, 1923, 2439, 2298],
        [128, 2392, 406, 2886],
        [2726, 2243, 3002, 2577],
        [912, 1967, 1130, 2487],
        [494, 1893, 680, 2149],
        [68, 2118, 329, 2453],
        [2204, 1877, 2314, 2236],
        [1195, 1885, 1345, 2377],
        [1751, 1883, 1926, 2327],
        [2815, 1587, 2948, 1947]
    ]

    image_path = '/home/serfati/Social-Distance-Using-TensorFlow-API-Object/d5.jpg'
    frame = Image.open(image_path)
    #detect(frame)