import itertools
import math
from random import randrange
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from lib.config import *
import lib.config as globals
from scipy.spatial import KDTree
from distance import pixel_meter

plt.rcParams.update({'figure.max_open_warning': 0})


class SocialDistance:

    @staticmethod
    def calculate_coord(bbox, width=1, height=1):
        xmin = bbox[0] * width
        ymin = bbox[1] * height
        xmax = bbox[2] * width
        ymax = bbox[3] * height

        return [xmin, ymin, xmax - xmin, ymax - ymin]

    @staticmethod
    def calculate_centr(coord):
        return coord[0] + (coord[2] / 2), coord[1] + (coord[3] / 2)

    @staticmethod
    def calculate_centr_distances(centroid_1, centroid_2):
        return math.sqrt((centroid_2[0] - centroid_1[0]) ** 2 + (centroid_2[1] - centroid_1[1]) ** 2)

    @staticmethod
    def calculate_perm(centroids):
        permutations = []
        for current_permutation in itertools.permutations(centroids, 2):
            if current_permutation[::-1] not in permutations:
                permutations.append(current_permutation)
        return permutations

    @staticmethod
    def midpoint(p1, p2):
        return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

    @staticmethod
    def calculate_slope(x1, y1, x2, y2):
        if x2 - x1 != 0:
            m = (y2 - y1) / (x2 - x1)
            return m
        return 0

    @staticmethod
    def closest_oor(mid, kdtree, lengths):
        """ KDTree """
        kdtree_q = kdtree.query(mid)  # 0-dist to self -> +1
        return lengths[kdtree_q[1]]

    def update_px_meter(self, frame):
        self.px_meter_res = pixel_meter.convert(frame)

    def detect(self, frame):
        h, w, _ = frame.img.shape
        w_x, w_y, w_w, w_h = cv2.getWindowImageRect(globals.project)

        if not frame.persons:
            try:
                fig, ax = plt.subplots(figsize=(w / w_w * 16, h / w_h * 9), dpi=100, frameon=False)
            except:
                fig, ax = plt.subplots(figsize=(40, 30), dpi=100, frameon=False)

            ax.imshow(cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB), interpolation='nearest')
            ax.annotate("No Persons Detected!", xy=(frame.img.shape[0] / 2, frame.img.shape[1] / 2), color='white',
                        xytext=(frame.img.shape[0] / 2, frame.img.shape[1] / 2.5 - 10), fontsize=15,
                        bbox=dict(facecolor='#52c4ac', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)

            ax.axis('off')
            ax.margins(0, 0)
            plt.tight_layout(pad=0)
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = np.array(fig.canvas.get_renderer()._renderer)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame.violations = 0
            frame.dists = []
            frame.mapping = []
            frame.img = img
            plt.cla()
            plt.close(fig)
            return

        # Get width and height
        width, height = 1, 1

        # Multi pixel-meter references
        px_meter_res = self.px_meter_res
        kdtree = KDTree(px_meter_res[1])

        # Calculate normalized coordinates for boxes
        centroids = []
        coordinates = []
        mapping = dict()
        for box, conf in frame.persons:
            coord = self.calculate_coord(box, width, height)
            centr = self.calculate_centr(coord)
            centroids.append(centr)
            coordinates.append(coord)
            mapping[box] = centr
        # Calculate all permutations
        permutations = self.calculate_perm(centroids)
        # Display boxes and centroids

        try:
            fig, ax = plt.subplots(figsize=(w / w_w * 16, h / w_h * 9), dpi=100, frameon=False)
        except:
            fig, ax = plt.subplots(figsize=(40, 30), dpi=100, frameon=False)

        ax.axis('off')
        ax.margins(0, 0)
        for coord, centr in zip(coordinates, centroids):
            ax.add_patch(patches.Rectangle(
                (coord[0], coord[1]), coord[2], coord[3], linewidth=2, edgecolor='y', facecolor='none', zorder=10))
            ax.add_patch(patches.Circle(
                (centr[0], centr[1]), 3, color='yellow', zorder=20))

        violations = 0
        dists = []
        # Display lines between centroids
        for perm in permutations:
            dist = self.calculate_centr_distances(perm[0], perm[1])
            middle = self.midpoint(perm[0], perm[1])

            # Multi pixel-meter references
            px_meter_val = self.closest_oor(middle, kdtree, px_meter_res[0])

            # px_meter_val = 220

            dist_m = dist / px_meter_val
            dists.append((perm, dist_m * 1e2))

            x1 = perm[0][0]
            x2 = perm[1][0]
            y1 = perm[0][1]
            y2 = perm[1][1]

            slope = self.calculate_slope(x1, y1, x2, y2)
            dy = math.sqrt(3 ** 2 / (slope ** 2 + 1))
            dx = -slope * dy

            # Display randomly the position of our distance text
            if randrange(10) % 2 == 0:
                Dx = middle[0] - dx * 10
                Dy = middle[1] - dy * 10
            else:
                Dx = middle[0] + dx * 10
                Dy = middle[1] + dy * 10

            if dist_m < 2.0:
                violations += 1
                ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='white', xytext=(Dx, Dy), fontsize=10,
                            arrowprops=dict(
                                arrowstyle='->', lw=1.5, color='yellow'),
                            bbox=dict(facecolor='red', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]),
                        linewidth=2, color='crimson', zorder=15)
            elif 2.0 < dist_m < 3.5:
                ax.annotate("{}m".format(round(dist_m, 2)), xy=middle, color='black', xytext=(Dx, Dy), fontsize=6,
                            arrowprops=dict(
                                arrowstyle='->', lw=1, color='skyblue'),
                            bbox=dict(facecolor='y', edgecolor='white', boxstyle='round', pad=0.2), zorder=30)
                ax.plot((perm[0][0], perm[1][0]), (perm[0][1], perm[1][1]),
                        linewidth=0.5, color='skyblue', zorder=15)
            else:
                pass

        plt.tight_layout(pad=0)
        ax.imshow(cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB), interpolation='nearest')

        # This allows you to save each frame in a folder
        fig.savefig("TEST.png", bbox_inches='tight', pad_inches=0)

        # Convert figure to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.array(fig.canvas.get_renderer()._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame.violations = violations
        frame.dists = dists
        frame.mapping = mapping
        frame.img = img

        print('**violations**')
        print(frame.violations)
        plt.cla()
        plt.close(fig)
        return
