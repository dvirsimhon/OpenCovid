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
        if not frame.persons: return

        # Get width and height
        width, height = 1, 1

        # Pixel per meters
        # In this case, we are considering that 180px approximately is 1 meter
        # average_px_meter = pixel_meter.convert(frame)
        average_px_meter = 180

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
        h, w, _ = frame.img.shape
        dpi = 100
        # Display boxes and centroids
        fig, ax = plt.subplots(figsize=(20, 15), dpi=dpi, frameon=False)

        # plt.axis('off')
        ax.axis('off')
        ax.margins(0, 0)
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

        plt.tight_layout(pad=0)
        ax.imshow(cv2.cvtColor(frame.img, cv2.COLOR_BGR2RGB), interpolation='nearest')

        # This allows you to show the inference
        # plt.show()

        # This allows you to save each frame in a folder
        # fig.savefig("TEST.png", bbox_inches='tight', pad_inches=0)

        # Convert figure to numpy
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.array(fig.canvas.get_renderer()._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        frame.img = img
        return


if __name__ == '__main__':
    persons = [((2714.0, 2232.0, 3009.0, 2595.0), 0.45305538177490234),
               ((468.0, 1884.0, 667.0, 2145.0), 0.48174849152565),
               ((2808.0, 1566.0, 2954.0, 1954.0), 0.5031518936157227),
               ((90.0, 2152.0, 309.0, 2386.0), 0.524933397769928),
               ((2164.0, 1859.0, 2314.0, 2274.0), 0.5871360301971436),
               ((992.0, 1345.0, 1097.0, 1638.0), 0.6563081741333008),
               ((1176.0, 1888.0, 1358.0, 2372.0), 0.6762101054191589),
               ((2719.0, 2771.0, 3379.0, 3024.0), 0.6987053751945496),
               ((2310.0, 1922.0, 2428.0, 2307.0), 0.7082932591438293),
               ((1922.0, 1584.0, 2052.0, 1941.0), 0.7286142110824585),
               ((915.0, 1985.0, 1137.0, 2488.0), 0.7876796722412109),
               ((775.0, 2215.0, 1000.0, 2742.0), 0.7930412888526917),
               ((1761.0, 1879.0, 1937.0, 2313.0), 0.8024228811264038),
               ((166.0, 2397.0, 393.0, 2894.0), 0.8230567574501038),
               ((348.0, 2274.0, 605.0, 2823.0), 0.8839704394340515)]

    class Frame:
        img = cv2.imread('ptt.jpg')
    frame = Frame()
    frame.persons = persons
    dist = SocialDistance()
    dist.detect(frame)
