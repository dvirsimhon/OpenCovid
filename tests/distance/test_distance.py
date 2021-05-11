import logging
import unittest

import numpy as np
from scipy.spatial import KDTree

from distance.social_distance import SocialDistance
from lib import config


class TestDistance(unittest.TestCase):

    def setUp(self, log_file='test_distance.log'):
        logging.basicConfig(filename=log_file, level=logging.DEBUG)
        self.obj = SocialDistance()

    def test_closest_oor(self):
        lengths = np.array([80, 25, 105, 303])
        data = np.array([[28.47, 83.43], [28.45, 80.42], [28.16, 79.36], [82.29, 20.39]])
        mid = [27, 82]
        kdtree = KDTree(data)
        self.assertEqual(80, self.obj.closest_oor(mid, kdtree, lengths))
        self.assertEqual(0, self.obj.closest_oor(mid, kdtree, [0, 1, 2, 3]))
        with self.assertRaises(Exception) as _:
            self.obj.closest_oor([], kdtree, [])

    def test_calculate_centr_distances(self):
        centroid_1 = (1838.5, 2105.0)
        centroid_2 = (1700.5, 1105.0)
        self.assertAlmostEqual(1009, self.obj.calculate_centr_distances(centroid_1, centroid_2), delta=1)
        self.assertAlmostEqual(0, self.obj.calculate_centr_distances((0, 0), (0, 0)), delta=1)

    def test_midpoint(self):
        p1 = (1838.5, 2105.0)
        p2 = (1700.5, 1105.0)
        self.assertEqual((1769.5, 1605.0), self.obj.midpoint(p1, p2))
        self.assertAlmostEqual((0.0, 0.0), self.obj.midpoint((0, 0), (0, 0)), delta=1)

    def test_calculate_perm(self):
        centroids = [
            (1838.5, 2105.0),
            (838.5, 3105.0),
            (2838.5, 2105.0),
            (3838.5, 105.0),
        ]
        expected = [((1838.5, 2105.0), (838.5, 3105.0)),
                    ((1838.5, 2105.0), (2838.5, 2105.0)),
                    ((1838.5, 2105.0), (3838.5, 105.0)),
                    ((838.5, 3105.0), (2838.5, 2105.0)),
                    ((838.5, 3105.0), (3838.5, 105.0)),
                    ((2838.5, 2105.0), (3838.5, 105.0))
                    ]
        self.assertEqual(expected, self.obj.calculate_perm(centroids))
        self.assertEqual(6, len(self.obj.calculate_perm(centroids)))
        self.assertEqual(0, len(self.obj.calculate_perm([])))

    def test_calculate_slope(self):
        axis = (1838.5, 2105.0, 1700.5, 1105.0)
        self.assertAlmostEqual(7, self.obj.calculate_slope(*axis), delta=1)
        axis = (1838.5, 2105.0, 1838.5, 1105.0)
        self.assertEqual(0, self.obj.calculate_slope(*axis))
        self.assertAlmostEqual(0, self.obj.calculate_slope(0, 0, 0, 0), delta=1)

    def test_calculate_centr(self):
        coord = 1751, 1883, 175, 444
        self.assertEqual((1838.5, 2105.0), self.obj.calculate_centr(coord))
        self.assertEqual((0, 0), self.obj.calculate_centr((0, 0, 0, 0)))

    def test_calculate_coord(self):
        bbox = 1751, 1883, 1926, 2327
        self.assertEqual([1751, 1883, 175, 444], self.obj.calculate_coord(bbox, 1, 1))
        self.assertEqual([0, 0, 0, 0], self.obj.calculate_coord(bbox, 0, 0))

    def test_detect(self):
        import cv2

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

        class Frame: img = cv2.imread('test.jpg')
        frame = Frame()
        frame.persons = persons
        config.initialize()
        self.obj.start(frame=frame)
        self.obj.detect(frame=frame)
        self.assertEqual(105, len(frame.dists))
        self.assertEqual(11, len(frame.violations))
        frame.persons = []
        self.obj.detect(frame=frame)
        self.assertEqual(0, len(frame.dists))


def get_unit_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDistance))
    return suite


def get_integration_test_suite():
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDistance)
    unittest.TextTestRunner(verbosity=2).run(suite)
