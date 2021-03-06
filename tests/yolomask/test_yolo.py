import unittest
import cv2
import lib.config
import sys,os
sys.path.append(os.path.realpath('../yolomask'))
from yolomask.mask_inference import YoloMask
from yolomask.person_inference import YoloPerson

class TestYolo(unittest.TestCase):
    def setUp(self, log_file='test_yolo.log'):
        lib.config.initialize()

    def test_yoloperson(self):
        with self.assertRaises(FileNotFoundError) as _:
            self.assertRaises(FileNotFoundError, YoloPerson(weights='notfound'))

        # init weights
        yoloperson = YoloPerson(weights='../yolomask/weights/yolov5s.pt')
        self.assertIsNotNone(yoloperson)

        self.assertEqual(yoloperson.classes, [0])

        class Frame: img = cv2.imread('yolomask/test.jpg')
        frame = Frame()
        yoloperson.detect(frame=frame)
        self.assertEqual(15, len(frame.persons))
        with self.assertRaises(StopIteration) as _:
            yoloperson.detect(frame=None)

        yoloperson.conf_thres = 0.90
        yoloperson.detect(frame=frame)
        self.assertEqual(0, len(frame.persons))


    def test_yolomask(self):
        with self.assertRaises(FileNotFoundError) as _:
            self.assertRaises(FileNotFoundError, YoloMask(weights='notfound'))

        # init weights
        yolomask = YoloMask(weights='../yolomask/weights/yolomask.pt')
        self.assertIsNotNone(yolomask)

        self.assertEqual(yolomask.classes, [0, 1])

        class Frame: img = cv2.imread('yolomask/test.jpg')

        frame = Frame()
        yolomask.detect(frame=frame)
        self.assertEqual(6, len(frame.masks))
        with self.assertRaises(StopIteration) as _:
            yolomask.detect(frame=None)

        yolomask.conf_thres = 0.90
        yolomask.detect(frame=frame)
        self.assertEqual(0, len(frame.masks))



def get_unit_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestYolo))
    return suite


def get_integration_test_suite():
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestYolo)
    unittest.TextTestRunner(verbosity=2).run(suite)