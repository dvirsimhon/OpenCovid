import unittest
import sys
from yolomask.mask_inference import YoloMask
from yolomask.person_inference import YoloPerson


class TestYolo(unittest.TestCase):
    def setUp(self, log_file='test_distance.log'):
        self.yoloperson = YoloPerson()
        self.yolomask = YoloMask()

    def test_yoloperson(self):
        pass

    def test_yolomask(self):
        pass

def get_unit_test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestYolo))
    return suite


def get_integration_test_suite():
    pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestYolo)
    unittest.TextTestRunner(verbosity=2).run(suite)
