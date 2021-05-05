import unittest

from OpenCovid.tests.yolomask.test_detect import DetectTestCase

def get_unit_test_suite():
    suite = unittest.TestSuite()

    suite.addTest(DetectTestCase('test_detect'))

    return suite

def get_integration_test_suite():
    suite = unittest.TestSuite()

    return suite


