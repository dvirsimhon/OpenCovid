import unittest

from OpenCovid.tests.lib.test_framestream import FrameStreamTestCase
from OpenCovid.tests.lib.test_opencovid import OpenCovidTestCase

def get_unit_test_suite():
    suite = unittest.TestSuite()

    suite.addTest(FrameStreamTestCase('test_next_frame'))

    suite.addTest(OpenCovidTestCase('test_add_analyze_filter'))
    suite.addTest(OpenCovidTestCase('test_reset'))

    return suite

def get_integration_test_suite():
    suite = unittest.TestSuite()

    suite.addTest(OpenCovidTestCase('test_set_frame_src'))

    return suite
