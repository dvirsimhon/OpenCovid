import unittest

from OpenCovid.lib.config import *
from OpenCovid.lib.opencovid import FrameStream


class FrameStreamTestCase(unittest.TestCase):
    def setUp(self):

        self.fs = FrameStream()
        self.fs_none = FrameStream(None)


    def test_next_frame(self):


        pass

    def tearDown(self):
        pass