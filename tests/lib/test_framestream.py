import unittest
from lib.opencovid import FrameStream
from lib.config import *

class FrameStreamTestCase(unittest.TestCase):
    def setUp(self):

        self.fs = FrameStream()
        self.fs_none = FrameStream(None)


    def test_next_frame(self):
        ret, f = self.fs_none.next_frame()
        self.assertEqual(ret or f is not None,False,"ret should be false, Frame should not exists")
        ret, f = self.fs_not_legal.next_frame()
        self.assertEqual(ret or f is not None, False, "ret should be false, Frame should not exists")

        ret, f = self.fs.next_frame()
        self.assertEqual(ret and f is not None, True, "ret should be true, Frame should exists")


    def tearDown(self):
        pass