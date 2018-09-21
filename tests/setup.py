import unittest


class TestSetup(unittest.TestCase):
    """Test Dependencies from `setup.py`."""

    def test__sys(self):
        import sys
        return self.assertIsNotNone(sys)

    def test__torch(self):
        import tensorflow
        return self.assertIsNotNone(tensorflow)

    def test_tqdm(self):
        import tqdm
        return self.assertIsNotNone(tqdm)

    def test_Pillow(self):
        import PIL
        return self.assertIsNotNone(PIL)


if __name__ == '__main__':
    unittest.main()
