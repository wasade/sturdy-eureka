import pandas as pd
import numpy as np
import pandas.testing as pdt
import numpy.testing as npt
import unittest
from se._core import DistanceMatrix
import os


base = os.path.dirname(os.path.abspath(__file__))


class Tests(unittest.TestCase):
    def setUp(self):
        self.dm = DistanceMatrix(os.path.join(base, 'data/test_basic.hdf5'))
        self.mask_dm = DistanceMatrix(os.path.join(base,
                                                   'data/test_basic.hdf5'),
                                      mask=['c', 'a', 'e'])

    def test_getitem(self):
        with self.assertRaisesRegex(KeyError, "Unknown ID: X"):
            self.dm['X']

        exp = pd.DataFrame([['b', 'a', 0.14285714285714285],
                            ['b', 'b', 0.0],
                            ['b', 'c', 0.42857142857142855],
                            ['b', 'd', 0.5714285714285714],
                            ['b', 'e', 0.7142857142857143]],
                           columns=['i', 'j', 'value'])
        obs = self.dm['b']
        pdt.assert_frame_equal(obs, exp)

    def test_init(self):
        self.assertEqual(self.dm.ids, ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(self.dm._index, {'a': 0, 'b': 1, 'c': 2,
                                          'd': 3, 'e': 4})
        self.assertEqual(self.dm._inv_index, {0: 'a', 1: 'b', 2: 'c', 3: 'd',
                                              4: 'e'})
        npt.assert_equal(self.dm._mask, np.array([1, 1, 1, 1, 1], dtype=bool))
        self.assertEqual(self.dm._mask_ids, ['a', 'b', 'c', 'd', 'e'])

        self.assertEqual(self.mask_dm.ids, ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(self.mask_dm._index, {'a': 0, 'b': 1, 'c': 2,
                                               'd': 3, 'e': 4})
        self.assertEqual(self.mask_dm._inv_index, {0: 'a', 1: 'b', 2: 'c',
                                                   3: 'd', 4: 'e'})
        npt.assert_equal(self.mask_dm._mask, np.array([1, 0, 1, 0, 1],
                         dtype=bool))
        self.assertEqual(self.mask_dm._mask_ids, ['a', 'c', 'e'])

    def test_init_bad_mask(self):
        with self.assertRaisesRegex(KeyError, "Mask includes IDs not in the"):
            DistanceMatrix(os.path.join(base, 'data/test_basic.hdf5'),
                           mask=['c', 'a', 'X'])

    def test_within(self):
        exp = pd.DataFrame([['b', 'b', 0.0],
                            ['b', 'd', 0.5714285714285714],
                            ['b', 'e', 0.7142857142857143],
                            ['d', 'b', 0.5714285714285714],
                            ['d', 'd', 0.0],
                            ['d', 'e', 1.0],
                            ['e', 'b', 0.7142857142857143],
                            ['e', 'd', 1.0],
                            ['e', 'e', 0.0]],
                           columns=['i', 'j', 'value'],
                           index=[0, 1, 2, 0, 1, 2, 0, 1, 2])
        obs = self.dm.within(['b', 'd', 'e'])
        pdt.assert_frame_equal(obs, exp)

    def test_between(self):
        exp = pd.DataFrame([['b', 'a', 0.14285714285714285],
                            ['b', 'c', 0.42857142857142855],
                            ['b', 'e', 0.7142857142857143],
                            ['d', 'a', 0.42857142857142855],
                            ['d', 'c', 0.7142857142857143],
                            ['d', 'e', 1.0]],
                           columns=['i', 'j', 'value'],
                           index=[0, 1, 2, 0, 1, 2])
        obs = self.dm.between(['b', 'd'], ['a', 'e', 'c'])
        pdt.assert_frame_equal(obs, exp)

    def test_transpose(self):
        exp = pd.DataFrame([['b', 'a', 0.14285714285714285],
                            ['b', 'c', 0.42857142857142855],
                            ['b', 'e', 0.7142857142857143],
                            ['d', 'a', 0.42857142857142855],
                            ['d', 'c', 0.7142857142857143],
                            ['d', 'e', 1.0]],
                           columns=['i', 'j', 'value'],
                           index=[0, 1, 2, 0, 1, 2])
        t = self.dm.T()
        self.assertTrue(t._transposed)
        obs = t.between(['b', 'd'], ['a', 'e', 'c'])
        pdt.assert_frame_equal(obs, exp)


if __name__ == '__main__':
    unittest.main()
