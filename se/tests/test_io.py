import unittest
import io
import numpy as np
import numpy.testing as npt
import h5py
from se._spec import format_spec
from se._io import convert_from_ascii


class ConvertTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_convert_from_ascii(self):
        dat = io.StringIO(simple_dm)
        exp = np.array([[0, 0.1, 0.2],
                        [0.1, 0, 0.3],
                        [0.2, 0.3, 0]])
        obs = h5py.File('foo', driver='core', backing_store=False)
        convert_from_ascii(dat, obs)
        npt.assert_array_equal(obs['matrix'][:],
                               exp)

        self.assertEqual(dict(obs.attrs),
                         format_spec)


simple_dm = """	A	B	C
A	0	0.1	0.2
B	0.1	0	0.3
C	0.2	0.3	0
"""


if __name__ == '__main__':
    unittest.main()
