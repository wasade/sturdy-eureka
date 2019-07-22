import h5py
import pandas as pd
import numpy as np


class DistanceMatrix(object):
    """A Binary DisSimilarity Matrix"""
    def __init__(self, fp, mask=None, transposed=False):
        """Instantiate access into a Binary DisSimilarity Matrix

        Parameters
        ----------
        fp : h5grp
            The object to instantiate access into
        mask : list or tuple of str, optional
            IDs retain in the matrix
        transposed : bool, optional
            Whether to consider the matrix transposed. Warning: transposed
            operations will be much slower as they will be column order
            access.
        """
        self._transposed = transposed
        self._fp = fp
        self._f = h5py.File(self._fp, 'r', libver='latest')
        self._mat = self._f['matrix']

        ids = self._f['order'][:]
        self.ids = tuple([i.decode('utf-8') for i in ids])
        self._index = {i: idx for idx, i in enumerate(self.ids)}
        self._inv_index = {v: k for k, v in self._index.items()}

        # masking in h5py is most efficient if using boolean arrays
        # instead of fancy slicing as is typical with numpy
        if mask is None:
            mask = set(self.ids)
            # if we don't have a mask, set one that is all the things
            self._mask = np.ones(len(self._index), dtype=bool)
            mask = self.ids
        else:
            mask = set(mask)
            if not mask.issubset(set(self.ids)):
                raise KeyError("Mask includes IDs not in the matrix")

            self._mask = np.asarray([i in mask for i in self._index])

        # store the IDs of the mask, but in matrix order
        self._mask_ids = [i for i in self.ids if i in mask]

    def __getitem__(self, k):
        """Get the values associated with an ID

        Parameters
        ----------
        k : str
            The ID to fetch

        Returns
        -------
        pd.DataFrame
            (i, j, value) representing the source ID ("i"), the target ID ("j")
            and the distance ("value").

        Raises
        ------
        KeyError
            If the ID is not known.
        """
        if k not in self._index:
            raise KeyError("Unknown ID: %s" % k)

        offset = self._index[k]
        fmt = (offset, self._mask)

        if self._transposed:
            fmt = fmt[::-1]

        vals = self._mat[fmt]
        base = pd.DataFrame([(k, j) for k, j in zip([k] * len(vals),
                                                    self._mask_ids)],
                            columns=['i', 'j'])
        base['value'] = vals
        return base

    def within(self, keys):
        """Gather within sample distances for provided IDs

        Parameters
        ----------
        keys : iterable of str
            The IDs to get within distances of.

        Returns
        -------
        pd.DataFrame
            (i, j, value) representing the source ID ("i"), the target ID ("j")
            and the distance ("value").
        """
        new_obj = self.__class__(self._fp, mask=keys)
        return pd.concat([new_obj[k] for k in keys])

    def between(self, grp_i, grp_j):
        """Gather distances between sample groups

        Parameters
        ----------
        grp_i : iterable of str
            The source group to get distances from.
        grp_j : iterable of str
            The target group to get distances to.

        Returns
        -------
        pd.DataFrame
            (i, j, value) representing the source ID ("i"), the target ID ("j")
            and the distance ("value").
        """
        parts = []
        new_obj = self.__class__(self._fp, mask=grp_j)
        for i in grp_i:
            parts.append(new_obj[i])
        return pd.concat(parts)

    def T(self):
        """Access self as transpose"""
        return self.__class__(self._fp, mask=self._mask_ids,
                              transposed=not self._transposed)
