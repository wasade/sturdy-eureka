import h5py
import numpy as np
from ._spec import format_spec

VLEN_DTYPE = h5py.special_dtype(vlen=bytes)


def _header(h5grp):
    """Set format spec header information"""
    for k, v in format_spec.items():
        h5grp.attrs[k] = v


def _bytes_decoder(x):
    return x.decode('ascii')


def _passthrough_decoder(x):
    return x


def convert_from_ascii(fp, output):
    """Convert a classic ascii distance matrix to binary

    Parameters
    ----------
    fp : a file-like object
        A file-like object of the distance matrix to convert
    output : a h5py File object or group
        Where to write the output

    Raises
    ------
    IOError
        If fp does not appear to be a valid distance matrix
    """

    header = fp.readline().strip()
    if isinstance(header, bytes):
        header = header.decode('ascii')
        decoder = _bytes_decoder
    else:
        decoder = _passthrough_decoder

    _header(output)

    ids = header.split('\t')
    n_ids = len(ids)
    ids_ds = output.create_dataset('order', shape=(n_ids, ),
                                   dtype=VLEN_DTYPE)
    ids_ds[:] = ids

    mat = output.create_dataset('matrix', dtype=float,
                                shape=(n_ids, n_ids), chunks=(1, n_ids))
    for idx, (expected, row) in enumerate(zip(ids, fp)):
        rowid, remainder = decoder(row).split('\t', 1)

        if rowid != expected:
            raise IOError("Does not appear to be a distance matrix")

        try:
            mat[idx, :] = np.fromstring(remainder, count=n_ids, dtype=float,
                                        sep='\t')
        except ValueError:
            raise IOError("Does not appear to be a distance matrix")
