import numpy as np
from xtc.runtimes.types.ndarray import NDArray

def test_ndarray():
    alignment = 2 * 1024 * 1024
    NDArray.set_alloc_alignment(alignment)
    I, J = 100, 200
    shape = (100, 200)
    npin = np.arange(I*J).reshape(shape).astype("float32")
    print(f"Init NPArray: dims = {len(npin.shape)}, shape = {npin.shape}, dtype = {npin.dtype}, data = {npin.data}")
    array = NDArray(npin)
    print(f"To NDArray: dims = {array.dims}, shape = {array.shape}, dtype = {array.dtype}, size = {array.size}, data = {array.data}")
    npout = array.numpy()
    print(f"To NPArray: dims = {len(npout.shape)}, shape = {npout.shape}, dtype = {npout.dtype}, data = {npout.data}")
    assert np.array_equal(npin, npout)
    assert array.data % alignment == 0

