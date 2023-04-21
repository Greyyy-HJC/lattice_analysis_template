import numpy as np
import h5py as h5

from liblattice.preprocess.resampling import bootstrap

# generate a 3 dimensional x list to test bootstrap function
x = np.random.rand(10, 6, 3)

res = bootstrap(x, 5, axis=1)
print(np.shape(res))
print(res[0])


