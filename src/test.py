import scipy
import numpy as np

print(scipy.__version__)
print(np.__version__)

from scipy.stats import ks_2samp

a = np.array([1,2,3,4], dtype=float)
b = np.array([1,2,3,5], dtype=float)

print(ks_2samp(a, b))