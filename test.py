import numpy as np
a = np.array([10., 0., 0., 0.])
b = np.array([10., 0., 0., 0.])
print(np.all(a == b))
if np.all(a == b):
    print(np.array([0., 0., 0., 0.]))
    print(a)