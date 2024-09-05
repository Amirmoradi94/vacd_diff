from utils import distortion_factors
import numpy as np

mtx, dist = distortion_factors()
np.save('camera_matrix.npy', mtx)
np.save('distortion_coefficients.npy', dist)