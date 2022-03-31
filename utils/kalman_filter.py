import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanFilterWrapper:
    def __init__(self, x0, dt=1000/30, r=5.0, var=0.1):
        """Initialize the Kalman filter."""
        self.filter = KalmanFilter(dim_x=2 * x0.shape[0], dim_z=x0.shape[0])
        self.filter.x = np.stack((x0, np.zeros_like(x0))).transpose().flatten()
        self.filter.F = np.zeros((2 * x0.shape[0], 2 * x0.shape[0]))
        self.filter.H = np.zeros((x0.shape[0], 2 * x0.shape[0]))
        self.filter.R *= r
        self.filter.Q = Q_discrete_white_noise(2, dt, var, block_size=x0.shape[0])

        for i in range(x0.shape[0]):
            self.filter.F[2 * i, 2 * i] = 1.0
            self.filter.F[2 * i, 2 * i + 1] = 1.0
            self.filter.F[2 * i + 1, 2 * i] = 0.0
            self.filter.F[2 * i + 1, 2 * i + 1] = 1.0

            self.filter.H[i, 2 * i] = 1.0

    def __call__(self, x):
        """Compute the filtered signal."""
        self.filter.predict()
        self.filter.update(x)

        return self.filter.x[::2]

