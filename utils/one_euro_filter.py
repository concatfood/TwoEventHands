import math
import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


# original filter adjusted for numpy arrays
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=np.array([0.0]), min_cutoff=np.array([1.0]), beta=np.array([0.0]),
                 d_cutoff=np.array([1.0])):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = min_cutoff.astype(float) if min_cutoff.shape[0] == x0.shape[0]\
            else np.ones(x0.shape).astype(float)
        self.beta = beta.astype(float) if beta.shape[0] == x0.shape[0] else np.zeros(x0.shape).astype(float)
        self.d_cutoff = d_cutoff.astype(float) if d_cutoff.shape[0] == x0.shape[0] else np.ones(x0.shape).astype(float)
        # Previous values.
        self.x_prev = x0.astype(float)
        self.dx_prev = dx0.astype(float) if dx0.shape[0] == x0.shape[0] else np.zeros(x0.shape).astype(float)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
