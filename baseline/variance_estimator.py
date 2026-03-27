import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import fftconvolve

class VarianceEstimator:
    """Estimate the modified variance used in the yearly-trend weighting."""
    
    def __init__(self, t, N, E, Z, mlat):
        """Store the three rotated field components and station latitude."""
        self.df = pd.DataFrame({'datetime': t, 'N': N, 'E': E, 'Z': Z, 'mlat': mlat})

    def estimate(self):
        """Run the variance, delayed memory, and modified-variance steps."""
        self.get_v()
        self.get_f()
        self.get_d()
        self.get_u()

    def get_v(self):
        """Compute equation 11 on the native cadence with rolling cumulative sums."""
    
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
    
        dt = (self.df["datetime"].iloc[1] - self.df["datetime"].iloc[0]).total_seconds()
        w = int(24 * 3600 / dt)
    
        N = self.df["N"].values
        E = self.df["E"].values
        Z = self.df["Z"].values

        ss_n, count_n = rolling_sum_of_squares(N, w)
        ss_e, count_e = rolling_sum_of_squares(E, w)
        ss_z, count_z = rolling_sum_of_squares(Z, w)

        total_ss = ss_n + ss_e + ss_z
        total_count = count_n + count_e + count_z
        v = np.full_like(total_ss, np.nan, dtype=float)

        valid = total_count > 0
        v[valid] = total_ss[valid] / total_count[valid]
    
        self.df["v"] = v

    def get_v_old(self):
        """Reference implementation of equation 11 using explicit window loops."""
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        
        t = self.df["datetime"].values.astype("datetime64[s]").astype(float)
        N = self.df["N"].values
        E = self.df["E"].values
        Z = self.df["Z"].values
    
        v = np.full_like(N, np.nan, dtype=float)
        window_seconds = 24 * 3600  # 24 hours
    
        for i in tqdm(range(len(t)), desc="Computing v"):
            # indices of points within last 24 hours
            mask = (t >= t[i] - window_seconds) & (t <= t[i])
            if np.sum(mask) < 2:
                continue
            ss_n, count_n = sum_of_squares_ignore_nan(N[mask])
            ss_e, count_e = sum_of_squares_ignore_nan(E[mask])
            ss_z, count_z = sum_of_squares_ignore_nan(Z[mask])
            total_ss = ss_n + ss_e + ss_z
            total_count = count_n + count_e + count_z

            if total_count > 0:
                v[i] = total_ss / total_count
    
        self.df["v"] = v
        
    def get_f(self):
        """Compute the latitude scaling used in the delayed memory term."""
        self.df['fN'] = np.abs(np.cos(self.df['mlat']/180*np.pi))
        self.df['fE'] = 0
        self.df['fZ'] = np.abs(np.sin(self.df['mlat']/180*np.pi))

    def get_d(self):
        """Compute the causal 8-day memory term from equation 12."""
        
        self.df['AN'] = self.df['fN'] * self.df['v']
        self.df['AE'] = self.df['fE'] * self.df['v']
        self.df['AZ'] = self.df['fZ'] * self.df['v']
        
        dt = (self.df["datetime"].iloc[1] - self.df["datetime"].iloc[0]).total_seconds()
        w = int(8 * 24 * 3600 / dt)
    
        AN = self.df["AN"].values.copy()
        AE = self.df["AE"].values.copy()
        AZ = self.df["AZ"].values.copy()

        flag = self.df['mlat'].values > 60
        AN[flag] = 0
        AE[flag] = 0
        AZ[flag] = 0

        lag = np.arange(1, w + 1)
        k = 1 / w
        kernel = k * (1 + np.cos(lag * np.pi * k))

        def smooth(x):
            # "Full" convolution plus a one-sample shift keeps the filter causal.
            valid = np.isfinite(x).astype(float)
            x_filled = np.where(np.isfinite(x), x, 0.0)
            full = fftconvolve(x_filled, kernel, mode="full")
            support = fftconvolve(valid, np.ones_like(kernel), mode="full")
            y = np.concatenate(([0.0], full[:len(x) - 1]))
            support = np.concatenate(([0.0], support[:len(x) - 1]))
            y[support == 0] = np.nan
            return y

        self.df["dN"] = smooth(AN)
        self.df["dE"] = smooth(AE)
        self.df["dZ"] = smooth(AZ)
    
    def get_u(self):
        """Combine the instantaneous and delayed variance terms from equation 13."""
        
        self.df['uN'] = self.df['v'] + self.df['dN']
        self.df['uE'] = self.df['v'] + self.df['dE']
        self.df['uZ'] = self.df['v'] + self.df['dZ']


def rolling_sum_of_squares(x, window_size):
    """Return rolling sum of squared deviations and valid-sample count."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    x_filled = np.where(valid, x, 0.0)

    sum_x = rolling_window_sum(x_filled, window_size)
    sum_x2 = rolling_window_sum(x_filled**2, window_size)
    count = rolling_window_sum(valid.astype(float), window_size)

    ss = np.zeros_like(sum_x, dtype=float)
    nonzero = count > 0
    ss[nonzero] = sum_x2[nonzero] - (sum_x[nonzero] ** 2) / count[nonzero]
    return ss, count


def rolling_window_sum(x, window_size):
    """Compute the trailing window sum for a one-dimensional array."""
    csum = np.cumsum(x)
    csum = np.concatenate(([0.0], csum))

    i = np.arange(len(x))
    i0 = np.maximum(0, i - window_size)
    return csum[i + 1] - csum[i0]


def sum_of_squares_ignore_nan(x):
    """Return sum of squared deviations and the number of finite samples."""
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    count = np.sum(valid)

    if count == 0:
        return 0.0, 0

    finite_x = x[valid]
    ss = np.sum((finite_x - np.mean(finite_x)) ** 2)
    return ss, count
