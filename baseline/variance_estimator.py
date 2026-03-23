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
    
        def rolling_var(x):
            x = np.nan_to_num(x)
    
            S = np.cumsum(x)
            S2 = np.cumsum(x**2)
    
            S = np.concatenate(([0], S))
            S2 = np.concatenate(([0], S2))
    
            i = np.arange(len(x))
            i0 = np.maximum(0, i - w)
    
            n = i - i0 + 1
    
            sum_x = S[i+1] - S[i0]
            sum_x2 = S2[i+1] - S2[i0]
    
            mean = sum_x / n
            var = sum_x2 / n - mean**2
    
            return var
    
        v = (rolling_var(N) + rolling_var(E) + rolling_var(Z)) / 3.0
    
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
            dN = N[mask] - np.mean(N[mask])
            dE = E[mask] - np.mean(E[mask])
            dZ = Z[mask] - np.mean(Z[mask])
            v[i] = (
                np.sum(dN**2) + np.sum(dE**2) + np.sum(dZ**2)
            ) / (3 * np.sum(mask))
    
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
            x = np.nan_to_num(x)
            full = fftconvolve(x, kernel, mode="full")
            return np.concatenate(([0.0], full[:len(x) - 1]))

        self.df["dN"] = smooth(AN)
        self.df["dE"] = smooth(AE)
        self.df["dZ"] = smooth(AZ)
    
    def get_u(self):
        """Combine the instantaneous and delayed variance terms from equation 13."""
        
        self.df['uN'] = self.df['v'] + self.df['dN']
        self.df['uE'] = self.df['v'] + self.df['dE']
        self.df['uZ'] = self.df['v'] + self.df['dZ']
