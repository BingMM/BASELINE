import numpy as np
import pandas as pd
from tqdm import tqdm

class BaselineEstimator:
    """Estimate the daily and yearly baseline terms for one field component."""
    
    def __init__(self, t, x, u, mlat, component):
        """Store the component time series and the modified variance weights."""
        self.df = pd.DataFrame({"datetime": t, "x": x, "u": u, "mlat": mlat})
        self.component = component
    
    def get_baseline(self):
        """Run the available baseline-estimation steps in paper order."""
        self.get_FWHM_stat()
        self.get_QD()
        self.get_QY()
        self.get_QO()

    def get_FWHM_stat(self):
        """Evaluate the latitude-dependent FWHM threshold from equation 10."""
        if self.component == 'N':
            self.df['FWHM_stat'] = 30 - .20 * self.df['mlat'] + 17 * np.exp(-(self.df['mlat']-76)**2/100)
        elif self.component == 'E':
            self.df['FWHM_stat'] =  7 + .12 * self.df['mlat'] + 14 * np.exp(-(self.df['mlat']-78)**2/150)
        elif self.component == 'Z':
            self.df['FWHM_stat'] =  5 + .13 * self.df['mlat'] + 19 * np.exp(-(self.df['mlat']-78)**2/150)
        else:
            raise ValueError("component must be one of 'N', 'E', or 'Z'")

    def step_1a(self):
        """Step 1a: estimate one typical value per day on a noon-centered grid."""
    
        day = self.df["datetime"].dt.floor("D")

        daily = self.df.groupby(day)["x"].apply(
            lambda values: get_typical_value(values.values)[0]
        )
    
        daily.index = daily.index + pd.Timedelta(hours=12)
        self.QD_step_1a = daily

    def step_1b(self):
        """Step 1b: resample the daily values to the native cadence."""
        t_nodes = self.QD_step_1a.index.values.astype('datetime64[s]').astype(float)
        y_nodes = self.QD_step_1a.values
        t_full = self.df["datetime"].values.astype('datetime64[s]').astype(float)
        qd_interp = cubic_convolution_interpolate(t_nodes, y_nodes, t_full)
    
        self.df["step_1b"] = qd_interp
        self.df["residual_step_1"] = self.df["x"] - self.df["step_1b"]

    def step_1c(self):
        """Step 1c: estimate 48 semi-hourly values per day with an expanding window."""
        
        self.df["bin30"] = (
            self.df["datetime"].dt.hour * 2 +
            self.df["datetime"].dt.minute // 30
        )
    
        self.df["day"] = self.df["datetime"].dt.floor("D")
        days = self.df["day"].sort_values().unique()
        max_window_days = _get_max_odd_window_size(days.size)
    
        results = []
        weight = []
        
        for day in tqdm(days, total=days.size, desc='Semi-hournly typical values'):
            for b in range(48):
    
                window_days = min(3, max_window_days)
                value = np.nan
                sigma = np.nan
                fallback = None
    
                while window_days <= max_window_days:
                    half = window_days // 2
    
                    mask = (
                        (self.df["day"] >= day - pd.Timedelta(days=half)) &
                        (self.df["day"] <= day + pd.Timedelta(days=half)) &
                        (self.df["bin30"] == b)
                    )
    
                    vals = self.df.loc[mask, "residual_step_1"].dropna().values
                    
                    if len(vals) < 5:
                        window_days += 2
                        continue
    
                    FWHM_stat = np.mean(self.df.loc[mask, "FWHM_stat"].dropna().values)
                    
                    mu, sigma = get_typical_value(vals)
                    fallback = (mu, sigma, window_days)
                    
                    fwhm = 2.355 * sigma

                    if np.isfinite(fwhm) and fwhm <= FWHM_stat:
                        value = mu
                        break
                    window_days += 2

                if not np.isfinite(value) and fallback is not None:
                    # Fall back to the widest finite estimate if the threshold is never met.
                    value, sigma, window_days = fallback
    
                timestamp = day + pd.Timedelta(minutes=30 * b + 15)
                results.append((timestamp, value))
                denom = sigma**2 * window_days / 3
                weight_value = 1 / denom if np.isfinite(denom) and denom > 0 else 0.0
                weight.append((timestamp, weight_value))
    
        idx, vals = zip(*results)
        self.QD_step_1c = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
                
        idx, vals = zip(*weight)
        self.QD_step_1c_w = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
       
    def step_1d(self, a=-0.5, sigma_days=1/48):
        """Step 1d: smooth the semi-hourly estimates and resample to full cadence."""
        t_nodes = self.QD_step_1c.index.values.astype('datetime64[s]').astype(float)
        y_nodes = self.QD_step_1c.values
        w_nodes = self.QD_step_1c_w.values
        y_smooth = weighted_gaussian_smooth(
            t_nodes, y_nodes, w_nodes, sigma_days=sigma_days
        )
    
        t_full = self.df["datetime"].values.astype('datetime64[s]').astype(float)
        y_interp = cubic_convolution_interpolate(t_nodes, y_smooth, t_full, a=a)
    
        self.df["QD"] = y_interp

    def step_1e(self):
        """Step 1e: subtract the daily baseline contribution."""
        self.df['x_QD'] = self.df['x'] - self.df['QD']

    def get_QD(self):
        """Compute the full daily baseline term."""
        self.step_1a()
        self.step_1b()
        self.step_1c()
        self.step_1d()
        self.step_1e()

    def get_QY(self):
        """Compute the yearly trend term."""
        self.step_2a()
        self.step_2b()
        self.step_2c()

    def step_2a(self):
        """Step 2a: estimate daily values from an expanding 17-day window."""
    
        self.df["day"] = self.df["datetime"].dt.floor("D")
    
        days = self.df["day"].sort_values().unique()
        max_window_days = _get_max_odd_window_size(days.size)
    
        results = []
        for day in tqdm(days, total=days.size, desc='step_2a'):
    
            window_days = min(17, max_window_days)
            value = np.nan
            u = np.nan
            fallback = None
    
            while window_days <= max_window_days:
                half = window_days // 2
    
                mask = (
                    (self.df["day"] >= day - pd.Timedelta(days=half)) &
                    (self.df["day"] <= day + pd.Timedelta(days=half))
                )
    
                vals = self.df.loc[mask, "x_QD"].dropna().values
    
                if len(vals) < 10:
                    window_days += 2
                    continue
    
                FWHM_stat = np.mean(self.df.loc[mask, "FWHM_stat"].dropna().values)
                u = np.mean(self.df.loc[mask, 'u'].dropna().values)    
    
                mu, sigma = get_typical_value(vals)
                fallback = (mu, u, window_days)
                fwhm = 2.355 * sigma
    
                if np.isfinite(fwhm) and fwhm <= FWHM_stat:
                    value = mu
                    break
                window_days += 2

            if not np.isfinite(value) and fallback is not None:
                value, u, window_days = fallback
    
            timestamp = day + pd.Timedelta(hours=12)
            denom = u * window_days / 17
            weight = 1 / denom if np.isfinite(denom) and denom > 0 else 0.0
            results.append((timestamp, value, weight))
    
        idx, vals, weight = zip(*results)
        self.QD_step_2a = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
        self.QD_step_2a_w = pd.Series(weight, index=pd.to_datetime(idx)).sort_index()

    def step_2b(self, a=-0.5, sigma_days=30.0):
        """Step 2b: smooth the daily trend estimates and resample them."""
        t_nodes = self.QD_step_2a.index.values.astype('datetime64[s]').astype(float)
        y_nodes = self.QD_step_2a.values
        w_nodes = self.QD_step_2a_w.values
        y_smooth = weighted_gaussian_smooth(
            t_nodes, y_nodes, w_nodes, sigma_days=sigma_days
        )
    
        t_full = self.df["datetime"].values.astype('datetime64[s]').astype(float)
        y_interp = cubic_convolution_interpolate(t_nodes, y_smooth, t_full, a=a)
    
        self.df["QY"] = y_interp

    def step_2c(self):
        """Step 2c: subtract the yearly trend from the daily-corrected series."""
        self.df['x_QD_QY'] = self.df['x_QD'] - self.df['QY']

    def get_QO(self):
        """Placeholder for the quiet-day residual offset term."""
        1+1

def get_typical_value(vals, max_iter=5, tol=1e-3):
    """
    Estimate a robust central value and spread using IRLS Gaussian weights.

    This is a pragmatic replacement for the paper's histogram mode logic in
    equations 4 and 5. It tracks the densest part of a skewed distribution
    without being as sensitive to long tails as a simple mean.
    """

    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]

    if vals.size < 5:
        return np.nan, np.nan

    # Initial guess (robust)
    mu = np.median(vals)
    mad = np.median(np.abs(vals - mu))
    sigma = 1.4826 * mad if mad > 0 else np.std(vals)

    if sigma == 0 or not np.isfinite(sigma):
        return mu, 0.0

    for _ in range(max_iter):
        diff = vals - mu
        w = np.exp(-0.5 * (diff / sigma)**2)

        mu_new = np.sum(w * vals) / np.sum(w)
        sigma_new = np.sqrt(np.sum(w * (vals - mu_new)**2) / np.sum(w))

        if np.abs(mu_new - mu) < tol:
            mu, sigma = mu_new, sigma_new
            break

        mu, sigma = mu_new, sigma_new

    return mu, sigma


def cubic_convolution_interpolate(t_nodes, y_nodes, t_full, a=-0.5):
    """Interpolate node values onto a target grid with cubic convolution."""
    t_nodes = np.asarray(t_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    t_full = np.asarray(t_full, dtype=float)

    y_interp = np.full_like(t_full, np.nan, dtype=float)

    finite = np.isfinite(t_nodes) & np.isfinite(y_nodes)
    t_nodes = t_nodes[finite]
    y_nodes = y_nodes[finite]

    if t_nodes.size == 0:
        return y_interp

    if t_nodes.size == 1:
        y_interp[:] = y_nodes[0]
        return y_interp

    dt_nodes = np.median(np.diff(t_nodes))
    if not np.isfinite(dt_nodes) or dt_nodes <= 0:
        return y_interp

    for i, t in tqdm(enumerate(t_full), total=t_full.size, desc='Cubic conv interpolation'):
        u = (t - t_nodes[0]) / dt_nodes
        k = int(np.floor(u))

        val = 0.0
        wsum = 0.0

        for j in range(k - 1, k + 3):
            if 0 <= j < len(y_nodes) and np.isfinite(y_nodes[j]):
                wj = cubic_convolution_weight(u - j, a=a)
                val += y_nodes[j] * wj
                wsum += wj

        if wsum > 0:
            y_interp[i] = val / wsum

    return y_interp


def cubic_convolution_weight(x, a=-0.5):
    """Evaluate the cubic-convolution kernel at one or more offsets."""
    ax = np.abs(np.asarray(x, dtype=float))
    w = np.zeros_like(ax)

    m1 = ax < 1
    m2 = (ax >= 1) & (ax < 2)

    w[m1] = (a + 2) * ax[m1]**3 - (a + 3) * ax[m1]**2 + 1
    w[m2] = a * ax[m2]**3 - 5 * a * ax[m2]**2 + 8 * a * ax[m2] - 4 * a

    return float(w) if np.ndim(w) == 0 else w


def weighted_gaussian_smooth(t_nodes, y_nodes, w_nodes, sigma_days):
    """Apply Gaussian temporal smoothing with user-supplied point weights."""
    y_smooth = np.zeros_like(y_nodes)
    sigma = sigma_days * 86400.0

    for i, ti in tqdm(enumerate(t_nodes), total=t_nodes.size, desc='Smoothing'):
        dt = t_nodes - ti
        temporal_weights = np.exp(-0.5 * (dt / sigma)**2)
        weights = w_nodes * temporal_weights
        mask = np.isfinite(y_nodes) & np.isfinite(weights)

        if np.sum(mask) == 0:
            y_smooth[i] = np.nan
        else:
            y_smooth[i] = np.sum(y_nodes[mask] * weights[mask]) / np.sum(weights[mask])

    return y_smooth


def _get_max_odd_window_size(num_days):
    """Return the widest odd window that fits inside the available number of days."""
    if num_days <= 0:
        return 1
    return num_days if num_days % 2 == 1 else num_days - 1
