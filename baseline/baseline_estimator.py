import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm import tqdm

class BaselineEstimator:
    """Estimate the daily and yearly baseline terms for one field component."""
    
    def __init__(
        self,
        t,
        x,
        u,
        mlat,
        component,
        step_1d_a=-0.5,
        step_1d_sigma_days=1 / (3 * 24),
        step_1d_adaptive_sigma=True,
        step_1d_max_sigma_multiplier=6.0,
        step_2b_a=-0.5,
        step_2b_sigma_days=30.0,
        typical_value_method="irls",
        typical_value_histogram_bins="fd",
        step_1c_plot_diagnostics=False,
        step_1c_plot_dir="figures/QD_diag",
    ):
        """Store the component time series, weights, and tunable smoothing parameters."""
        self.df = pd.DataFrame({"datetime": t, "x": x, "u": u, "mlat": mlat})
        self.component = component
        self.step_1d_a = step_1d_a
        self.step_1d_sigma_days = step_1d_sigma_days
        self.step_1d_adaptive_sigma = step_1d_adaptive_sigma
        self.step_1d_max_sigma_multiplier = step_1d_max_sigma_multiplier
        self.step_2b_a = step_2b_a
        self.step_2b_sigma_days = step_2b_sigma_days
        if typical_value_method not in {"irls", "mode"}:
            raise ValueError("typical_value_method must be one of 'irls' or 'mode'")
        self.typical_value_method = typical_value_method
        self.typical_value_histogram_bins = typical_value_histogram_bins
        self.step_1c_plot_diagnostics = bool(step_1c_plot_diagnostics)
        step_1c_plot_dir = Path(step_1c_plot_dir)
        if not step_1c_plot_dir.is_absolute():
            step_1c_plot_dir = Path(__file__).resolve().parents[1] / step_1c_plot_dir
        self.step_1c_plot_dir = step_1c_plot_dir
    
    def get_baseline(
        self,
        step_1d_a=None,
        step_1d_sigma_days=None,
        step_1d_adaptive_sigma=None,
        step_1d_max_sigma_multiplier=None,
        step_2b_a=None,
        step_2b_sigma_days=None,
    ):
        """Run the available baseline-estimation steps in paper order."""
        self.get_FWHM_stat()
        self.get_QD(
            step_1d_a=step_1d_a,
            step_1d_sigma_days=step_1d_sigma_days,
            step_1d_adaptive_sigma=step_1d_adaptive_sigma,
            step_1d_max_sigma_multiplier=step_1d_max_sigma_multiplier,
        )
        self.get_QY(
            step_2b_a=step_2b_a,
            step_2b_sigma_days=step_2b_sigma_days,
        )
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
            lambda values: self._estimate_typical_value(values.values)[0]
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
                final_diag = None
    
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
                    
                    if self.step_1c_plot_diagnostics:
                        mu, sigma, diag = self._estimate_typical_value(
                            vals,
                            return_diagnostics=True,
                        )
                    else:
                        mu, sigma = self._estimate_typical_value(vals)
                        diag = None
                    fallback = (mu, sigma, window_days)
                    
                    fwhm = 2.355 * sigma

                    if np.isfinite(fwhm) and fwhm <= FWHM_stat:
                        value = mu
                        final_diag = diag
                        break
                    window_days += 2

                if not np.isfinite(value):
                    sigma = np.nan
                    window_days = np.nan
    
                timestamp = day + pd.Timedelta(minutes=30 * b + 15)
                self._maybe_plot_step_1c_diagnostic(timestamp, window_days, final_diag)
                results.append((timestamp, value))
                denom = sigma**2 * window_days / 3
                weight_value = 1 / denom if np.isfinite(denom) and denom > 0 else 0.0
                weight.append((timestamp, weight_value))
    
        idx, vals = zip(*results)
        self.QD_step_1c = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
                
        idx, vals = zip(*weight)
        self.QD_step_1c_w = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
       
    def step_1d(
        self,
        a=None,
        sigma_days=None,
        adaptive_sigma=None,
        max_sigma_multiplier=None,
    ):
        """Step 1d: smooth the semi-hourly estimates and resample to full cadence."""
        a = self.step_1d_a if a is None else a
        sigma_days = self.step_1d_sigma_days if sigma_days is None else sigma_days
        adaptive_sigma = (
            self.step_1d_adaptive_sigma if adaptive_sigma is None else adaptive_sigma
        )
        max_sigma_multiplier = (
            self.step_1d_max_sigma_multiplier
            if max_sigma_multiplier is None
            else max_sigma_multiplier
        )

        t_nodes = self.QD_step_1c.index.values.astype('datetime64[s]').astype(float)
        y_nodes = self.QD_step_1c.values
        w_nodes = self.QD_step_1c_w.values
        y_smooth, sig_m = weighted_gaussian_smooth(
            t_nodes,
            y_nodes,
            w_nodes,
            sigma_days=sigma_days,
            adaptive_sigma=adaptive_sigma,
            max_sigma_multiplier=max_sigma_multiplier,
        )
    
        t_full = self.df["datetime"].values.astype('datetime64[s]').astype(float)
        y_interp = cubic_convolution_interpolate(t_nodes, y_smooth, t_full, a=a)
    
        self.y_smooth = y_smooth
        self.sig_m = sig_m
        self.df["QD"] = y_interp

    def step_1e(self):
        """Step 1e: subtract the daily baseline contribution."""
        self.df['x_QD'] = self.df['x'] - self.df['QD']

    def get_QD(
        self,
        step_1d_a=None,
        step_1d_sigma_days=None,
        step_1d_adaptive_sigma=None,
        step_1d_max_sigma_multiplier=None,
    ):
        """Compute the full daily baseline term."""
        self.step_1a()
        self.step_1b()
        self.step_1c()
        self.step_1d(
            a=step_1d_a,
            sigma_days=step_1d_sigma_days,
            adaptive_sigma=step_1d_adaptive_sigma,
            max_sigma_multiplier=step_1d_max_sigma_multiplier,
        )
        self.step_1e()

    def get_QY(self, step_2b_a=None, step_2b_sigma_days=None):
        """Compute the yearly trend term."""
        self.step_2a()
        self.step_2b(a=step_2b_a, sigma_days=step_2b_sigma_days)
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
    
                mu, sigma = self._estimate_typical_value(vals)
                fallback = (mu, u, window_days)
                fwhm = 2.355 * sigma
    
                if np.isfinite(fwhm) and fwhm <= FWHM_stat:
                    value = mu
                    break
                window_days += 2

            if not np.isfinite(value):
                u = np.nan
                window_days = np.nan
    
            timestamp = day + pd.Timedelta(hours=12)
            denom = u * window_days / 17
            weight = 1 / denom if np.isfinite(denom) and denom > 0 else 0.0
            results.append((timestamp, value, weight))
    
        idx, vals, weight = zip(*results)
        self.QD_step_2a = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
        self.QD_step_2a_w = pd.Series(weight, index=pd.to_datetime(idx)).sort_index()

    def step_2b(self, a=None, sigma_days=None):
        """Step 2b: smooth the daily trend estimates and resample them."""
        a = self.step_2b_a if a is None else a
        sigma_days = self.step_2b_sigma_days if sigma_days is None else sigma_days

        t_nodes = self.QD_step_2a.index.values.astype('datetime64[s]').astype(float)
        y_nodes = self.QD_step_2a.values
        w_nodes = self.QD_step_2a_w.values
        y_smooth, _ = weighted_gaussian_smooth(
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

    def _estimate_typical_value(self, vals, return_diagnostics=False):
        """Evaluate the configured typical-value estimator."""
        return get_typical_value(
            vals,
            method=self.typical_value_method,
            histogram_bins=self.typical_value_histogram_bins,
            return_diagnostics=return_diagnostics,
        )

    def _maybe_plot_step_1c_diagnostic(self, timestamp, window_days, diagnostics):
        """Write a diagnostic histogram for one accepted semi-hourly mode estimate."""
        if not self.step_1c_plot_diagnostics:
            return
        if diagnostics is None or not diagnostics.get("fit_success", False):
            return

        import matplotlib.pyplot as plt

        self.step_1c_plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        centers = diagnostics["centers"]
        counts = diagnostics["counts"]
        widths = diagnostics["widths"]
        ax.bar(centers, counts, width=widths, align="center", alpha=0.6, label="Histogram")

        x_fit = diagnostics["x_fit"]
        y_fit = diagnostics["y_fit"]
        ax.plot(x_fit, y_fit, color="tab:red", linewidth=2, label="Gaussian fit")
        ax.axvline(diagnostics["typical_value"], color="tab:green", linestyle="--", label="Typical value")

        timestamp_pd = pd.Timestamp(timestamp)
        ax.set_title(f"{self.component} {timestamp_pd:%Y-%m-%d %H:%M} window={int(window_days)} days")
        ax.set_xlabel("Residual field [nT]")
        ax.set_ylabel("Count")
        ax.legend()

        filename = f"{self.component}_{timestamp_pd:%Y%m%dT%H%M%S}.png"
        fig.savefig(self.step_1c_plot_dir / filename, bbox_inches="tight")
        plt.close(fig)


def get_typical_value(
    vals,
    method="irls",
    histogram_bins="fd",
    max_iter=5,
    tol=1e-3,
    return_diagnostics=False,
):
    """Estimate a typical value and spread using a selectable strategy."""
    if method == "irls":
        mu, sigma = _get_typical_value_irls(vals, max_iter=max_iter, tol=tol)
        diagnostics = None
    elif method == "mode":
        mu, sigma, diagnostics = _get_typical_value_mode(vals, histogram_bins=histogram_bins)
    else:
        raise ValueError("method must be one of 'irls' or 'mode'")

    if return_diagnostics:
        return mu, sigma, diagnostics
    return mu, sigma


def _get_typical_value_irls(vals, max_iter=5, tol=1e-3):
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


def _get_typical_value_mode(vals, histogram_bins="fd"):
    """
    Estimate the typical value from a histogram mode plus a Gaussian fit.

    This follows the paper's stated workflow more closely than the IRLS
    approximation: derive a binned probability distribution, determine the
    mode, and fit a Gaussian whose width is later used in the acceptance test.
    """

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size < 5:
        return np.nan, np.nan, None

    if np.all(vals == vals[0]):
        return vals[0], 0.0, None

    try:
        edges = np.histogram_bin_edges(vals, bins=histogram_bins)
    except ValueError:
        edges = np.histogram_bin_edges(vals, bins="auto")

    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.allclose(edges[0], edges[-1]):
        sigma = np.std(vals)
        return np.median(vals), sigma if np.isfinite(sigma) else np.nan, None

    min_bins = min(64, max(16, int(np.sqrt(vals.size))))
    num_bins = edges.size - 1
    if num_bins < min_bins:
        edges = np.histogram_bin_edges(vals, bins=min_bins)

    counts, edges = np.histogram(vals, bins=edges)
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if counts.size == 0 or np.max(counts) <= 0:
        return np.nan, np.nan, None
    modal_bins = np.flatnonzero(counts == np.max(counts))
    mode_value = np.mean(centers[modal_bins])

    sigma0 = 1.4826 * np.median(np.abs(vals - np.median(vals)))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = np.std(vals)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return mode_value, 0.0, None

    amp0 = float(np.max(counts))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                _gaussian_pdf_shape,
                centers,
                counts,
                p0=(amp0, mode_value, sigma0),
                bounds=([0.0, np.min(vals), 1e-12], [np.inf, np.max(vals), np.inf]),
                maxfev=10000,
            )
            amplitude, mu_fit, sigma_fit = params
            sigma_fit = abs(float(sigma_fit))
            fit_success = True
    except (RuntimeError, ValueError):
        amplitude = np.nan
        mu_fit = np.nan
        sigma_fit = np.nan
        fit_success = False

    if not np.isfinite(sigma_fit):
        sigma_fit = np.std(vals)
    if not np.isfinite(sigma_fit):
        return mode_value, np.nan, None

    left = max(modal_bins[0] - 1, 0)
    right = min(modal_bins[-1] + 1, counts.size - 1)
    local_mean = np.mean(counts[left:right + 1].astype(float))

    # Paper eq. (4): replace an isolated modal spike with the Gaussian center.
    if np.isfinite(amplitude) and amplitude > local_mean and np.isfinite(mu_fit):
        typical_value = mu_fit
    else:
        typical_value = mode_value

    diagnostics = None
    if fit_success:
        x_fit = np.linspace(edges[0], edges[-1], 512)
        diagnostics = {
            "centers": centers,
            "counts": counts,
            "widths": np.diff(edges),
            "mode_value": mode_value,
            "typical_value": typical_value,
            "fit_success": True,
            "amplitude": float(amplitude),
            "mu_fit": float(mu_fit),
            "sigma_fit": float(sigma_fit),
            "x_fit": x_fit,
            "y_fit": _gaussian_pdf_shape(x_fit, amplitude, mu_fit, sigma_fit),
        }

    return typical_value, sigma_fit, diagnostics


def _gaussian_pdf_shape(x, amplitude, mu, sigma):
    """Evaluate a Gaussian on histogram-bin centers."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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


def weighted_gaussian_smooth(
    t_nodes,
    y_nodes,
    w_nodes,
    sigma_days,
    adaptive_sigma=False,
    max_sigma_multiplier=1.0,
):
    """Apply Gaussian temporal smoothing with user-supplied point weights."""
    y_smooth = np.zeros_like(y_nodes)
    sig_m = np.ones_like(y_nodes)
    base_sigma = sigma_days * 86400.0
    finite_weights = w_nodes[np.isfinite(w_nodes) & (w_nodes > 0)]
    global_max_weight = np.max(finite_weights) if finite_weights.size > 0 else np.nan

    sigma_values = np.full_like(y_nodes, base_sigma, dtype=float)
    if adaptive_sigma and np.isfinite(global_max_weight):
        finite = np.isfinite(w_nodes) & (w_nodes > 0)
        relative_weight = np.full_like(w_nodes, np.nan, dtype=float)
        relative_weight[finite] = w_nodes[finite] / global_max_weight

        sigma_multiplier = np.ones_like(w_nodes, dtype=float)
        sigma_multiplier[finite] = 1.0 / np.sqrt(relative_weight[finite])
        sigma_multiplier = np.clip(sigma_multiplier, 1.0, max_sigma_multiplier)

        sigma_values[finite] = base_sigma * sigma_multiplier[finite]
        sig_m = sigma_multiplier

    for i, ti in tqdm(enumerate(t_nodes), total=t_nodes.size, desc='Smoothing'):
        dt = t_nodes - ti
        temporal_weights = np.exp(-0.5 * (dt / sigma_values)**2)
        weights = w_nodes * temporal_weights
        mask = np.isfinite(y_nodes) & np.isfinite(weights) & (weights > 0)

        if np.sum(mask) == 0:
            y_smooth[i] = np.nan
        else:
            weight_sum = np.sum(weights[mask])
            if weight_sum <= 0:
                y_smooth[i] = np.nan
            else:
                y_smooth[i] = np.sum(y_nodes[mask] * weights[mask]) / weight_sum

    return y_smooth, sig_m


def _get_max_odd_window_size(num_days):
    """Return the widest odd window that fits inside the available number of days."""
    if num_days <= 0:
        return 1
    return num_days if num_days % 2 == 1 else num_days - 1
