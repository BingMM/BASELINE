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
        step_1c_min_window_days=3,
        typical_value_method="paper_mode",
        typical_value_histogram_bins="fd",
        step_1c_plot_diagnostics=False,
        step_1c_diagnostic_time_range=None,
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
        self.step_1c_min_window_days = _validate_odd_window_days(
            step_1c_min_window_days,
            name="step_1c_min_window_days",
        )
        if typical_value_method != "paper_mode":
            raise ValueError("typical_value_method must be 'paper_mode'")
        self.typical_value_method = "paper_mode"
        self.typical_value_histogram_bins = typical_value_histogram_bins
        self.step_1c_plot_diagnostics = bool(step_1c_plot_diagnostics)
        self.step_1c_diagnostic_time_range = _normalize_time_range(
            step_1c_diagnostic_time_range
        )
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
        step_1c_checkpoint_path=None,
        reuse_step_1c_checkpoint=False,
        write_step_1c_checkpoint=False,
    ):
        """Run the available baseline-estimation steps in paper order."""
        self.get_FWHM_stat()
        self.get_QD(
            step_1d_a=step_1d_a,
            step_1d_sigma_days=step_1d_sigma_days,
            step_1d_adaptive_sigma=step_1d_adaptive_sigma,
            step_1d_max_sigma_multiplier=step_1d_max_sigma_multiplier,
            step_1c_checkpoint_path=step_1c_checkpoint_path,
            reuse_step_1c_checkpoint=reuse_step_1c_checkpoint,
            write_step_1c_checkpoint=write_step_1c_checkpoint,
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
        status = []
        diagnostics = []
        
        for day in tqdm(days, total=days.size, desc='Semi-hournly typical values'):
            for b in range(48):
    
                timestamp = day + pd.Timedelta(minutes=30 * b + 15)
                plot_this_diagnostic = self._should_plot_step_1c_diagnostic(timestamp)
                window_days = min(self.step_1c_min_window_days, max_window_days)
                value = np.nan
                sigma = np.nan
                plot_diag = None
                status_value = "too_few_samples"
                max_n_samples = 0
                last_n_samples = 0
                last_window_days = np.nan
                last_mu = np.nan
                last_sigma = np.nan
                last_sigma_weight = np.nan
                last_fwhm = np.nan
                last_fwhm_stat = np.nan

                target_mask = (self.df["day"] == day) & (self.df["bin30"] == b)
                target_n_samples = int(np.isfinite(self.df.loc[target_mask, "x"]).sum())
                if target_n_samples == 0:
                    status_value = "missing_input"
                    results.append((timestamp, np.nan))
                    weight.append((timestamp, 0.0))
                    status.append((timestamp, status_value))
                    diagnostics.append((
                        timestamp,
                        target_n_samples,
                        max_n_samples,
                        last_n_samples,
                        last_window_days,
                        last_mu,
                        last_sigma,
                        last_sigma_weight,
                        last_fwhm,
                        last_fwhm_stat,
                    ))
                    continue
    
                while window_days <= max_window_days:
                    half = window_days // 2
    
                    mask = (
                        (self.df["day"] >= day - pd.Timedelta(days=half)) &
                        (self.df["day"] <= day + pd.Timedelta(days=half)) &
                        (self.df["bin30"] == b)
                    )
    
                    window_df = (
                        self.df.loc[mask, ["day", "residual_step_1"]]
                        .dropna(subset=["residual_step_1"])
                        .copy()
                    )
                    vals = window_df["residual_step_1"].values
                    n_samples = len(vals)
                    max_n_samples = max(max_n_samples, n_samples)
                    
                    if n_samples < 5:
                        window_days += 2
                        continue
    
                    FWHM_stat = np.mean(self.df.loc[mask, "FWHM_stat"].dropna().values)
                    
                    need_diagnostics = plot_this_diagnostic
                    if need_diagnostics:
                        mu, sigma, diag = self._estimate_typical_value(
                            vals,
                            return_diagnostics=True,
                        )
                    else:
                        mu, sigma = self._estimate_typical_value(vals)
                        diag = None
                    if diag is not None:
                        diag["n_samples"] = n_samples
                        diag["FWHM_stat"] = FWHM_stat
                        diag["sigma_over_FWHM_stat"] = (
                            sigma / FWHM_stat
                            if np.isfinite(sigma)
                            and np.isfinite(FWHM_stat)
                            and FWHM_stat > 0
                            else np.nan
                        )
                        _add_step_1c_per_day_diagnostics(
                            diagnostics=diag,
                            window_df=window_df,
                            target_day=day,
                        )
                        plot_diag = diag

                    last_n_samples = n_samples
                    last_window_days = window_days
                    last_mu = mu
                    last_sigma = sigma
                    last_fwhm_stat = FWHM_stat
                    sigma_weight = get_weight_sigma(vals, mu, sigma)
                    last_sigma_weight = sigma_weight

                    if not np.isfinite(mu) or not np.isfinite(sigma):
                        status_value = "typical_value_failed"
                        window_days += 2
                        continue
                    
                    fwhm = 2.355 * sigma
                    last_fwhm = fwhm

                    # Eq. 8 compares Gaussian sigma directly to the empirical
                    # FWHM_stat curve. Despite the name, do not convert sigma
                    # to Gaussian FWHM for the acceptance test.
                    if np.isfinite(sigma) and sigma <= FWHM_stat:
                        value = mu
                        status_value = "ok"
                        break
                    status_value = "fwhm_rejected"
                    window_days += 2

                if not np.isfinite(value):
                    sigma = np.nan
                    window_days = np.nan
                    if max_n_samples == 0:
                        status_value = "no_residual_samples"
                    elif max_n_samples < 5:
                        status_value = "too_few_samples"
    
                plot_window_days = window_days if np.isfinite(window_days) else last_window_days
                self._maybe_plot_step_1c_diagnostic(
                    timestamp,
                    plot_window_days,
                    plot_diag,
                    status_value,
                )
                results.append((timestamp, value))
                denom = last_sigma_weight**2 * window_days / 3
                weight_value = 1 / denom if np.isfinite(denom) and denom > 0 else 0.0
                weight.append((timestamp, weight_value))
                status.append((timestamp, status_value))
                diagnostics.append((
                    timestamp,
                    target_n_samples,
                    max_n_samples,
                    last_n_samples,
                    last_window_days,
                    last_mu,
                    last_sigma,
                    last_sigma_weight,
                    last_fwhm,
                    last_fwhm_stat,
                ))
    
        idx, vals = zip(*results)
        self.QD_step_1c = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()
                
        idx, vals = zip(*weight)
        self.QD_step_1c_w = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()

        idx, vals = zip(*status)
        self.QD_step_1c_status = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()

        columns = [
            "target_n_samples",
            "max_n_samples",
            "last_n_samples",
            "last_window_days",
            "last_mu",
            "last_sigma",
            "last_sigma_weight",
            "last_fwhm",
            "last_fwhm_stat",
        ]
        self.QD_step_1c_diagnostics = (
            pd.DataFrame.from_records(
                diagnostics,
                columns=["datetime", *columns],
            )
            .set_index("datetime")
            .sort_index()
        )
       
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
        if hasattr(self, "QD_step_1c_status"):
            status_values = self.QD_step_1c_status.reindex(self.QD_step_1c.index).values
            missing_input_mask = status_values == "missing_input"
            y_smooth[missing_input_mask] = np.nan
    
        t_full = self.df["datetime"].values.astype('datetime64[s]').astype(float)
        y_interp = cubic_convolution_interpolate(t_nodes, y_smooth, t_full, a=a)
        y_interp[~np.isfinite(self.df["x"].values)] = np.nan
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
        step_1c_checkpoint_path=None,
        reuse_step_1c_checkpoint=False,
        write_step_1c_checkpoint=False,
    ):
        """Compute the full daily baseline term."""
        self.step_1a()
        self.step_1b()
        checkpoint_path = None if step_1c_checkpoint_path is None else Path(step_1c_checkpoint_path)
        if (
            reuse_step_1c_checkpoint
            and checkpoint_path is not None
            and checkpoint_path.exists()
        ):
            self.load_step_1c_checkpoint(checkpoint_path)
        else:
            self.step_1c()
            if write_step_1c_checkpoint and checkpoint_path is not None:
                self.save_step_1c_checkpoint(checkpoint_path)
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
            target_mask = self.df["day"] == day
            target_n_samples = int(np.isfinite(self.df.loc[target_mask, "x_QD"]).sum())

            if target_n_samples == 0:
                timestamp = day + pd.Timedelta(hours=12)
                results.append((timestamp, np.nan, 0.0))
                continue

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
    
                # Eq. 8 compares Gaussian sigma directly to the empirical
                # FWHM_stat curve. Despite the name, do not convert sigma
                # to Gaussian FWHM for the acceptance test.
                if np.isfinite(sigma) and sigma <= FWHM_stat:
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
        y_interp[~np.isfinite(self.df["x_QD"].values)] = np.nan
        self.df["QY"] = y_interp

    def step_2c(self):
        """Step 2c: subtract the yearly trend from the daily-corrected series."""
        self.df['x_QD_QY'] = self.df['x_QD'] - self.df['QY']

    def get_QO(self):
        """Placeholder for the quiet-day residual offset term."""
        1+1

    def save_step_1c_checkpoint(self, path):
        """Persist Step 1c outputs so later runs can resume from Step 1d."""
        required = (
            "QD_step_1c",
            "QD_step_1c_w",
            "QD_step_1c_status",
            "QD_step_1c_diagnostics",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                "Cannot save Step 1c checkpoint before Step 1c is complete: "
                + ", ".join(missing)
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "component": self.component,
            "target_index": self._get_step_1c_target_index(),
            "QD_step_1c": self.QD_step_1c,
            "QD_step_1c_w": self.QD_step_1c_w,
            "QD_step_1c_status": self.QD_step_1c_status,
            "QD_step_1c_diagnostics": self.QD_step_1c_diagnostics,
        }
        pd.to_pickle(payload, path)

    def load_step_1c_checkpoint(self, path):
        """Load previously computed Step 1c outputs for the current record."""
        payload = pd.read_pickle(Path(path))
        if payload.get("component") != self.component:
            raise ValueError(
                "Step 1c checkpoint component mismatch: "
                f"expected {self.component!r}, got {payload.get('component')!r}"
            )

        target_index = payload.get("target_index")
        if not isinstance(target_index, pd.DatetimeIndex):
            target_index = pd.DatetimeIndex(target_index)
        if not target_index.equals(self._get_step_1c_target_index()):
            raise ValueError(
                "Step 1c checkpoint does not match the current time grid/day layout"
            )

        self.QD_step_1c = payload["QD_step_1c"].copy()
        self.QD_step_1c_w = payload["QD_step_1c_w"].copy()
        self.QD_step_1c_status = payload["QD_step_1c_status"].copy()
        self.QD_step_1c_diagnostics = payload["QD_step_1c_diagnostics"].copy()

    def _estimate_typical_value(self, vals, return_diagnostics=False):
        """Evaluate the configured typical-value estimator."""
        return get_typical_value(
            vals,
            method=self.typical_value_method,
            histogram_bins=self.typical_value_histogram_bins,
            return_diagnostics=return_diagnostics,
        )

    def _should_plot_step_1c_diagnostic(self, timestamp):
        """Return whether Step 1c should write a diagnostic plot for this node."""
        if not self.step_1c_plot_diagnostics:
            return False

        if self.step_1c_diagnostic_time_range is None:
            return True

        start_time, stop_time = self.step_1c_diagnostic_time_range
        timestamp = pd.Timestamp(timestamp)
        if start_time is not None and timestamp < start_time:
            return False
        if stop_time is not None and timestamp > stop_time:
            return False
        return True

    def _get_step_1c_target_index(self):
        """Return the expected semi-hourly target timestamps for the current record."""
        days = (
            self.df["datetime"]
            .dt.floor("D")
            .sort_values()
            .unique()
        )
        timestamps = [
            day + pd.Timedelta(minutes=30 * b + 15)
            for day in days
            for b in range(48)
        ]
        return pd.DatetimeIndex(timestamps)

    def _maybe_plot_step_1c_diagnostic(self, timestamp, window_days, diagnostics, status):
        """Write a diagnostic histogram for one semi-hourly mode estimate."""
        if not self._should_plot_step_1c_diagnostic(timestamp):
            return
        if diagnostics is None:
            return

        import matplotlib.pyplot as plt

        plot_dir = self.step_1c_plot_dir / self.component
        plot_dir.mkdir(parents=True, exist_ok=True)

        per_day_counts = diagnostics.get("per_day_counts")
        has_per_day = (
            isinstance(per_day_counts, np.ndarray)
            and per_day_counts.ndim == 2
            and per_day_counts.size > 0
        )

        if has_per_day:
            fig, (ax, ax_days) = plt.subplots(
                2,
                1,
                figsize=(8, 7),
                gridspec_kw={"height_ratios": [3.3, 1.4]},
                sharex=True,
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_days = None

        centers = diagnostics["centers"]
        counts = diagnostics["counts"]
        widths = diagnostics["widths"]
        ax.bar(centers, counts, width=widths, align="center", alpha=0.6, label="Histogram")

        if diagnostics.get("fit_success", False):
            x_fit = diagnostics["x_fit"]
            y_fit = diagnostics["y_fit"]
            ax.plot(x_fit, y_fit, color="tab:red", linewidth=2, label="Gaussian fit")
        else:
            ax.text(
                0.02,
                0.95,
                "Gaussian fit failed",
                transform=ax.transAxes,
                va="top",
            )

        mode_value = diagnostics.get("mode_value")
        if mode_value is not None and np.isfinite(mode_value):
            ax.axvline(mode_value, color="tab:orange", linestyle=":", label="Histogram mode")

        typical_value = diagnostics.get("typical_value")
        if typical_value is not None and np.isfinite(typical_value):
            ax.axvline(typical_value, color="tab:green", linestyle="--", label="Typical value")

        x_min = min(-100.0, np.nanmin(centers - 0.5 * widths))
        x_max = max(100.0, np.nanmax(centers + 0.5 * widths))
        ax.set_xlim(x_min, x_max)
        ax.text(
            0.02,
            0.95,
            _format_step_1c_diagnostic_text(diagnostics),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )

        timestamp_pd = pd.Timestamp(timestamp)
        window_label = f"{int(window_days)} days" if np.isfinite(window_days) else "n/a"
        ax.set_title(
            f"{self.component} {timestamp_pd:%Y-%m-%d %H:%M} "
            f"window={window_label} status={status}"
        )
        ax.set_ylabel("Count")
        ax.legend()

        if ax_days is not None:
            day_offsets = diagnostics.get("per_day_offsets")
            day_labels = diagnostics.get("per_day_labels")
            day_centers = diagnostics.get("per_day_day_centers")
            x_edges = diagnostics.get("edges")
            if (
                isinstance(day_offsets, np.ndarray)
                and isinstance(day_labels, (list, tuple))
                and isinstance(day_centers, np.ndarray)
                and isinstance(x_edges, np.ndarray)
                and day_offsets.size > 0
            ):
                y_edges = _centers_to_edges(day_centers)
                mesh = ax_days.pcolormesh(
                    x_edges,
                    y_edges,
                    per_day_counts,
                    shading="auto",
                    cmap="Blues",
                )
                yticks = day_centers
                yticklabels = [f"{offset:+d}" for offset in day_offsets]
                if len(yticklabels) > 11:
                    step = int(np.ceil(len(yticklabels) / 11))
                    yticks = yticks[::step]
                    yticklabels = yticklabels[::step]
                ax_days.set_yticks(yticks)
                ax_days.set_yticklabels(yticklabels)
                ax_days.axhline(0.0, color="tab:red", linewidth=1.0, alpha=0.7)
                ax_days.set_ylabel("Day offset")
                ax_days.set_xlabel("Residual field [nT]")
            else:
                ax.set_xlabel("Residual field [nT]")
        else:
            ax.set_xlabel("Residual field [nT]")

        filename = f"{self.component}_{timestamp_pd:%Y%m%dT%H%M%S}.png"
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)


def get_typical_value(
    vals,
    method="paper_mode",
    histogram_bins="fd",
    max_iter=5,
    tol=1e-3,
    return_diagnostics=False,
):
    """Estimate the paper-style typical value and spread."""
    if method != "paper_mode":
        raise ValueError("method must be 'paper_mode'")

    mu, sigma, diagnostics = _get_typical_value_paper_mode(vals)

    if return_diagnostics:
        return mu, sigma, diagnostics
    return mu, sigma


def get_weight_sigma(vals, typical_value, sigma_fit, central_fraction=68.0):
    """
    Estimate the uncertainty used for Step 1c weighting.

    The SuperMAG acceptance test uses the fitted Gaussian sigma. For weighting,
    we also account for ambiguous multi-modal distributions by measuring the
    RMS spread of the central sample mass around the selected typical value.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    sigma_candidates = []
    if np.isfinite(sigma_fit) and sigma_fit > 0:
        sigma_candidates.append(float(sigma_fit))

    if vals.size >= 5 and np.isfinite(typical_value):
        distances = np.abs(vals - typical_value)
        radius = np.nanpercentile(distances, central_fraction)
        core = vals[distances <= radius]
        if core.size >= 5:
            sigma_core = np.sqrt(np.mean((core - typical_value) ** 2))
            if np.isfinite(sigma_core) and sigma_core > 0:
                sigma_candidates.append(float(sigma_core))

    if not sigma_candidates:
        return np.nan
    return max(sigma_candidates)


def _get_typical_value_paper_mode(
    vals,
    bin_width=1.0,
    min_samples=15,
):
    """
    Estimate the paper-style typical value using fixed 1 nT histogram bins.

    The returned typical value is the histogram mode from equation (5). The
    Gaussian fit uses the full histogram distribution so the Step 1c acceptance
    test widens broad or ambiguous windows instead of accepting narrow local
    peaks.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.nan, np.nan, None

    counts, edges = _fixed_width_histogram(vals, bin_width=bin_width)
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if counts.size == 0 or np.max(counts) <= 0:
        return np.nan, np.nan, None

    modal_bins = np.flatnonzero(counts == np.max(counts))
    mode_value = float(np.mean(centers[modal_bins]))

    diagnostics = {
        "centers": centers,
        "counts": counts,
        "widths": np.diff(edges),
        "mode_value": mode_value,
        "typical_value": mode_value,
        "fit_success": False,
        "amplitude": np.nan,
        "mu_fit": np.nan,
        "sigma_fit": np.nan,
        "method": "paper_mode",
        "bin_width": float(bin_width),
        "min_samples": int(min_samples),
        "spike_replaced": False,
        "failure_reason": None,
    }

    if vals.size < min_samples:
        diagnostics["failure_reason"] = "too_few_samples_for_histogram"
        return np.nan, np.nan, diagnostics

    if np.all(vals == vals[0]):
        diagnostics["failure_reason"] = "constant_values"
        return np.nan, np.nan, diagnostics

    populated = counts > 0
    if np.sum(populated) < 3:
        diagnostics["failure_reason"] = "too_few_populated_fit_bins"
        return mode_value, np.nan, diagnostics

    second_peak_count = _get_second_peak_count(counts, modal_bins)
    mode_count = float(np.max(counts))
    diagnostics["mode_count"] = mode_count
    diagnostics["second_peak_count"] = second_peak_count
    diagnostics["mode_dominance"] = (
        mode_count / second_peak_count
        if second_peak_count > 0
        else np.inf
    )

    fit_weight_sum = np.sum(counts)
    mu0 = np.sum(centers * counts) / fit_weight_sum
    sigma0 = np.sqrt(np.sum(counts * (centers - mu0) ** 2) / fit_weight_sum)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = bin_width

    amp0 = mode_count
    mu_lower = edges[0]
    mu_upper = edges[-1]
    sigma_upper = max(bin_width, (edges[-1] - edges[0]) / 2)
    sigma0 = float(np.clip(sigma0, bin_width / 2, sigma_upper))
    mu0 = float(np.clip(mu0, mu_lower, mu_upper))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                _gaussian_pdf_shape,
                centers,
                counts,
                p0=(amp0, mu0, sigma0),
                bounds=(
                    [0.0, mu_lower, bin_width / 2],
                    [np.inf, mu_upper, sigma_upper],
                ),
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

    diagnostics["fit_success"] = fit_success
    diagnostics["amplitude"] = float(amplitude) if np.isfinite(amplitude) else np.nan
    diagnostics["mu_fit"] = float(mu_fit) if np.isfinite(mu_fit) else np.nan
    diagnostics["sigma_fit"] = float(sigma_fit) if np.isfinite(sigma_fit) else np.nan
    diagnostics["fit_mu_minus_mode"] = (
        float(mu_fit - mode_value)
        if np.isfinite(mu_fit) and np.isfinite(mode_value)
        else np.nan
    )

    if not fit_success or not np.isfinite(sigma_fit):
        diagnostics["failure_reason"] = "gaussian_fit_failed"
        return mode_value, np.nan, diagnostics

    left = max(int(modal_bins[0]) - 1, 0)
    right = min(int(modal_bins[-1]) + 1, counts.size - 1)
    eq4_local_mean = float(np.mean(counts[left:right + 1]))
    diagnostics["eq4_local_mean"] = eq4_local_mean
    diagnostics["eq4_condition_met"] = (
        bool(amplitude > eq4_local_mean)
        if np.isfinite(amplitude)
        else False
    )

    # The paper wording around eq. (4) is internally inconsistent. This
    # implementation follows the only interpretation that matches the spike-
    # handling explanation: keep the mode when the fitted Gaussian peak height
    # exceeds the local three-bin average around the modal region; otherwise use
    # the Gaussian center to avoid isolated-bin spikes.
    typical_value = mode_value
    if not diagnostics["eq4_condition_met"] and np.isfinite(mu_fit):
        typical_value = float(mu_fit)
        diagnostics["spike_replaced"] = True

    diagnostics["typical_value"] = float(typical_value)
    x_fit = np.linspace(edges[0], edges[-1], 512)
    diagnostics["x_fit"] = x_fit
    diagnostics["y_fit"] = _gaussian_pdf_shape(x_fit, amplitude, mu_fit, sigma_fit)
    diagnostics["edges"] = edges

    return typical_value, sigma_fit, diagnostics


def _fixed_width_histogram(vals, bin_width=1.0):
    """Return counts and stable integer-centered fixed-width histogram edges."""
    vals = np.asarray(vals, dtype=float)
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")

    min_edge = np.floor(np.min(vals) / bin_width) * bin_width - bin_width / 2
    max_edge = np.ceil(np.max(vals) / bin_width) * bin_width + bin_width / 2
    if np.isclose(min_edge, max_edge):
        min_edge -= bin_width
        max_edge += bin_width

    edges = np.arange(min_edge, max_edge + bin_width, bin_width)
    counts, edges = np.histogram(vals, bins=edges)
    return counts, edges


def _add_step_1c_per_day_diagnostics(diagnostics, window_df, target_day):
    """Attach per-day histogram counts for Step 1c diagnostic plots."""
    edges = diagnostics.get("edges")
    if not isinstance(edges, np.ndarray) or edges.ndim != 1 or edges.size < 2:
        return
    if window_df.empty:
        return

    grouped = []
    target_day = pd.Timestamp(target_day).normalize()
    unique_days = np.sort(window_df["day"].unique())
    for day_value in unique_days:
        day_ts = pd.Timestamp(day_value).normalize()
        day_vals = window_df.loc[window_df["day"] == day_value, "residual_step_1"].values
        if day_vals.size == 0:
            continue
        counts, _ = np.histogram(day_vals, bins=edges)
        grouped.append(
            (
                int((day_ts - target_day).days),
                day_ts.strftime("%m-%d"),
                counts.astype(float),
            )
        )

    if not grouped:
        return

    diagnostics["per_day_offsets"] = np.asarray([item[0] for item in grouped], dtype=int)
    diagnostics["per_day_labels"] = [item[1] for item in grouped]
    diagnostics["per_day_counts"] = np.vstack([item[2] for item in grouped])
    diagnostics["per_day_day_centers"] = np.asarray(
        [float(item[0]) for item in grouped],
        dtype=float,
    )


def _centers_to_edges(centers):
    """Convert monotonically increasing centers to edges for pcolormesh."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], midpoints, [last]))


def _get_second_peak_count(counts, modal_bins):
    """Return the highest non-modal bin count for mode-dominance diagnostics."""
    counts = np.asarray(counts, dtype=float)
    non_modal = np.ones(counts.size, dtype=bool)
    non_modal[np.asarray(modal_bins, dtype=int)] = False
    if not np.any(non_modal):
        return 0.0
    return float(np.max(counts[non_modal]))


def _format_step_1c_diagnostic_text(diagnostics):
    """Format compact Step 1c fit diagnostics for histogram plots."""
    sigma = diagnostics.get("sigma_fit", np.nan)
    threshold = diagnostics.get("FWHM_stat", np.nan)
    ratio = diagnostics.get("sigma_over_FWHM_stat", np.nan)
    mode_dominance = diagnostics.get("mode_dominance", np.nan)
    eq4_local_mean = diagnostics.get("eq4_local_mean", np.nan)
    eq4_condition_met = diagnostics.get("eq4_condition_met", None)
    mu_delta = diagnostics.get("fit_mu_minus_mode", np.nan)
    n_samples = diagnostics.get("n_samples", np.nan)

    lines = [
        f"n={_fmt_diag_value(n_samples, decimals=0)}",
        f"sigma={_fmt_diag_value(sigma)}",
        f"threshold={_fmt_diag_value(threshold)}",
        f"sigma/threshold={_fmt_diag_value(ratio, decimals=2)}",
        f"mode dominance={_fmt_diag_value(mode_dominance, decimals=2)}",
        f"eq4 local mean={_fmt_diag_value(eq4_local_mean)}",
        f"eq4 met={eq4_condition_met if eq4_condition_met is not None else 'n/a'}",
        f"fit mu - mode={_fmt_diag_value(mu_delta)}",
    ]

    failure_reason = diagnostics.get("failure_reason")
    if failure_reason:
        lines.append(f"reason={failure_reason}")
    if diagnostics.get("spike_replaced", False):
        lines.append("spike replaced")

    return "\n".join(lines)


def _fmt_diag_value(value, decimals=1):
    """Format finite diagnostic values while keeping NaN explicit."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if not np.isfinite(value):
        return "n/a"
    if decimals == 0:
        return f"{int(round(value))}"
    return f"{value:.{decimals}f}"


def _gaussian_pdf_shape(x, amplitude, mu, sigma):
    """Evaluate a Gaussian on histogram-bin centers."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _normalize_time_range(time_range):
    """Normalize an optional two-item timestamp range."""
    if time_range is None:
        return None

    try:
        start_time, stop_time = time_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "step_1c_diagnostic_time_range must be a two-item "
            "(start_time, stop_time) tuple"
        ) from exc

    start_time = None if start_time is None else pd.Timestamp(start_time)
    stop_time = None if stop_time is None else pd.Timestamp(stop_time)
    if (
        start_time is not None
        and stop_time is not None
        and start_time > stop_time
    ):
        raise ValueError("step_1c_diagnostic_time_range start_time must be <= stop_time")

    return start_time, stop_time


def _validate_odd_window_days(window_days, name):
    """Validate that a symmetric expanding-window size is a positive odd integer."""
    if isinstance(window_days, (np.integer, int)):
        value = int(window_days)
    elif isinstance(window_days, float) and float(window_days).is_integer():
        value = int(window_days)
    else:
        raise ValueError(f"{name} must be a positive odd integer")

    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")
    return value


def cubic_convolution_interpolate(t_nodes, y_nodes, t_full, a=-0.5):
    """Interpolate node values onto a target grid with cubic convolution."""
    t_nodes = np.asarray(t_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    t_full = np.asarray(t_full, dtype=float)

    y_interp = np.full_like(t_full, np.nan, dtype=float)

    finite_t = np.isfinite(t_nodes)
    t_nodes = t_nodes[finite_t]
    y_nodes = y_nodes[finite_t]

    if t_nodes.size == 0:
        return y_interp

    if t_nodes.size == 1:
        if np.isfinite(y_nodes[0]):
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
