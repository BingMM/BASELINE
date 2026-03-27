import numpy as np
import pandas as pd
from tqdm import tqdm

from .baseline_estimator import (
    cubic_convolution_interpolate,
    get_typical_value,
    weighted_gaussian_smooth,
)


class CoordinateRotator:
    """
    Rotate geographic-like X/Y/Z measurements into local magnetic N/E/Z.

    The paper only sketches the declination-estimation procedure, so this
    implementation follows the same broad approach:

    1. Estimate an instantaneous declination from the horizontal components.
    2. Determine one typical declination value per day using a 17-day window.
    3. Smooth the daily declination values.
    4. Interpolate the declination back to the native cadence.
    5. Rotate X/Y into N/E and carry Z through unchanged.
    """

    def __init__(self, t, x, y, z, window_days=17, smoothing_sigma_days=30.0):
        """Store the input data and configuration for the rotation step."""
        self.df = pd.DataFrame({"datetime": t, "X": x, "Y": y, "Z": z})
        self.window_days = int(window_days)
        self.smoothing_sigma_days = float(smoothing_sigma_days)

    def rotate(self):
        """Estimate the declination and rotate the horizontal components."""
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.get_declination_raw()
        self.get_declination_daily()
        self.smooth_declination()
        self.interpolate_declination()
        self.apply_rotation()

    def get_declination_raw(self):
        """Estimate an instantaneous declination angle from the horizontal field."""
        x = self.df["X"].values.astype(float)
        y = self.df["Y"].values.astype(float)
        finite = np.isfinite(x) & np.isfinite(y)

        q_raw = np.full_like(x, np.nan, dtype=float)
        q_raw[finite] = np.unwrap(np.arctan2(y[finite], x[finite]))
        self.df["q_raw"] = q_raw

    def get_declination_daily(self):
        """Estimate one typical declination value per day on a noon-centered grid."""
        self.df["day"] = self.df["datetime"].dt.floor("D")
        days = self.df["day"].sort_values().unique()
        max_window_days = _get_max_odd_window_size(days.size)
        window_days = min(self.window_days, max_window_days)

        results = []
        for day in tqdm(days, total=days.size, desc="Declination daily values"):
            half = window_days // 2
            mask = (
                (self.df["day"] >= day - pd.Timedelta(days=half))
                & (self.df["day"] <= day + pd.Timedelta(days=half))
            )

            q_values = self.df.loc[mask, "q_raw"].values
            q_typical, _ = get_typical_value(q_values)
            timestamp = day + pd.Timedelta(hours=12)
            results.append((timestamp, q_typical))

        idx, vals = zip(*results)
        self.q_daily = pd.Series(vals, index=pd.to_datetime(idx)).sort_index()

    def smooth_declination(self):
        """Smooth the daily declination values on the daily node grid."""
        t_nodes = self.q_daily.index.values.astype("datetime64[s]").astype(float)
        y_nodes = self.q_daily.values
        weights = np.ones_like(y_nodes, dtype=float)
        self.q_smooth = weighted_gaussian_smooth(
            t_nodes,
            y_nodes,
            weights,
            sigma_days=self.smoothing_sigma_days,
        )

    def interpolate_declination(self, a=-0.5):
        """Interpolate the smoothed declination back to the full cadence."""
        t_nodes = self.q_daily.index.values.astype("datetime64[s]").astype(float)
        t_full = self.df["datetime"].values.astype("datetime64[s]").astype(float)
        self.df["q"] = cubic_convolution_interpolate(
            t_nodes,
            self.q_smooth,
            t_full,
            a=a,
        )

    def apply_rotation(self):
        """Rotate X/Y into N/E using the estimated declination."""
        q = self.df["q"].values
        x = self.df["X"].values.astype(float)
        y = self.df["Y"].values.astype(float)

        self.df["N"] = x * np.cos(q) + y * np.sin(q)
        self.df["E"] = -x * np.sin(q) + y * np.cos(q)
        self.df["Z"] = self.df["Z"].values.astype(float)

    def get_components(self):
        """Return the rotated N, E, and Z component arrays."""
        return (
            self.df["N"].values,
            self.df["E"].values,
            self.df["Z"].values,
        )


def _get_max_odd_window_size(num_days):
    """Return the widest odd window that fits inside the available number of days."""
    if num_days <= 0:
        return 1
    return num_days if num_days % 2 == 1 else num_days - 1
