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
        2. Determine one typical declination value per day using a symmetric
           odd-day window (17 days by default).
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
        """
        Estimate the declination and rotate the horizontal components.

        After this method runs, the dataframe contains:

        - `q_raw`: instantaneous declination estimate
        - `q`: smoothed/interpolated declination used for rotation
        - `N`, `E`, `Z`: rotated components
        """
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.get_declination_raw()
        self.get_declination_daily()
        self.smooth_declination()
        self.interpolate_declination()
        self.apply_rotation()

    def get_declination_raw(self):
        """Estimate an instantaneous declination angle from the horizontal `X/Y` field."""
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


class InverseCoordinateRotator:
    """
    Rotate NEZ baseline products back into XYZ using an existing rotator.

    This helper is intentionally separate from both the forward rotator and the
    baseline estimators so that scientific workflows can keep those steps
    explicit.
    """

    def __init__(self, rotator):
        """Store a completed forward rotator and its interpolated declination."""
        if not isinstance(rotator, CoordinateRotator):
            raise TypeError("rotator must be a CoordinateRotator instance")
        if "q" not in rotator.df:
            raise ValueError("rotator must be run before inverse rotation")
        self.rotator = rotator

    def rotate_baselines(self, be_e, be_n, be_u):
        """
        Return baseline products in XYZ from finished `E`, `N`, and `Z` estimators.

        Parameters
        ----------
        be_e, be_n, be_u
            Baseline estimators for the rotated `E`, `N`, and vertical `Z`
            components. The third argument is named `be_u` for convenience in
            calling code, but the estimator itself must still be
            `component="Z"`.
        """
        self._validate_estimator(be_e, "E")
        self._validate_estimator(be_n, "N")
        self._validate_estimator(be_u, "Z")

        q = self.rotator.df["q"].values.astype(float)
        qd_x, qd_y, qd_z = _rotate_nez_to_xyz(
            q,
            be_n.df["QD"].values,
            be_e.df["QD"].values,
            be_u.df["QD"].values,
        )
        qy_x, qy_y, qy_z = _rotate_nez_to_xyz(
            q,
            be_n.df["QY"].values,
            be_e.df["QY"].values,
            be_u.df["QY"].values,
        )
        x_corr, y_corr, z_corr = _rotate_nez_to_xyz(
            q,
            be_n.df["x_QD_QY"].values,
            be_e.df["x_QD_QY"].values,
            be_u.df["x_QD_QY"].values,
        )

        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(self.rotator.df["datetime"]).values,
                "q": q,
                "X": self.rotator.df["X"].values.astype(float),
                "Y": self.rotator.df["Y"].values.astype(float),
                "Z": self.rotator.df["Z"].values.astype(float),
                "QD_X": qd_x,
                "QD_Y": qd_y,
                "QD_Z": qd_z,
                "QY_X": qy_x,
                "QY_Y": qy_y,
                "QY_Z": qy_z,
                "baseline_X": qd_x + qy_x,
                "baseline_Y": qd_y + qy_y,
                "baseline_Z": qd_z + qy_z,
                "X_corr": x_corr,
                "Y_corr": y_corr,
                "Z_corr": z_corr,
            }
        )
        return df

    def _validate_estimator(self, estimator, expected_component):
        """Check that an estimator matches the rotator time grid and component."""
        if getattr(estimator, "component", None) != expected_component:
            raise ValueError(
                f"expected {expected_component!r} baseline estimator, "
                f"got {getattr(estimator, 'component', None)!r}"
            )

        required_columns = {"datetime", "QD", "QY", "x_QD_QY"}
        missing = required_columns.difference(estimator.df.columns)
        if missing:
            raise ValueError(
                "baseline estimator is missing required outputs: "
                + ", ".join(sorted(missing))
            )

        rotator_time = pd.DatetimeIndex(pd.to_datetime(self.rotator.df["datetime"]))
        estimator_time = pd.DatetimeIndex(pd.to_datetime(estimator.df["datetime"]))
        if not rotator_time.equals(estimator_time):
            raise ValueError("rotator and baseline estimator time grids do not match")


def _rotate_nez_to_xyz(q, n, e, z):
    """Rotate `N/E/Z` values back into `X/Y/Z` using declination `q`."""
    q = np.asarray(q, dtype=float)
    n = np.asarray(n, dtype=float)
    e = np.asarray(e, dtype=float)
    z = np.asarray(z, dtype=float)
    x = n * np.cos(q) - e * np.sin(q)
    y = n * np.sin(q) + e * np.cos(q)
    return x, y, z


def _get_max_odd_window_size(num_days):
    """Return the widest odd window that fits inside the available number of days."""
    if num_days <= 0:
        return 1
    return num_days if num_days % 2 == 1 else num_days - 1
