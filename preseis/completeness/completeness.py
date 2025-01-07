import numpy as np
import xarray as xr
import scipy.stats as stats
from fast_poibin import PoiBin
import rioxarray


def initiate_stations_and_grid_from_config(config):
    # from config initiate the stations dataset object
    stations_config = config["stations"]
    x = stations_config["x"]
    y = stations_config["y"]
    z = stations_config["z"]
    stations = xr.Dataset(
        data_vars=dict(
            x=("station", x),
            y=("station", y),
            z=("station", z),
            noise_mean=("station", stations_config["noise"]),
            noise_SD=("station", stations_config["noise_sd"]),
        ),
    )

    # Defining the grid of our model space from the config
    grid_config = config["grid"]
    grid = grid_initalize(
        grid_config["spacing"],
        grid_config["x_min"],
        grid_config["x_max"],
        grid_config["y_min"],
        grid_config["y_max"],
        grid_config["z_min"],
        grid_config["z_max"],
    )
    return stations, grid


def grid_initalize(spacing, minx, maxx, miny, maxy, minz, maxz):
    # Define grid and return xarray
    xsmp = np.arange(minx, maxx + spacing, spacing)
    ysmp = np.arange(miny, maxy + spacing, spacing)
    zsmp = np.arange(minz, maxz + spacing, spacing)

    grid = (
        xr.Dataset(
            coords={
                "x": xsmp,
                "y": ysmp,
                "z": zsmp,
            }
        )
        .rio.write_crs("EPSG:28992")
        .rio.write_coordinate_system()
    )
    return grid


def count_probability_samples(combined_probability_array, min_prob, max_prob):
    return len(
        combined_probability_array[
            np.where(
                (combined_probability_array > min_prob)
                & (combined_probability_array < max_prob)
            )
        ]
    )


def ground_motion_model(
    local_magnitude, event_depth, station_depth, epicentral_distance, model_params=None
):
    """
    Calculate peak ground velocity as a function of local magnitude
    following the methodology of Ruigrok, Kruiver, Dost (2023), 2.2.2
    Inputs: local_magnitude
            event_depth, km
            epicentral_distance, km
    Fitting Parameters: c1, c2, c4, c4a, e1, e2, d (defned in table 2.1)
    Outputs: Y_mod, Peak Ground Velocity mm/s
    """

    if model_params is None:
        model_params = dict(
            c1_shallow=-0.2,
            c1_deep=-1.6,
            c2=1.96,
            c4=-3.44,
            c4a=-1.62,
            e1=0.45,
            e2=-0.8,
            d=8.0,
        )

    c1 = np.where(
        station_depth <= 0.2, model_params["c1_shallow"], model_params["c1_deep"]
    )
    c2 = model_params["c2"]
    c4 = model_params["c4"]
    c4a = model_params["c4a"]
    e1 = model_params["e1"]
    e2 = model_params["e2"]
    d = model_params["d"]

    # Effective point-source distance
    h = np.exp(e1 * local_magnitude + e2)
    Rf = np.sqrt(epicentral_distance**2 + event_depth**2 + h**2)

    # Geometrical spreading
    geom_spreading = np.where(
        Rf <= d,
        c4 * np.log(Rf),
        c4 * np.log(d) + c4a * np.log(Rf / d),
    )
    return c1 + c2 * local_magnitude + geom_spreading


def epicentral_distance_calc_grid(grid_x, grid_y, sta_x, sta_y):
    """
    Given an array of the grid's x coordinates and the grids y coordinates, return an xarray of the epicentral distance of each x,y location
    """
    return ((grid_x - sta_x) ** 2 + (grid_y - sta_y) ** 2) ** 0.5


# Log-normal statistical comparison
def probability_of_exceedance(mu1, sigma1, mu2, sigma2, threshold):
    """
    given two independent normal distributions, calculate the probability P(x1 - x2 > threshold)
    where x1 ~ dist1 and x2 ~ dist2 i.e. if threshold =2 then this is the probability that dist1 is twice or greater
    than dist2
    """
    mu = mu1 - mu2
    sigma = np.sqrt(sigma1**2 + sigma2**2)
    poe = stats.norm.sf(threshold, loc=mu, scale=sigma)
    return poe


def calculate_probability_of_NThreshold_occurances(probabilities, threshold):
    """
    given an array of probabilities for events A,B,C ... N calculate the probability that N events or more occur.
    """
    return np.sum(PoiBin(probabilities).pmf[threshold:])


def interpolate_magnitude_of_completeness(
    probabilities, magnitudes, magnitude_of_completeness_percentile
):
    """
    given a list of probabilites for detections at NStations threshold,
    corresponding to each magnitude in self.EQ_magnitudes, calculate the magnitude which corresponds to the
    pre-defined magnitude_of_completeness_percentile. Handle the fact that magnitudes are on a log scale.
    """
    return np.log10(
        np.interp(
            magnitude_of_completeness_percentile,
            probabilities,
            10 ** np.array(magnitudes),
        )
    )
