import numpy as np
from utility_funcs import guess_params
from wave_funcs import *


def met_hastings_proposal(coords: np.ndarray, rng: np.random.Generator, param_ranges: np.ndarray, noise: np.ndarray, yerr: float, wave_fcn, wave_kwargs: dict) -> tuple[np.ndarray]:
    """
    Updates each walker with a new position vector (parameters) and a ratio of the log probability of
    the new position to the old position of each.

    Args:
        coords (np.ndarray): The positions of the walkers, shape = (n-walkers, n-dimensions).
        rng (np.random.Generator): Used for guessing new parameters.
        kwargs:
        param_ranges: Ranges for guessing parameters when making proposals.
        noise: The noisy data that proposed waves are compared to.
        yerr: 0.0 - 1.0, the percentage of uncertainty in the data.
        wave_fcn: The callable function that will be used for generating waves at each proposal.
        wave_kwargs: Keyword arguments to be passed to the wave_fcn for consistency.

    Returns:
        tuple[np.ndarray]: New position vector for each walker (n-walkers, n-dimensions) and log ratios (n-walkers,).
    """
    new_coords = np.empty_like(coords)      # new position vector for each walker
    log_ratios = np.empty(coords.shape[0])  # a ratio for each walker
    for i in range(coords.shape[0]):        # for each walker
        old_likelihood = gaussian(coords[i], noise, yerr, wave_fcn, wave_kwargs)
        # propose new parameters
        theta = guess_params(param_ranges, rng)
        new_likelihood = gaussian(theta, noise, yerr, wave_fcn, wave_kwargs)
        # log ratio of the new likelihood to the old likelihood
        log_ratios[i] = new_likelihood / old_likelihood
        # new position vector in the parameter space
        new_coords[i] = theta
    return new_coords, log_ratios


def gaussian(theta: np.ndarray, noise: np.ndarray, yerr: float, wave_fcn, wave_kwargs: dict) -> np.float64:
    wave = eval(wave_fcn.__name__)(theta, **wave_kwargs)
    return -0.5 * np.sum(((noise - wave)/yerr) ** 2)


def log_prior(theta: np.ndarray, ranges: np.ndarray):
    if np.all(ranges[:,0] < theta) and np.all(theta < ranges[:,1]):
        return 0.0
    return -np.inf


def log_probability(theta: np.ndarray, noise: np.ndarray, yerr: float, ranges: np.ndarray, wave_fcn, wave_kwargs: dict) -> np.float64:
    lp = log_prior(theta, ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + gaussian(theta, noise, yerr, wave_fcn, wave_kwargs)